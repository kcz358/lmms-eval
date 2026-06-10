import time
import os
from typing import List, Optional, Union

import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from tqdm import tqdm
from transformers import AutoTokenizer

from lmms_eval import utils
from lmms_eval.api.instance import GenerationResult, Instance, TokenCounts
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.protocol import ChatMessages

try:
    import lmms_engine.models  # noqa: F401
    from lmms_engine.models import AeroRealtimeForConditionalGeneration, AeroRealtimeProcessor
    from lmms_engine.models.qwen2_audio import apply_liger_kernel_to_qwen2_audio
    from lmms_engine.models.qwen3_vl.qwen3_vl_ops import patch_embed_forward
    from transformers.models.qwen3_vl import modeling_qwen3_vl
except Exception as e:
    eval_logger.error(f"Failed to import AeroRealtime dependencies. {e}")
    eval_logger.error("Please install lmms-engine and ensure it is importable to use aero_realtime_chat")
    AeroRealtimeForConditionalGeneration = None
    AeroRealtimeProcessor = None
    apply_liger_kernel_to_qwen2_audio = None
    patch_embed_forward = None
    modeling_qwen3_vl = None

try:
    import librosa
except Exception:
    librosa = None

try:
    from qwen_vl_utils import fetch_video
    from qwen_vl_utils import vision_process as qwen_vision_process
except Exception:
    fetch_video = None
    qwen_vision_process = None


@register_model("aero_realtime_chat")
class AeroRealtimeChat(lmms):
    is_simple = False

    DEFAULT_GEN_KWARGS = {
        "max_new_tokens": 128,
        "temperature": 0.0,
        "top_p": None,
        "num_beams": 1,
    }

    def __init__(
        self,
        pretrained: str,
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache: bool = True,
        attn_implementation: Optional[str] = None,
        max_num_frames: int = 32,
        fps: Optional[float] = None,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        trust_remote_code: Optional[bool] = False,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        if AeroRealtimeForConditionalGeneration is None:
            raise ImportError("AeroRealtime dependencies are not available")
        if apply_liger_kernel_to_qwen2_audio is None:
            raise ImportError("Qwen2-Audio patch is not available")
        if patch_embed_forward is None or modeling_qwen3_vl is None:
            raise ImportError("Qwen3-VL patch embed patch is not available")
        if fetch_video is None or qwen_vision_process is None:
            raise ImportError("qwen_vl_utils is required for AeroRealtime video loading")
        if librosa is None:
            raise ImportError("librosa is required for AeroRealtime audio extraction")

        valid_attn_implementations = [None, "flash_attention_2", "sdpa", "eager"]
        if attn_implementation not in valid_attn_implementations:
            raise ValueError(f"attn_implementation must be one of {valid_attn_implementations}, got {attn_implementation}")

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        model_kwargs = {
            "dtype": "bfloat16",
            "device_map": self.device_map,
        }
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        apply_liger_kernel_to_qwen2_audio()
        modeling_qwen3_vl.Qwen3VLVisionPatchEmbed.forward = patch_embed_forward
        self._model = AeroRealtimeForConditionalGeneration.from_pretrained(pretrained, trust_remote_code=trust_remote_code, **model_kwargs).eval()
        self.processor = AeroRealtimeProcessor.from_pretrained(pretrained, trust_remote_code=trust_remote_code)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=trust_remote_code)
        self.processor.tokenizer.padding_side = "left"

        self.max_num_frames = max_num_frames
        self.fps = fps
        self.system_prompt = system_prompt
        self._config = self.model.config
        self._max_length = 2048
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        return self._model

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests):
        raise NotImplementedError("Loglikelihood is not implemented for AeroRealtimeChat")

    def flatten(self, items):
        flattened = []
        for group in items:
            for item in group:
                flattened.append(item)
        return flattened

    def _build_video_kwargs(self):
        video_kwargs = {
            "min_frames": 1,
            "max_frames": self.max_num_frames,
        }
        if self.fps is not None:
            video_kwargs["fps"] = self.fps
        return video_kwargs

    def _load_video_with_metadata(self, video_path: str):
        video_dict = {
            "type": "video",
            "video": f"file://{video_path}",
            **self._build_video_kwargs(),
        }

        os.environ["FORCE_QWENVL_VIDEO_READER"] = "decord"
        qwen_vision_process.FORCE_QWENVL_VIDEO_READER = "decord"
        qwen_vision_process.get_video_reader_backend.cache_clear()
        video_inputs, _ = fetch_video(
            video_dict,
            return_video_sample_fps=True,
            return_video_metadata=True,
        )

        frames, video_metadata = video_inputs
        return frames.numpy(), video_metadata

    def _extract_audio_from_video(self, video_path: str):
        target_sr = self.processor.feature_extractor.sampling_rate
        try:
            audio, _ = librosa.load(video_path, sr=target_sr, mono=True)
            return audio.astype(np.float32)
        except Exception:
            eval_logger.warning(f"Failed to extract audio from {video_path}, using 1s silence fallback")
            return np.zeros(target_sr, dtype=np.float32)

    def _build_generate_kwargs(self, gen_kwargs):
        current = {**self.DEFAULT_GEN_KWARGS, **gen_kwargs}
        if current.get("temperature", 0) > 0:
            current["do_sample"] = True
        else:
            current["do_sample"] = False
            current["temperature"] = None
            current["top_p"] = None
            current.pop("top_k", None)

        generate_kwargs = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "max_new_tokens": current["max_new_tokens"],
            "use_cache": self.use_cache,
            "do_sample": current["do_sample"],
        }
        for key in ("temperature", "top_p", "top_k", "num_beams"):
            val = current.get(key)
            if val is not None:
                generate_kwargs[key] = val
        return generate_kwargs

    def _to_hf_messages(self, messages: ChatMessages):
        hf_messages = []
        if self.system_prompt:
            hf_messages.append({"role": "system", "content": self.system_prompt})
        hf_messages.extend(messages.to_hf_messages())
        return hf_messages

    def _extract_inputs(self, chat_messages: List[ChatMessages]):
        hf_messages = [self._to_hf_messages(message) for message in chat_messages]
        texts = self.processor.apply_chat_template(hf_messages, tokenize=False, add_generation_prompt=True)

        images = []
        loaded_videos = []
        video_metadatas = []
        audios = []
        for message in chat_messages:
            image_items, video_items, _ = message.extract_media()
            images.extend(image_items)
            for video_path in video_items:
                video_frames, video_metadata = self._load_video_with_metadata(video_path)
                loaded_videos.append(video_frames)
                video_metadatas.append(video_metadata)
                audios.append(self._extract_audio_from_video(video_path))

        processor_kwargs = {
            "text": texts,
            "images": images or None,
            "videos": loaded_videos or None,
            "video_metadata": video_metadatas or None,
            "videos_kwargs": {"do_sample_frames": False},
            "audio": audios or None,
            "sampling_rate": self.processor.feature_extractor.sampling_rate,
            "return_tensors": "pt",
        }

        if self.batch_size > 1:
            processor_kwargs["padding"] = True

        inputs = self.processor(**processor_kwargs)
        return texts, inputs

    def generate_until(self, requests: List[Instance]) -> List[GenerationResult]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        total_elapsed_time = 0
        total_tokens = 0

        for request in requests:
            _, doc_to_messages, gen_kwargs, doc_id, task, split = request.arguments
            messages = doc_to_messages(self.task_dict[task][split][doc_id])
            chat_messages = ChatMessages(**{"messages": messages})

            texts, inputs = self._extract_inputs([chat_messages])

            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            generate_kwargs = self._build_generate_kwargs(gen_kwargs)

            start_time = time.time()
            cont = self.model.generate(**inputs, **generate_kwargs)
            end_time = time.time()

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            total_elapsed_time += end_time - start_time
            total_tokens += sum(len(ids) for ids in generated_ids_trimmed)

            for i, (ans, context) in enumerate(zip(answers, texts)):
                res.append(GenerationResult(text=ans, token_counts=TokenCounts(output_tokens=len(generated_ids_trimmed[i]))))
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                eval_logger.debug(f"Question: {context}")
                eval_logger.debug(f"Model Response: {ans}")

            pbar.update(1)

        avg_speed = total_tokens / total_elapsed_time if total_elapsed_time > 0 else 0
        log_metrics(
            total_gen_tokens=total_tokens,
            total_elapsed_time=total_elapsed_time,
            avg_speed=avg_speed,
            additional_metrics={"rank": self.rank},
        )

        pbar.close()
        return res

    def generate_until_multi_round(self, requests):
        raise NotImplementedError("TODO: Implement multi-round generation")
