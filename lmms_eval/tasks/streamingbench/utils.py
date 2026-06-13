"""StreamingBench helper utilities.

Implements the path resolver from ``(question_id, config_name, task_type)``
to a local video file. Video archives are downloaded and namespaced under
``$HF_HOME/streamingbench/<subdir>/sample_<N>/video.mp4``.
"""

from __future__ import annotations

import ast
import glob
import os
import re
import zipfile
from collections import defaultdict
from functools import lru_cache
from typing import Optional

from filelock import FileLock
from huggingface_hub import snapshot_download
from loguru import logger as eval_logger


# Per-config base subdirectory (only used when the config maps 1:1 to a video
# archive). Contextual and Omni configs have heterogeneous task_types; for
# them we fall back to a per-task_type subdirectory chosen by ``task_type``.
CFG_TO_SUBDIR = {
    "Real_Time_Visual_Understanding": "Real-Time Visual Understanding",
    "Sequential_Question_Answering": "Sequential Question Answering",
    "Proactive_Output": "Proactive Output",
}

# Maps task_type strings (as they appear in csv rows) to the zip-archive
# folder name on disk after extraction. The "Misleading Context Recognition"
# row label maps to the "Misleading Context Understanding" folder name (yes,
# the dataset really has this mismatch).
TASK_TYPE_TO_SUBDIR = {
    "Misleading Context Recognition": "Misleading Context Understanding",
    "Anomaly Context Understanding": "Anomaly Context Understanding",
    "Emotion Recognition": "Emotion Recognition",
    "Multimodal Alignment": "Multimodal Alignment",
    "Scene Understanding": "Scene Understanding",
    "Source Discrimination": "Source Discrimination",
}


DATASET_REPO_ID = "mjuicem/StreamingBench"

_EXPECTED_SUBDIRS = set(CFG_TO_SUBDIR.values()) | set(TASK_TYPE_TO_SUBDIR.values())


def _streamingbench_root(hf_home: Optional[str] = None) -> str:
    if hf_home is None:
        hf_home = os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface"))
    return os.path.join(hf_home, "streamingbench")


def _videos_ready(root: str) -> bool:
    """Strict readiness check: every expected subdir has at least one
    ``sample_*/video.mp4`` under it."""
    if not os.path.isdir(root):
        return False
    for sub in _EXPECTED_SUBDIRS:
        cand = os.path.join(root, sub)
        if not os.path.isdir(cand):
            return False
        if not glob.glob(os.path.join(cand, "sample_*", "video.mp4")):
            return False
    return True


def _zip_subdir(zip_name: str) -> str:
    """Strip ``.zip`` and trailing ``_<a>-<b>`` part-range suffix."""
    base = os.path.splitext(zip_name)[0]
    return re.sub(r"_\d+-\d+$", "", base)


def _extract_zip_namespaced(zip_path: str, out_root: str) -> None:
    subdir = _zip_subdir(os.path.basename(zip_path))
    out_dir = os.path.join(out_root, subdir)
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = info.filename
            if name.startswith("__MACOSX/") or name.endswith(".DS_Store"):
                continue
            target = os.path.join(out_dir, name)
            if os.path.exists(target):
                continue
            zf.extract(info, out_dir)


@lru_cache(maxsize=1)
def _ensure_videos() -> str:
    """Download + unpack streamingbench video zips on first use.

    Returns the root path. Idempotent and multi-process safe via FileLock.
    Lazy: only triggers a download if videos aren't already on disk.
    """
    root = _streamingbench_root()
    if _videos_ready(root):
        return root

    os.makedirs(root, exist_ok=True)
    lock_path = os.path.join(root, ".streamingbench.lock")
    with FileLock(lock_path):
        if _videos_ready(root):
            return root
        eval_logger.info(f"[streamingbench] snapshot_download {DATASET_REPO_ID} (zips only)")
        snap = snapshot_download(
            DATASET_REPO_ID,
            repo_type="dataset",
            allow_patterns=["*.zip"],
        )
        zips = sorted(glob.glob(os.path.join(snap, "*.zip")))
        if not zips:
            raise RuntimeError(
                f"[streamingbench] no zips found under {snap}; check dataset repo access"
            )
        eval_logger.info(f"[streamingbench] extracting {len(zips)} zips -> {root}")
        for zp in zips:
            eval_logger.info(f"[streamingbench]   extract {os.path.basename(zp)}")
            _extract_zip_namespaced(zp, root)
        if not _videos_ready(root):
            raise RuntimeError(
                f"[streamingbench] extraction completed but no video.mp4 found under {root}"
            )
        eval_logger.info(f"[streamingbench] ready at {root}")
        return root


def _extract_sample_number(question_id: str) -> int:
    """Pull the sample number out of question_id.

    Examples:
        "Real-Time Visual Understanding_sample_3_2"  -> 3
        "Proactive Output_sample_1_1"                -> 1
        "Proactive Output_42_1"                      -> 42  (50-split style)
    """
    m = re.search(r"sample_(\d+)_(\d+)$", question_id)
    if m:
        return int(m.group(1))
    m = re.search(r"_(\d+)_(\d+)$", question_id)
    if m:
        return int(m.group(1))
    raise ValueError(f"Cannot extract sample number from question_id={question_id!r}")


def _config_subdir(config_name: str, task_type: Optional[str]) -> str:
    if config_name in ("Contextual_Understanding", "Omni_Source_Understanding"):
        if task_type is None or task_type not in TASK_TYPE_TO_SUBDIR:
            raise ValueError(
                f"{config_name} requires task_type from {list(TASK_TYPE_TO_SUBDIR)}, got {task_type!r}"
            )
        return TASK_TYPE_TO_SUBDIR[task_type]
    if config_name not in CFG_TO_SUBDIR:
        raise ValueError(f"Unknown streamingbench config: {config_name!r}")
    return CFG_TO_SUBDIR[config_name]


def resolve_video_path(
    *,
    question_id: str,
    config_name: str,
    hf_home: Optional[str] = None,
    task_type: Optional[str] = None,
) -> str:
    """Return absolute path to the source video for a streamingbench row.

    Raises FileNotFoundError when the expected mp4 is missing — caller should
    ensure videos have been pre-downloaded and unpacked.
    """
    if hf_home is None:
        _ensure_videos()
        hf_home = os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface"))
    subdir = _config_subdir(config_name, task_type)
    sample_n = _extract_sample_number(question_id)
    path = os.path.join(hf_home, "streamingbench", subdir, f"sample_{sample_n}", "video.mp4")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"streamingbench video not found at {path}; "
            f"ensure the dataset's video archives are downloaded and unpacked"
        )
    return path


STATIC_PROMPT_TEMPLATE = (
    "You are an advanced video question-answering AI assistant. You have been "
    "provided with some frames from the video and a multiple-choice question "
    "related to the video. Your task is to carefully analyze the video and "
    "provide the best answer to question, choosing from the four options "
    "provided. Respond with only the letter (A, B, C, or D) of the correct "
    "option.\n\nQuestion: {question}\n\nOptions:\n{a}\n{b}\n{c}\n{d}\n\n"
    "The best option is:"
)


def parse_options(raw: str) -> list[str]:
    options = ast.literal_eval(raw)
    if not options[0].startswith("A."):
        options = [f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(options)]
    return list(options)


def parse_timestamp(ts: str) -> int:
    """``"00:03:10" -> 190`` seconds. Accepts H:M:S or M:S or S."""
    parts = ts.split(":")
    secs = 0
    for chunk in parts:
        secs = secs * 60 + int(chunk)
    return secs


def compute_clip_range(*, ask_time: int, context_time: int) -> tuple[int, int]:
    """Return ``(start, end)`` seconds matching the official ``context_time``
    semantics: ``context_time<=0`` -> ``[0, ask_time]``;
    otherwise ``[max(0, ask_time - context_time), ask_time]``."""
    if context_time <= 0:
        return 0, ask_time
    return max(0, ask_time - context_time), ask_time


def build_static_prompt(*, question: str, options: list[str]) -> str:
    a, b, c, d = options
    return STATIC_PROMPT_TEMPLATE.format(question=question, a=a, b=b, c=c, d=d)


def streamingbench_static_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    config_name = lmms_eval_specific_kwargs["config_name"]
    context_time = int(lmms_eval_specific_kwargs.get("context_time", 60))

    video_path = resolve_video_path(
        question_id=doc["question_id"],
        config_name=config_name,
        task_type=doc.get("task_type"),
    )
    ask_time = parse_timestamp(doc["time_stamp"])
    start, end = compute_clip_range(ask_time=ask_time, context_time=context_time)
    options = parse_options(doc["options"])
    prompt = build_static_prompt(question=doc["question"], options=options)

    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "url": video_path,
                    "start_time": float(start),
                    "end_time": float(end),
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]


def streamingbench_static_process_results(doc, results):
    response = results[0] if results else ""
    pred_letter = (response.strip()[:1] if response else "").upper()
    gold = doc["answer"].strip().upper()
    correct = int(pred_letter == gold)
    return {"acc": {"correct": correct, "category": doc.get("task_type", "")}}


def streamingbench_static_aggregate(items, args=None):
    """``items`` is a list of dicts ``{"correct": 0/1, "category": str}``.
    Returns ``{<category>: pct, ..., "average": macro-avg-pct}``."""
    buckets: dict[str, list[int]] = defaultdict(list)
    for it in items:
        buckets[it["category"]].append(int(it["correct"]))
    out: dict[str, float] = {}
    for cat, vals in buckets.items():
        out[cat] = 100.0 * sum(vals) / max(len(vals), 1)
    out["average"] = sum(out.values()) / max(len(out), 1) if out else 0.0
    return out


SEQUENTIAL_PROMPT_TEMPLATE = (
    "You are an advanced video question-answering AI assistant. You have been "
    "provided with some frames from the video and a series of multiple-choice "
    "questions related to the video. The earlier questions and their gold "
    "answers are provided as context.\n\n"
    "{history}\n"
    "Now please answer the current question by responding with only the "
    "letter (A, B, C, or D) of the correct option.\n\n"
    "Question: {question}\n\nOptions:\n{a}\n{b}\n{c}\n{d}\n\n"
    "The best option is:"
)


_SEQ_SUFFIX = re.compile(r"_(\d+)_(\d+)$")


def _sequential_sample_and_sub(question_id: str) -> tuple[int, int]:
    m = _SEQ_SUFFIX.search(question_id)
    if m is None:
        raise ValueError(f"Cannot parse sequential question_id: {question_id!r}")
    return int(m.group(1)), int(m.group(2))


def _render_prior_history(prior_rows: list[dict]) -> str:
    if not prior_rows:
        return ""
    lines = [
        f"At timestamp {row['time_stamp']}, Q: {row['question']}; A: {row['answer']};"
        for row in prior_rows
    ]
    return "Prior Q&A history:\n" + "\n".join(lines)


def build_sequential_prompt(*, question: str, options: list[str], history: str) -> str:
    a, b, c, d = options
    history_block = history if history else "Prior Q&A history: (none)"
    return SEQUENTIAL_PROMPT_TEMPLATE.format(
        history=history_block, question=question, a=a, b=b, c=c, d=d
    )


def streamingbench_sequential_process_docs(dataset):
    """Annotate each row with ``_prior_history``: a serialized string of all
    earlier-turn Q&As within the same ``sample_N``."""
    rows = list(dataset)
    by_sample: dict[int, list[tuple[int, dict]]] = defaultdict(list)
    for row in rows:
        try:
            sample_n, sub = _sequential_sample_and_sub(row["question_id"])
        except ValueError:
            continue
        by_sample[sample_n].append((sub, row))
    for sample_n in by_sample:
        by_sample[sample_n].sort(key=lambda pair: pair[0])

    histories: dict[str, str] = {}
    for sample_n, pairs in by_sample.items():
        for idx, (sub, row) in enumerate(pairs):
            priors = [r for _, r in pairs[:idx]]
            histories[row["question_id"]] = _render_prior_history(priors)

    def _attach(example):
        example["_prior_history"] = histories.get(example["question_id"], "")
        return example

    return dataset.map(_attach)


def streamingbench_sequential_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    context_time = int(lmms_eval_specific_kwargs.get("context_time", 60))

    video_path = resolve_video_path(
        question_id=doc["question_id"],
        config_name="Sequential_Question_Answering",
        task_type=doc.get("task_type"),
    )
    ask_time = parse_timestamp(doc["time_stamp"])
    start, end = compute_clip_range(ask_time=ask_time, context_time=context_time)
    options = parse_options(doc["options"])
    history = doc.get("_prior_history", "")
    prompt = build_sequential_prompt(question=doc["question"], options=options, history=history)

    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "url": video_path,
                    "start_time": float(start),
                    "end_time": float(end),
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]


PROACTIVE_YESNO_PROMPT = (
    "You are watching a streaming video. The user has asked the following "
    "instruction:\n\n{question}\n\nBased on the video segment up to this "
    "moment, is it the right time to fulfill the instruction now? Answer "
    "with only 'yes' or 'no'."
)

PROACTIVE_OPEN_PROMPT = (
    "It is now the right moment. Please answer the original instruction "
    "concisely: {question}"
)


def _proactive_default_max_round(doc: dict, user_cap: Optional[int] = None) -> int:
    start = parse_timestamp(doc["time_stamp"])
    gt = parse_timestamp(doc["ground_truth_time_stamp"])
    base = max(1, gt - start + 4)
    if user_cap is not None:
        return min(base, int(user_cap))
    return base


def _proactive_messages(*, video_path: str, start: float, end: float, prompt_text: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "video", "url": video_path, "start_time": float(start), "end_time": float(end)},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]


def streamingbench_proactive_doc_to_messages(
    doc, lmms_eval_specific_kwargs=None, previous_output=None, round_idx=None, previous_round_info=None
):
    """Multi-round doc_to_messages for proactive output task.

    Round 0 returns a plain messages list. Subsequent rounds return a 4-tuple
    ``(raw_messages, terminal, previous_output, previous_round_info)``.
    """
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    user_cap = lmms_eval_specific_kwargs.get("max_round_proactive")
    max_round = _proactive_default_max_round(doc, user_cap=user_cap)

    video_path = resolve_video_path(
        question_id=doc["question_id"],
        config_name="Proactive_Output",
        task_type=doc.get("task_type"),
    )
    start_time = float(parse_timestamp(doc["time_stamp"]))
    question = doc["question"]

    if round_idx is None or round_idx == 0:
        return _proactive_messages(
            video_path=video_path,
            start=start_time,
            end=start_time + 1.0,
            prompt_text=PROACTIVE_YESNO_PROMPT.format(question=question),
        )

    prev_info = dict(previous_round_info or {})
    prev_info.setdefault("max_round", max_round)
    prev_info.setdefault("answer_round", None)
    prev_info.setdefault("final_time", None)
    last_resp = (previous_output[-1] if previous_output else "").strip().lower()

    if prev_info.get("answer_round") is not None and round_idx == prev_info["answer_round"] + 1:
        prev_info["final_answer"] = previous_output[-1] if previous_output else ""
        return [], True, previous_output, prev_info

    if round_idx > prev_info["max_round"]:
        return [], True, previous_output, prev_info

    if "yes" in last_resp:
        end = start_time + round_idx
        prev_info["answer_round"] = round_idx
        prev_info["final_time"] = end
        msgs = _proactive_messages(
            video_path=video_path,
            start=start_time,
            end=end,
            prompt_text=PROACTIVE_OPEN_PROMPT.format(question=question),
        )
        return msgs, False, previous_output, prev_info

    end = start_time + round_idx + 1
    msgs = _proactive_messages(
        video_path=video_path,
        start=start_time,
        end=end,
        prompt_text=PROACTIVE_YESNO_PROMPT.format(question=question),
    )
    return msgs, False, previous_output, prev_info


def proactive_score_one(doc: dict, *, last_time: float, last_answer: str) -> dict:
    gt_time = float(parse_timestamp(doc["ground_truth_time_stamp"]))
    gt_output = str(doc["ground_truth_output"]).strip()
    time_correct = int(abs(last_time - gt_time) <= 2)
    answer_correct = int(time_correct and (gt_output in (last_answer or "")))
    return {"time_correct": time_correct, "answer_correct": answer_correct}


def streamingbench_proactive_process_results(doc, results):
    """``results`` is the per-sample response list from the multi-round runner.
    May be wrapped in an extra list — flatten one level if needed.

    Reconstructs ``last_time`` and ``last_answer`` from the response list:
      - find index ``k_yes`` of first response containing 'yes' (case-insensitive).
      - if not found: ``last_time = start + len(results)``, ``last_answer = results[-1]``.
      - else: open-form is round ``k_yes + 1``, with clip end ``start + (k_yes + 1)``.
        ``last_time = start + k_yes + 1``; ``last_answer = results[k_yes+1]`` if available
        else ``results[k_yes]``.
    """
    if isinstance(results, (list, tuple)) and len(results) == 1 and isinstance(results[0], (list, tuple)):
        results = results[0]
    flat = [str(r) for r in results] if isinstance(results, (list, tuple)) else [str(results)]
    start = float(parse_timestamp(doc["time_stamp"]))
    k_yes: Optional[int] = None
    for i, r in enumerate(flat):
        if "yes" in r.strip().lower():
            k_yes = i
            break
    if k_yes is None:
        last_time = start + len(flat)
        last_answer = flat[-1] if flat else ""
    else:
        last_time = start + k_yes + 1
        last_answer = flat[k_yes + 1] if k_yes + 1 < len(flat) else flat[k_yes]
    scored = proactive_score_one(doc, last_time=last_time, last_answer=last_answer)
    scored["category"] = doc.get("task_type", "")
    return {"proactive": scored}


def streamingbench_proactive_aggregate(items, args=None):
    buckets: dict[str, dict[str, list[int]]] = defaultdict(lambda: {"time": [], "answer": []})
    for it in items:
        buckets[it["category"]]["time"].append(int(it["time_correct"]))
        buckets[it["category"]]["answer"].append(int(it["answer_correct"]))
    out: dict[str, float] = {}
    answer_means: list[float] = []
    for cat, d in buckets.items():
        time_pct = 100.0 * sum(d["time"]) / max(len(d["time"]), 1)
        ans_pct = 100.0 * sum(d["answer"]) / max(len(d["answer"]), 1)
        out[f"{cat}_time"] = time_pct
        out[f"{cat}_answer"] = ans_pct
        answer_means.append(ans_pct)
    out["average"] = sum(answer_means) / max(len(answer_means), 1) if answer_means else 0.0
    return out


def streamingbench_static_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Stub: same prompt as ``streamingbench_static_doc_to_messages`` minus
    the video content. lmms-eval calls this during task __init__ sanity check."""
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    options = parse_options(doc["options"])
    return build_static_prompt(question=doc["question"], options=options)


def streamingbench_sequential_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Stub mirroring ``streamingbench_sequential_doc_to_messages``."""
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    options = parse_options(doc["options"])
    history = doc.get("_prior_history", "")
    return build_sequential_prompt(question=doc["question"], options=options, history=history)


def streamingbench_proactive_doc_to_text(
    doc, lmms_eval_specific_kwargs=None, previous_output=None, round_idx=None, previous_round_info=None
):
    """Stub mirroring the initial-round yes/no prompt of proactive."""
    return PROACTIVE_YESNO_PROMPT.format(question=doc["question"])
