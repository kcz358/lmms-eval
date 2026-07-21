import os
import re
import zipfile
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from filelock import FileLock
from huggingface_hub import hf_hub_download

INSTRUCTION = "Please answer the audio in the video."
VIDEO_REPO_ID = "kcz358/QIVD"
HF_HOME = Path(os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface")))
QIVD_ROOT = HF_HOME / "qivd"
VIDEOS_DIR = QIVD_ROOT / "videos"


def _specific_kwargs(kwargs):
    kwargs = kwargs or {}
    return kwargs.get("default", kwargs)


@lru_cache(maxsize=1)
def get_qivd_videos_dir():
    if VIDEOS_DIR.is_dir() and any(VIDEOS_DIR.glob("*.mp4")):
        return VIDEOS_DIR
    QIVD_ROOT.mkdir(parents=True, exist_ok=True)
    with FileLock(str(QIVD_ROOT / ".qivd.lock")):
        if VIDEOS_DIR.is_dir() and any(VIDEOS_DIR.glob("*.mp4")):
            return VIDEOS_DIR
        archive = hf_hub_download(
            repo_id=VIDEO_REPO_ID,
            filename="qivd_videos.zip",
            repo_type="dataset",
        )
        with zipfile.ZipFile(archive) as source:
            source.extractall(QIVD_ROOT)
    if not VIDEOS_DIR.is_dir() or not any(VIDEOS_DIR.glob("*.mp4")):
        raise RuntimeError(f"QIVD extraction failed: {VIDEOS_DIR} is empty")
    return VIDEOS_DIR


def qivd_doc_to_visual(doc, lmms_eval_specific_kwargs=None):
    data_dir = _specific_kwargs(lmms_eval_specific_kwargs).get("data_dir")
    if data_dir:
        root = Path(os.path.expanduser(data_dir))
        candidates = [root / doc["video_name"], root / "videos" / doc["video_name"]]
    else:
        candidates = [get_qivd_videos_dir() / doc["video_name"]]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError(
            f"QIVD video {doc['video_name']} not found under {candidates}. "
            "Set lmms_eval_specific_kwargs.default.data_dir "
            "to the directory containing the extracted MP4 files."
        )
    return [str(path)]


def qivd_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    video_path = qivd_doc_to_visual(doc, lmms_eval_specific_kwargs)[0]
    return [
        {
            "role": "user",
            "content": [
                {"type": "video", "url": video_path},
                {"type": "text", "text": INSTRUCTION},
            ],
        }
    ]


def qivd_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return INSTRUCTION


def _normalize(text):
    return " ".join(re.findall(r"[a-z0-9]+", (text or "").lower()))


def qivd_process_results(doc, results):
    prediction = results[0].strip() if results else ""
    pred = _normalize(prediction)
    short = _normalize(doc["short_answer"])
    detailed = _normalize(doc["answer"])
    score = float(bool(pred) and pred in {short, detailed})
    return {
        "qivd_accuracy": {
            "score": score,
            "id": doc["id"],
            "category": doc["category"],
            "question": doc["question"],
            "prediction": prediction,
            "short_answer": doc["short_answer"],
            "answer": doc["answer"],
        }
    }


def qivd_aggregate(results, args=None):
    by_category = defaultdict(list)
    for result in results:
        by_category[result["category"]].append(result["score"])
    category_scores = {
        category: 100.0 * sum(scores) / len(scores)
        for category, scores in sorted(by_category.items())
    }
    all_scores = [result["score"] for result in results]
    category_scores["overall"] = 100.0 * sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(category_scores)
    return category_scores["overall"]
