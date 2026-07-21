import argparse
import json
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Pack QIVD MP4 files into a ZIP archive")
    parser.add_argument("video_dir", type=Path, help="Directory containing QIVD MP4 files")
    parser.add_argument("--output", type=Path, default=Path("qivd_videos.zip"))
    parser.add_argument("--samples", type=Path, help="Upstream samples.json")
    parser.add_argument("--metadata-output", type=Path, default=Path("metadata.jsonl"))
    args = parser.parse_args()

    videos = sorted(args.video_dir.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"No MP4 files found in {args.video_dir}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_STORED) as archive:
        for video in videos:
            archive.write(video, f"videos/{video.name}")
    if args.samples:
        samples = json.loads(args.samples.read_text())["samples"]
        with args.metadata_output.open("w") as output:
            for sample in samples:
                category = sample.get("category") or {}
                output.write(
                    json.dumps(
                        {
                            "id": sample["_id"]["$oid"],
                            "video_name": Path(sample["filepath"]).name,
                            "question": sample["question"],
                            "answer": sample["answer"],
                            "short_answer": sample["short_answer"],
                            "answer_timestamp": sample["answer_timestamp"],
                            "category": category.get("label", ""),
                        }
                    )
                    + "\n"
                )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
