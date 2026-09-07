"""
Module B1 runner — transcription + speaker diarization.

Runs transcriber.py on one video or a directory of videos.
Output: data/interim/track_b/<video_id>/transcript.json

Usage:
    # single video
    python src/track_b/module_b1_transcription/main_b1.py \
        --video data/raw/infer/99kfQqIiJcg_75_1.mp4 --max-speakers 2

    # batch (directory)
    python src/track_b/module_b1_transcription/main_b1.py \
        --video-dir data/raw/infer/real/ --max-speakers 2
"""

import argparse
import sys
import os
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from src.track_b.module_b1_transcription.transcriber import process_video, TRACK_B_INTERIM

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def main():
    parser = argparse.ArgumentParser(description="Module B1: transcription + speaker diarization.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=str, help="Single video file path.")
    group.add_argument("--video-dir", type=str, help="Directory of video files (batch mode).")
    parser.add_argument(
        "--out-root", type=str, default=str(TRACK_B_INTERIM),
        help=f"Output root (default: {TRACK_B_INTERIM})",
    )
    parser.add_argument("--min-speakers", type=int, default=1)
    parser.add_argument("--max-speakers", type=int, default=4)
    args = parser.parse_args()

    out_root = Path(args.out_root)

    if args.video:
        videos = [Path(args.video)]
    else:
        video_dir = Path(args.video_dir)
        videos = sorted(f for f in video_dir.iterdir() if f.suffix.lower() in VIDEO_EXTS)
        print(f"[B1] Found {len(videos)} videos in {video_dir}")

    for video_path in videos:
        process_video(
            video_path,
            out_root=out_root,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
        )

    print(f"\n[B1] Done. Transcripts saved to: {out_root}")


if __name__ == "__main__":
    main()
