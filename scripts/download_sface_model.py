#!/usr/bin/env python3
"""Download the SFace ONNX model used by Module 1 scene-cut detection."""
import urllib.request
from pathlib import Path

URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
    "face_recognition_sface/face_recognition_sface_2021dec.onnx"
)
DEST = (
    Path(__file__).resolve().parent.parent
    / "models" / "sface" / "face_recognition_sface_2021dec.onnx"
)


def _progress(count, block_size, total_size):
    if total_size > 0:
        pct = min(100, count * block_size * 100 // total_size)
        print(f"\r  {pct}%  ({count * block_size / 1e6:.1f} MB)", end="", flush=True)


def main():
    if DEST.exists():
        print(f"Already exists: {DEST}")
        return
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading SFace model → {DEST}")
    urllib.request.urlretrieve(URL, DEST, reporthook=_progress)
    print(f"\nDone. {DEST.stat().st_size / 1e6:.1f} MB saved.")


if __name__ == "__main__":
    main()
