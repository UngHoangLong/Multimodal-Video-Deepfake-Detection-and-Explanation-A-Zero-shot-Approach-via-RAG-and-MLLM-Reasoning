"""
Prepare MAVOS-DD-EN data for the Track A pipeline.

Source dataset (data/external/mavos_dd_en/) is organized by generative method
(infer/roop/, infer/echomimic/, ...) and split (genuine/train/, genuine/val/).
The pipeline (src/utils/paths.py) expects flat directories:
    data/raw/genuine/*.mp4      <- Module 3 training input
    data/raw/genuine_val/*.mp4  <- threshold calibration input
    data/raw/infer/*.mp4        <- Module 1-5 evaluation input

This script reads manifest.csv and symlinks each referenced video into the
correct flat destination (no copy, saves disk + time).

Usage:
    python scripts/prepare_mavosdd_data.py
"""

import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "data" / "external" / "mavos_dd_en"
MANIFEST = SRC_ROOT / "manifest.csv"

DEST_GENUINE_TRAIN = PROJECT_ROOT / "data" / "raw" / "genuine"
DEST_GENUINE_VAL = PROJECT_ROOT / "data" / "raw" / "genuine_val"
DEST_INFER = PROJECT_ROOT / "data" / "raw" / "infer"


def symlink_unique(local_paths, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    seen = set()
    linked, skipped_dup, skipped_missing = 0, 0, 0
    for local_path in local_paths:
        basename = Path(local_path).name
        if basename in seen:
            skipped_dup += 1
            continue
        seen.add(basename)
        src = SRC_ROOT / local_path
        if not src.exists():
            skipped_missing += 1
            continue
        dst = dest_dir / basename
        if dst.exists() or dst.is_symlink():
            continue
        dst.symlink_to(src.resolve())
        linked += 1
    return linked, skipped_dup, skipped_missing


def main():
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"[prepare] {len(rows)} rows in manifest.csv")

    genuine_train_paths = [r["local_path"] for r in rows if r["usage"] == "genuine_train"]
    genuine_val_paths = [r["local_path"] for r in rows if r["usage"] == "genuine_val"]
    infer_paths = [r["local_path"] for r in rows if r["usage"] in ("eval_track1", "eval_track2")]

    for name, paths, dest in [
        ("genuine_train -> data/raw/genuine", genuine_train_paths, DEST_GENUINE_TRAIN),
        ("genuine_val -> data/raw/genuine_val", genuine_val_paths, DEST_GENUINE_VAL),
        ("eval_track1+2 -> data/raw/infer", infer_paths, DEST_INFER),
    ]:
        linked, dup, missing = symlink_unique(paths, dest)
        print(f"[prepare] {name}: {linked} linked, {dup} dup-skipped, {missing} missing source")


if __name__ == "__main__":
    main()
