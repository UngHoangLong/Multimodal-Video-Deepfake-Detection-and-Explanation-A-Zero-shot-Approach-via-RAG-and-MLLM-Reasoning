"""
Sample 10 real + 10 fake videos from AV-Deepfake1M-PlusPlus for pipeline testing.

Usage:
    # Step 1: check repo structure (no download)
    python scripts/sample_av_deepfake1m_pp.py --check

    # Step 2: download sample
    python scripts/sample_av_deepfake1m_pp.py --download --output_dir data/raw/infer_test
"""

import argparse
import json
import random
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files

REPO_ID = "ControlNet/AV-Deepfake1M-PlusPlus"
REPO_TYPE = "dataset"
N_SAMPLE = 10
SEED = 42


# ---------------------------------------------------------------------------
# Step 1 — check repo structure
# ---------------------------------------------------------------------------

def check_structure():
    print(f"Listing files in {REPO_ID} ...")
    files = list(list_repo_files(REPO_ID, repo_type=REPO_TYPE))
    print(f"Total files: {len(files)}\n")

    top_dirs = sorted(set(f.split("/")[0] for f in files))
    print(f"=== Top-level entries ===")
    for d in top_dirs:
        print(" ", d)

    # Download val_metadata.json and show structure
    print("\n=== Downloading val_metadata.json to inspect ===")
    local = Path(hf_hub_download(REPO_ID, "val_metadata.json", repo_type=REPO_TYPE))
    with open(local) as f:
        data = json.load(f)

    print(f"Type: {type(data)}")
    if isinstance(data, list):
        print(f"Rows: {len(data)}")
        print("First row:", json.dumps(data[0], indent=2))
        labels = [str(r.get("label", r.get("type", r.get("fake", "?")))) for r in data]
        from collections import Counter
        print("Label distribution:", Counter(labels))
    elif isinstance(data, dict):
        print("Keys:", list(data.keys()))
        # Show first value
        first_key = list(data.keys())[0]
        print(f"First entry ({first_key}):", json.dumps(data[first_key], indent=2))

    # Download val.zip.001 and peek at its internal structure
    import zipfile
    print("\n=== Peeking inside val.zip.001 ===")
    zip_part = Path(hf_hub_download(REPO_ID, "val/val.zip.001", repo_type=REPO_TYPE))
    print(f"Downloaded: {zip_part}  ({zip_part.stat().st_size / 1e6:.1f} MB)")
    try:
        with zipfile.ZipFile(zip_part) as zf:
            names = zf.namelist()
            print(f"Files in this part: {len(names)}")
            print("First 15 entries:")
            for n in names[:15]:
                print(" ", n)
    except zipfile.BadZipFile:
        print("Part 001 alone is not a valid zip (multi-part — need all parts to open).")
        print("Will need to download all 54 parts and combine.")


# ---------------------------------------------------------------------------
# Step 2 — download sample
# ---------------------------------------------------------------------------

def _load_metadata(output_dir: Path):
    """
    Try to download and parse the metadata/annotation file.
    Returns list of dicts with at least 'file', 'label' keys.
    Adjust the field names below if the actual CSV/JSON uses different column names.
    """
    # Common metadata filenames to try
    candidates = [
        "metadata.json", "metadata.csv",
        "labels.json", "labels.csv",
        "annotations.json", "annotations.csv",
        "train.json", "test.json", "val.json",
    ]

    all_files = list(list_repo_files(REPO_ID, repo_type=REPO_TYPE))

    meta_file = None
    for c in candidates:
        matches = [f for f in all_files if f.endswith(c) or f == c]
        if matches:
            meta_file = matches[0]
            break

    if meta_file is None:
        # Fallback: show user what's available and exit
        meta_files = [f for f in all_files if any(
            kw in f.lower() for kw in ["meta", "label", "annot", "csv", "json", "tsv"]
        )]
        print("Could not find metadata file automatically. Candidates:")
        for f in meta_files[:20]:
            print(" ", f)
        print("\nEdit METADATA_FILE in this script to point to the correct file.")
        return None, None

    print(f"Found metadata: {meta_file}")
    local_meta = Path(hf_hub_download(REPO_ID, meta_file, repo_type=REPO_TYPE))

    if meta_file.endswith(".json"):
        with open(local_meta) as f:
            data = json.load(f)
        # Handle list or dict-of-lists
        if isinstance(data, list):
            rows = data
        elif isinstance(data, dict):
            # Try common wrapper keys
            for key in ("data", "samples", "annotations", "videos"):
                if key in data:
                    rows = data[key]
                    break
            else:
                rows = list(data.values())[0] if data else []
    else:
        # CSV
        import csv
        with open(local_meta, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    return rows, local_meta


def _infer_label(row: dict) -> str:
    """Return 'real' or 'fake' from a metadata row. Adjust field names as needed."""
    for key in ("label", "type", "fake", "is_fake", "category", "class"):
        val = row.get(key, "")
        if val is None:
            continue
        val = str(val).lower().strip()
        if val in ("real", "0", "false", "genuine"):
            return "real"
        if val in ("fake", "1", "true", "deepfake", "manipulated"):
            return "fake"
    return "unknown"


def _infer_video_path(row: dict) -> str:
    """Return the video file path field. Adjust as needed."""
    for key in ("video_path", "path", "file", "filename", "video", "clip_path"):
        val = row.get(key)
        if val:
            return str(val)
    return ""


def download_sample(output_dir: Path, n: int, seed: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    real_dir = output_dir / "real"
    fake_dir = output_dir / "fake"
    real_dir.mkdir(exist_ok=True)
    fake_dir.mkdir(exist_ok=True)

    rows, meta_path = _load_metadata(output_dir)
    if rows is None:
        return

    print(f"Total metadata rows: {len(rows)}")
    if rows:
        print("Sample row keys:", list(rows[0].keys()))
        print("Sample row:", rows[0])

    real_rows = [r for r in rows if _infer_label(r) == "real"]
    fake_rows = [r for r in rows if _infer_label(r) == "fake"]
    print(f"Real: {len(real_rows)}  Fake: {len(fake_rows)}")

    rng = random.Random(seed)
    sample_real = rng.sample(real_rows, min(n, len(real_rows)))
    sample_fake = rng.sample(fake_rows, min(n, len(fake_rows)))

    def _download(rows_to_dl, dest_dir, tag):
        for i, row in enumerate(rows_to_dl):
            vpath = _infer_video_path(row)
            if not vpath:
                print(f"  [{tag} {i+1}] No video path found in row: {row}")
                continue
            print(f"  [{tag} {i+1}/{len(rows_to_dl)}] {vpath}")
            try:
                tmp = Path(hf_hub_download(REPO_ID, vpath, repo_type=REPO_TYPE))
                dest = dest_dir / Path(vpath).name
                shutil.copy2(tmp, dest)
                print(f"    -> {dest}")
            except Exception as e:
                print(f"    ERROR: {e}")

    print(f"\nDownloading {len(sample_real)} real videos ...")
    _download(sample_real, real_dir, "real")

    print(f"\nDownloading {len(sample_fake)} fake videos ...")
    _download(sample_fake, fake_dir, "fake")

    print(f"\nDone. Output: {output_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="List repo structure only, no download.")
    parser.add_argument("--download", action="store_true",
                        help="Download sample videos.")
    parser.add_argument("--output_dir", type=str, default="data/raw/infer_test")
    parser.add_argument("--n", type=int, default=N_SAMPLE,
                        help="Number of real AND fake videos to download.")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if args.check:
        check_structure()
    elif args.download:
        download_sample(Path(args.output_dir), args.n, args.seed)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
