#!/usr/bin/env python3
"""
find_offscreen_speaker.py

Tìm TẤT CẢ genuine videos có audio-visual consistency features bất thường
(wer_score, temporal_anomaly, min_temporal_anomaly, semantic_anomaly...)
bất kể combined anomaly score có vượt threshold hay không.

Usage:
    python scripts/find_offscreen_speaker.py \
        --evidence-dir evidence_infer \
        --manifest data/external/mavos_dd_en/manifest.csv \
        --output offscreen_candidates.csv
"""

import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict

# Features đo sự không khớp giữa mặt và tiếng
AV_MISMATCH_FEATURES = {
    "wer_score",           # miệng không khớp với tiếng
    "temporal_anomaly",    # audio-visual temporal desync
    "min_temporal_anomaly",# worst-point desync
    "semantic_anomaly",    # semantic mismatch
    "min_cosine_anomaly",  # worst-point semantic mismatch
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence-dir", default="evidence_infer")
    ap.add_argument("--manifest", default="data/external/mavos_dd_en/manifest.csv")
    ap.add_argument("--output", default="offscreen_candidates.csv")
    args = ap.parse_args()

    # Load ground-truth labels
    labels = {}
    with open(args.manifest, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vid = Path(row["local_path"]).stem
            labels[vid] = row.get("label", "").strip().lower()

    evidence_dir = Path(args.evidence_dir)
    results = []

    for p in sorted(evidence_dir.glob("*_evidence.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue

        vid = data.get("video_metadata", {}).get("video_id",
              p.stem.replace("_evidence", ""))

        if data.get("video_metadata", {}).get("status") != "analyzed":
            continue
        if labels.get(vid) != "real":
            continue

        chunks = data.get("chunks", {})
        if not chunks:
            continue

        # Tìm chunk có max anomaly score
        best_id = max(chunks,
                      key=lambda k: chunks[k].get("anomaly", {})
                                               .get("joint_anomaly_score", 0))
        best_score = chunks[best_id].get("anomaly", {}).get("joint_anomaly_score", 0)

        # Gộp tất cả features của best chunk
        best_chunk = chunks[best_id]
        feats = {}
        feats.update(best_chunk.get("features", {}).get("visual", {}))
        feats.update(best_chunk.get("features", {}).get("audio_visual", {}))

        # Check các AV mismatch features
        triggered = {}
        for fname in AV_MISMATCH_FEATURES:
            if fname not in feats or not isinstance(feats[fname], dict):
                continue
            signal = feats[fname].get("signal", "NORMAL")
            if signal in ("FAR_ABOVE_NORMAL", "ABOVE_NORMAL"):
                triggered[fname] = {
                    "signal": signal,
                    "value": feats[fname].get("value"),
                    "pct": feats[fname].get("percentile_rank"),
                }

        if not triggered:
            continue

        # Đếm CRITICAL (FAR_ABOVE_NORMAL)
        n_critical = sum(1 for v in triggered.values() if v["signal"] == "FAR_ABOVE_NORMAL")

        results.append({
            "video_id": vid,
            "max_score": round(best_score, 4),
            "above_threshold": best_score > 15.54,
            "n_av_critical": n_critical,
            "n_av_elevated": len(triggered) - n_critical,
            "triggered_features": "|".join(
                f"{k}(pct={v['pct']})" for k, v in triggered.items()
            ),
        })

    # Sort by n_critical desc, then max_score desc
    results.sort(key=lambda r: (-r["n_av_critical"], -r["max_score"]))

    print(f"\nGenuine videos với AV mismatch features bất thường: {len(results)}")
    print(f"  Trong đó above threshold (15.54): {sum(1 for r in results if r['above_threshold'])}")
    print(f"  Below threshold (bị bỏ sót trước đây): {sum(1 for r in results if not r['above_threshold'])}")

    print(f"\nPhân bố số CRITICAL features:")
    from collections import Counter
    for n, cnt in sorted(Counter(r["n_av_critical"] for r in results).items(), reverse=True):
        label = "CRITICAL" if n > 0 else "chỉ ELEVATED"
        print(f"  {n} {label}: {cnt} videos")

    # Save
    out = Path(args.output)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "video_id", "max_score", "above_threshold",
            "n_av_critical", "n_av_elevated", "triggered_features",
        ])
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved -> {out} ({len(results)} videos)")

    # Save IDs to txt
    ids_path = out.parent / "offscreen_candidate_ids.txt"
    ids_path.write_text(
        "\n".join(r["video_id"] for r in results) + "\n", encoding="utf-8"
    )
    print(f"Video IDs -> {ids_path}")


if __name__ == "__main__":
    main()
