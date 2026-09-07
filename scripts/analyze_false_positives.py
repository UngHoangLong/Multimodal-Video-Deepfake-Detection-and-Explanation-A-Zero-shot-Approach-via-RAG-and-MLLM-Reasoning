#!/usr/bin/env python3
"""
analyze_false_positives.py

Phân tích genuine videos bị predict sai (false positives) để identify:
- off_screen_speaker : audio/AV features CRITICAL, visual features NORMAL
- scene_cut          : visual blending/kinematic CRITICAL, audio NORMAL
- mixed              : cả hai đều CRITICAL
- other              : score cao nhưng không có CRITICAL feature rõ ràng

Usage:
    python scripts/analyze_false_positives.py \
        --evidence-dir evidence_infer \
        --manifest data/external/mavos_dd_en/manifest.csv \
        --output false_positive_analysis.csv
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

THRESHOLD = 15.54

VISUAL_FEATURES = {
    "max_blending_flicker", "blending_variance",
    "max_kinematic_flicker", "max_rigid_violation",
    "mean_landmark_jitter", "max_blur_flicker",
    "blur_flicker_variance", "max_texture_flicker",
    "asymmetry_max", "blinking_variance",
    "mouth_movement_variance", "gaze_anomaly",
    "iris_jitter_variance",
}

AUDIO_AV_FEATURES = {
    "wer_score", "temporal_anomaly", "min_temporal_anomaly",
    "semantic_anomaly", "min_cosine_anomaly",
    "temporal_sync_variance",
    "vocal_jitter_relative", "vocal_shimmer_relative",
}

LOW_IS_SUSPICIOUS = {
    "vocal_jitter_relative", "vocal_shimmer_relative",
    "blinking_variance", "mouth_movement_variance", "iris_jitter_variance",
}


def is_critical(feat_name: str, feat_data: dict) -> bool:
    signal = feat_data.get("signal", "NORMAL")
    if feat_name in LOW_IS_SUSPICIOUS:
        return signal == "FAR_BELOW_NORMAL"
    return signal == "FAR_ABOVE_NORMAL"


def analyze_chunk(chunk_data: dict):
    feats = {}
    feats.update(chunk_data.get("features", {}).get("visual", {}))
    feats.update(chunk_data.get("features", {}).get("audio_visual", {}))

    critical_visual, critical_audio = [], []
    for fname, fdata in feats.items():
        if not isinstance(fdata, dict):
            continue
        if is_critical(fname, fdata):
            if fname in VISUAL_FEATURES:
                critical_visual.append(fname)
            elif fname in AUDIO_AV_FEATURES:
                critical_audio.append(fname)

    return critical_visual, critical_audio


def classify(critical_visual, critical_audio) -> str:
    hv = len(critical_visual) > 0
    ha = len(critical_audio) > 0
    if hv and ha:
        return "mixed"
    if ha:
        return "off_screen_speaker"
    if hv:
        return "scene_cut_or_visual"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence-dir", default="evidence_infer")
    ap.add_argument("--manifest", default="data/external/mavos_dd_en/manifest.csv")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--output", default="false_positive_analysis.csv")
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

        best_id = max(chunks,
                      key=lambda k: chunks[k].get("anomaly", {})
                                               .get("joint_anomaly_score", 0))
        best_score = chunks[best_id].get("anomaly", {}).get("joint_anomaly_score", 0)

        if best_score <= args.threshold:
            continue  # true negative

        cv, ca = analyze_chunk(chunks[best_id])
        cat = classify(cv, ca)

        results.append({
            "video_id": vid,
            "max_score": round(best_score, 4),
            "category": cat,
            "critical_visual": "|".join(cv),
            "critical_audio": "|".join(ca),
            "n_critical_visual": len(cv),
            "n_critical_audio": len(ca),
        })

    # Summary
    total = len(results)
    print(f"\nGenuine false positives (score > {args.threshold}): {total}")
    if total:
        for cat, cnt in sorted(Counter(r["category"] for r in results).items(),
                                key=lambda x: -x[1]):
            print(f"  {cat:25s}: {cnt:4d}  ({100*cnt/total:.1f}%)")

    # Save CSV
    out = Path(args.output)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "video_id", "max_score", "category",
            "critical_visual", "critical_audio",
            "n_critical_visual", "n_critical_audio",
        ])
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved -> {out}")

    # Also save off_screen_speaker IDs separately for easy exclusion
    oss_ids = [r["video_id"] for r in results if r["category"] == "off_screen_speaker"]
    if oss_ids:
        oss_path = out.parent / "off_screen_speaker_ids.txt"
        oss_path.write_text("\n".join(oss_ids) + "\n", encoding="utf-8")
        print(f"Off-screen speaker IDs -> {oss_path} ({len(oss_ids)} videos)")


if __name__ == "__main__":
    main()
