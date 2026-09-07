#!/usr/bin/env python3
"""
generate_case_figure.py — regenerate case study figures for the VERA paper.

Usage:
    python3 scripts/generate_case_figure.py --case 1   # case 1, 2, or 3

Each figure has:
  - Title bar (full width)
  - Left 38 %: 2×2 frame grid at correct 16:9 aspect ratio
  - Right 62 %: verdict sections (Primary Evidence, Visual Inspection,
                Metric Summary, Key Reasoning, Anomalous Features table)
  - Bottom strip: chunk metadata
"""

import argparse, json, textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.image import imread

# ── project paths ─────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent

CASES = {
    1: {
        "frame_dir":    BASE / "figures/case_studies/case1",
        "verdict_file": BASE / "src/track_a/module_5_agent/verdicts_qwen/"
                               "KNEGWrD08f8_39_1_3831097e-e5e6-4e97-851e-931dbb59e80a_verdict.json",
        "evidence_file":BASE / "evidence_infer/KNEGWrD08f8_39_1_evidence.json",
        "chunk_id":     "chunk_0001",
        "display_id":   "KNEGWrD08f8_39_1",
        "ground_truth": "REAL",
        "correct":      True,
        "title":        "Case 1 — Genuine video, correctly classified as GENUINE",
        "out":          BASE / "figures/appendix_case1.png",
        "out2":         BASE / "CSONET2026/figures/appendix_case1.png",
    },
    2: {
        "frame_dir":    BASE / "figures/case_studies/case2",
        "verdict_file": None,   # fill in as needed
        "evidence_file":None,
        "chunk_id":     "chunk_0001",
        "display_id":   "99kfQqIiJcg_75_1",
        "ground_truth": "REAL",
        "correct":      False,
        "title":        "Case 2 — Genuine video, incorrectly classified as FAKE (false positive)",
        "out":          BASE / "figures/appendix_case2.png",
        "out2":         BASE / "CSONET2026/figures/appendix_case2.png",
    },
    3: {
        "frame_dir":    BASE / "figures/case_studies/case3",
        "verdict_file": None,
        "evidence_file":None,
        "chunk_id":     "chunk_0000",
        "display_id":   "25603-LUn8IjZKBPg_85_2",
        "ground_truth": "FAKE",
        "correct":      True,
        "title":        "Case 3 — Fake video, correctly classified as FAKE",
        "out":          BASE / "figures/appendix_case3.png",
        "out2":         BASE / "CSONET2026/figures/appendix_case3.png",
    },
}

# ── palette ───────────────────────────────────────────────────────────────────
C_GREEN    = "#2d7a2d"
C_GREEN_BG = "#f0f7f0"
C_GREEN_BD = "#aaccaa"
C_ORANGE   = "#cc6600"
C_ORANGE_BG= "#fff5e6"
C_ORANGE_BD= "#ddaa66"
C_GRAY_BG  = "#f7f7f7"
C_RED      = "#cc2222"

LOW_SUSP = {
    "vocal_jitter_relative", "vocal_shimmer_relative",
    "blinking_variance", "mouth_movement_variance", "iris_jitter_variance",
}


def severity(name, signal):
    if name in LOW_SUSP:
        return ("CRITICAL" if signal == "FAR_BELOW_NORMAL" else
                "ELEVATED" if signal == "BELOW_NORMAL" else None)
    return ("CRITICAL" if signal == "FAR_ABOVE_NORMAL" else
            "ELEVATED" if signal == "ABOVE_NORMAL" else None)


def load_elevated(chunk):
    """Return list of (feature, value, p50, pct, severity) for ELEVATED/CRITICAL features."""
    all_f = {}
    all_f.update(chunk["features"].get("visual", {}))
    all_f.update(chunk["features"].get("audio_visual", {}))
    rows = []
    for fn, fd in all_f.items():
        if not isinstance(fd, dict) or fd.get("value") is None:
            continue
        sev = severity(fn, fd.get("signal", "NORMAL"))
        if sev:
            rows.append((fn, fd["value"], fd["genuine_p50"], fd["percentile_rank"], sev))
    rows.sort(key=lambda x: x[3], reverse=True)
    return rows


def add_section(fig, x, y, w, h, header, body_text, hcolor, bg, bdcolor,
                hfont=10.5, bfont=8.8, max_wrap=70):
    """Draw a bordered section with a header and wrapped body text."""
    ax = fig.add_axes([x, y, w, h])
    ax.set_facecolor(bg)
    ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_edgecolor(bdcolor); sp.set_linewidth(0.5)

    ax.text(0.015, 0.96, header, ha="left", va="top",
            fontsize=hfont, fontweight="bold", color=hcolor,
            transform=ax.transAxes)

    FW_loc  = fig.get_figwidth()
    FH_loc  = fig.get_figheight()
    DPI_loc = fig.dpi
    line_height_px = bfont * 1.40 * DPI_loc / 72.0
    section_h_px   = h * FH_loc * DPI_loc
    step = line_height_px / section_h_px

    if isinstance(body_text, list):
        lines = body_text
    else:
        char_px   = bfont * 0.55 * DPI_loc / 72.0
        sect_w_px = w * FW_loc * DPI_loc * 0.97
        wrap_w    = min(max_wrap, max(40, int(sect_w_px / char_px)))
        lines = textwrap.wrap(body_text, width=wrap_w)

    y_pos = 0.84
    for line in lines:
        if y_pos < 0.02:
            break
        if line == "":
            y_pos -= step * 0.4
            continue
        ax.text(0.015, y_pos, line, ha="left", va="top",
                fontsize=bfont, color="#333333", transform=ax.transAxes,
                clip_on=True)
        y_pos -= step
    return ax


def generate(case_id: int):
    cfg = CASES[case_id]

    # ── load data ─────────────────────────────────────────────────────────────
    verdict, evidence = None, None
    if cfg["verdict_file"] and Path(cfg["verdict_file"]).exists():
        with open(cfg["verdict_file"]) as f:
            verdict = json.load(f)["verdict"]
    if cfg["evidence_file"] and Path(cfg["evidence_file"]).exists():
        with open(cfg["evidence_file"]) as f:
            evidence = json.load(f)

    chunk      = evidence["chunks"][cfg["chunk_id"]] if evidence else None
    threshold  = evidence["model_metadata"]["threshold"] if evidence else 15.54
    anom_score = chunk["anomaly"]["joint_anomaly_score"] if chunk else 0.0
    t0  = chunk["time_metadata"]["start_sec"] if chunk else 0.0
    t1  = chunk["time_metadata"]["end_sec"]   if chunk else 4.0
    table_rows = load_elevated(chunk) if chunk else []

    frames = [imread(str(p))
              for p in sorted(cfg["frame_dir"].glob("*.jpg"))[:4]]
    if not frames:
        raise FileNotFoundError(f"No frames found in {cfg['frame_dir']}")

    # ── figure geometry ───────────────────────────────────────────────────────
    FW, FH = 20.0, 15.0      # figure width / height in inches (≈1.33:1 matches original)
    DPI    = 150

    # vertical bands (fractions of FH)
    TITLE_H  = 0.055
    STRIP_H  = 0.050
    BODY_Y   = STRIP_H + 0.005
    BODY_H   = 1.0 - TITLE_H - STRIP_H - 0.01

    # horizontal split — pixel-measured from original (2973×2301):
    #   frame right edge x=1443 → LEFT_W=0.485
    #   right panel start x=1540 → RIGHT_X=0.518, RGAP=0.033
    LEFT_W   = 0.485
    LMARGIN  = 0.005
    RGAP     = 0.033
    RIGHT_X  = LEFT_W + RGAP            # 0.518
    RIGHT_W  = 1.0 - RIGHT_X - 0.005   # 0.477

    # ── frame grid geometry (16:9 — professor's scaling fix) ──────────────────
    AR        = (FW / FH) * (9.0 / 16.0)   # h_frac per w_frac for 16:9
    F_GAP     = 0.010
    LABEL_H   = 0.022
    w_f       = (LEFT_W - LMARGIN - F_GAP) / 2   # 0.235
    h_f       = w_f * AR                           # 0.176 → 16:9 ✓

    row_h     = h_f + LABEL_H
    grid_h    = 2 * row_h + F_GAP
    grid_ybot = BODY_Y + (BODY_H - grid_h) / 2.0

    def frame_y(row): return grid_ybot + row * (row_h + F_GAP) + LABEL_H
    def frame_x(col): return LMARGIN  + col * (w_f + F_GAP)
    def label_y(row): return frame_y(row) + h_f + 0.003

    FRAME_LAYOUT = [(0, 1), (1, 1), (0, 0), (1, 0)]

    # ── right-panel section heights (pixel-measured from original) ─────────────
    SEC_GAP   = 0.007
    TABLE_H   = 0.115
    KR_H      = 0.110
    MS_H      = 0.150
    VI_H      = 0.150
    CARDS_H   = 0.085
    VIDID_H   = 0.030

    # layout from bottom up; table_y=0.108 from pixel measurement
    table_y = 0.108
    kr_y    = table_y + TABLE_H + SEC_GAP
    ms_y    = kr_y   + KR_H    + SEC_GAP
    vi_y    = ms_y   + MS_H    + SEC_GAP
    cards_y = BODY_Y + BODY_H - CARDS_H - 0.010
    vidid_y = cards_y - VIDID_H - 0.002
    pe_y    = vi_y + VI_H + SEC_GAP
    pe_h    = vidid_y - pe_y - 0.002

    # ── build figure ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(FW, FH), dpi=DPI)
    fig.patch.set_facecolor("white")

    # title
    ax_title = fig.add_axes([0, 1 - TITLE_H, 1, TITLE_H])
    ax_title.set_facecolor("white"); ax_title.axis("off")
    ax_title.text(0.5, 0.5, cfg["title"],
                  ha="center", va="center", fontsize=19, fontweight="bold",
                  color=C_GREEN, transform=ax_title.transAxes)

    # bottom strip
    ax_strip = fig.add_axes([0, 0, 1, STRIP_H])
    ax_strip.set_facecolor(C_GRAY_BG); ax_strip.axis("off")
    ax_strip.text(0.5, 0.5,
        f"{cfg['chunk_id']}   [{t0:.1f} s – {t1:.1f} s]     "
        f"anomaly score = {anom_score:.2f}   (threshold τ = {threshold:.2f})",
        ha="center", va="center", fontsize=13, color="#333333",
        transform=ax_strip.transAxes)

    # ── frames ────────────────────────────────────────────────────────────────
    for i, (col, row) in enumerate(FRAME_LAYOUT):
        fx = frame_x(col)
        fy = frame_y(row)

        # "Frame N" label
        lx = fx + w_f / 2
        ly = label_y(row)
        fig.text(lx, ly, f"Frame {i+1}",
                 ha="center", va="bottom", fontsize=10.5, color="#333333")

        # axes: width w_f, height h_f  →  actual size FW*w_f × FH*h_f inches
        # = (FW*w_f):(FH*h_f) = w_f/h_f * (FW/FH) = (1/AR)*(FW/FH)*(FW/FH)
        # = (16/9) → 16:9 ✓
        ax_f = fig.add_axes([fx, fy, w_f, h_f])
        ax_f.imshow(frames[i], aspect="auto")   # fill the 16:9 box exactly
        ax_f.set_xticks([]); ax_f.set_yticks([])
        for sp in ax_f.spines.values():
            sp.set_edgecolor("#aaaaaa"); sp.set_linewidth(0.7)

    # ── verdict cards ─────────────────────────────────────────────────────────
    gt_w   = RIGHT_W * 0.30
    pred_w = RIGHT_W * 0.36
    bad_w  = RIGHT_W - gt_w - pred_w

    for sx, sw, lines in [
        (RIGHT_X,              gt_w,   ["Ground truth", cfg["ground_truth"]]),
        (RIGHT_X + gt_w,       pred_w, ["Prediction",
                                         verdict["label"] if verdict else "—"]),
        (RIGHT_X + gt_w + pred_w, bad_w,
            ["✓  Correct" if cfg["correct"] else "✗  Incorrect"]),
    ]:
        ax_c = fig.add_axes([sx, cards_y, sw, CARDS_H])
        ax_c.set_facecolor("white"); ax_c.axis("off")
        for sp in ax_c.spines.values():
            sp.set_visible(True); sp.set_edgecolor("#cccccc"); sp.set_linewidth(0.8)
        if len(lines) == 2:
            ax_c.text(0.5, 0.75, lines[0], ha="center", va="center",
                      fontsize=8.5, color="#888888", transform=ax_c.transAxes)
            color = (C_GREEN if lines[1] in ("GENUINE", "REAL") else
                     C_RED   if lines[1] in ("FAKE",) else "#333333")
            ax_c.text(0.5, 0.30, lines[1], ha="center", va="center",
                      fontsize=15, fontweight="bold", color=color,
                      transform=ax_c.transAxes)
        else:
            color = C_GREEN if cfg["correct"] else C_RED
            ax_c.text(0.5, 0.50, lines[0], ha="center", va="center",
                      fontsize=13, fontweight="bold", color=color,
                      transform=ax_c.transAxes)

    # video ID
    fig.text(RIGHT_X + RIGHT_W / 2, vidid_y + VIDID_H / 2,
             f"Video ID: {cfg['display_id']}",
             ha="center", va="center", fontsize=8.5, color="#888888")

    # ── text sections ─────────────────────────────────────────────────────────
    if verdict:
        pe_items = verdict.get("primary_evidence", [])
        pe_lines = []
        for item in pe_items:
            wrapped = textwrap.wrap("• " + item, width=70)
            pe_lines.extend(wrapped)
            pe_lines.append("")

        add_section(fig, RIGHT_X, pe_y, RIGHT_W, pe_h,
                    "Primary Evidence", pe_lines,
                    C_GREEN, C_GREEN_BG, C_GREEN_BD)

        add_section(fig, RIGHT_X, vi_y, RIGHT_W, VI_H,
                    "MLLM Visual Inspection",
                    verdict.get("visual_inspection_summary", ""),
                    C_GREEN, C_GREEN_BG, C_GREEN_BD, max_wrap=70)

        add_section(fig, RIGHT_X, ms_y, RIGHT_W, MS_H,
                    "MLLM Metric Summary",
                    verdict.get("metric_summary", ""),
                    C_GREEN, C_GREEN_BG, C_GREEN_BD, max_wrap=70)

        add_section(fig, RIGHT_X, kr_y, RIGHT_W, KR_H,
                    "MLLM Key Reasoning",
                    verdict.get("key_reasoning", ""),
                    C_GREEN, C_GREEN_BG, C_GREEN_BD, max_wrap=70)

    # ── anomalous features table ───────────────────────────────────────────────
    ax_tbl = fig.add_axes([RIGHT_X, table_y, RIGHT_W, TABLE_H])
    ax_tbl.set_facecolor(C_ORANGE_BG); ax_tbl.axis("off")
    ax_tbl.set_xlim(0, 1); ax_tbl.set_ylim(0, 1)
    for sp in ax_tbl.spines.values():
        sp.set_visible(True); sp.set_edgecolor(C_ORANGE_BD); sp.set_linewidth(0.5)

    ax_tbl.text(0.015, 0.945, "Anomalous Features in Top Chunk",
                ha="left", va="top", fontsize=10.5, fontweight="bold",
                color=C_ORANGE, transform=ax_tbl.transAxes)

    # column x-positions for RIGHT_W=0.477 — feature takes ~15.5% of table
    CX  = [0.010, 0.185, 0.270, 0.395, 0.510, 0.595]
    HDR = ["Feature",   "",   "Value", "Genuine P50", "Pct", "Severity"]
    HDR_Y = 0.795

    # gray background behind header row
    ax_tbl.add_patch(mpatches.FancyBboxPatch(
        (0, 0.72), 1, 0.225,
        boxstyle="square,pad=0", linewidth=0,
        facecolor="#eeeeee", transform=ax_tbl.transAxes, zorder=0))

    for cx, hdr in zip(CX, HDR):
        ax_tbl.text(cx, HDR_Y, hdr, ha="left", va="top",
                    fontsize=9, fontweight="bold", color="#555555",
                    transform=ax_tbl.transAxes, zorder=1)

    # separator under header
    ax_tbl.plot([0, 1], [0.72, 0.72], color=C_ORANGE_BD, linewidth=0.9,
                transform=ax_tbl.transAxes, clip_on=False)

    row_y = 0.615
    for fn, val, p50, pct, sev in table_rows[:5]:
        sev_col  = "#cc3300" if sev == "CRITICAL" else "#cc7700"
        direction = "LOW ↓" if fn in LOW_SUSP else "HIGH ↑"
        vals = [fn, direction, f"{val:.4f}", f"{p50:.4f}", str(pct), f"★ {sev}"]
        cols = ["#333333", C_ORANGE, sev_col, "#555555", sev_col, sev_col]
        for cx, txt, col in zip(CX, vals, cols):
            ax_tbl.text(cx, row_y, txt, ha="left", va="top",
                        fontsize=8.8, color=col, transform=ax_tbl.transAxes,
                        clip_on=True)
        row_y -= 0.135
        if row_y < 0.02:
            break

    # ── save ─────────────────────────────────────────────────────────────────
    for out in [cfg["out"], cfg["out2"]]:
        fig.savefig(out, dpi=DPI, bbox_inches="tight",
                    facecolor="white", pad_inches=0.04)
        print(f"  ✓  {out}")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=int, choices=[1, 2, 3], default=1)
    args = ap.parse_args()
    print(f"Generating case {args.case} …")
    generate(args.case)
    print("Done.")
