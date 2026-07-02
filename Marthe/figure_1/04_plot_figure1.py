"""Figure 1 — horizontal boxenplots across shot levels.

3-panel figure (1-shot, 2-shot, 3-shot), 3 artefacts per panel sorted by S0
descending (L01 top, M02 bottom), 5 boxenplot rows per artefact:

  lower invalid  (dark red)
  lower valid    (dark green)
  neutral        (gray)
  raise valid    (light green)
  raise invalid  (light red)

The 1-shot panel combines two single-shot (idx0/1/2, three-argument) batches:
the original-direction conditions come from the pre-existing single-shot
gpt-5.5 isolation results, and the opposite-direction conditions from the
one-shot opposite batch.

Usage:
    python 04_plot_figure1.py \\
        --t1-orig <singleshot gpt55 isolation output.jsonl> \\
        --t1-opp  <oneshot_opp_output.jsonl> \\
        --t2      <turn2_output.jsonl> \\
        --t3      <turn3_output.jsonl>

Output: results/figure1.png
"""
import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from common import RESULTS_DIR, load_s0_from_runlog

# ── constants ─────────────────────────────────────────────────────────────────

# Conditions ordered bottom-to-top within each artefact block
#   label, direction_key, validity_key, fill_color
CONDITIONS = [
    ("lower invalid", "lower", "invalid", "#d62728"),
    ("lower valid",   "lower", "valid",   "#2ca02c"),
    ("neutral",       None,    None,      "#bbbbbb"),
    ("raise valid",   "raise", "valid",   "#98df8a"),
    ("raise invalid", "raise", "invalid", "#ff9896"),
]
N_STRIPS = len(CONDITIONS)
NEUTRAL_I = next(i for i, c in enumerate(CONDITIONS) if c[1] is None)

# Listed bottom-to-top so L01 (highest S0) appears at the top of the y-axis
ARTEFACT_ORDER_BT = ["M02", "S02", "L01"]


# ── data loading ─────────────────────────────────────────────────────────────

def extract_text(body):
    return "\n".join(
        c.get("text", "")
        for item in body.get("output", [])
        if item.get("type") == "message"
        for c in item.get("content", [])
        if c.get("type") == "output_text"
    )


def parse_score(t):
    m = re.search(r"-?\d+", t)
    return int(m.group()) if m else None


def load_singleshot(*paths):
    data = defaultdict(list)
    for path in paths:
        if not path:
            continue
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            parts = r["custom_id"].split("|")
            if len(parts) < 5 or not parts[3].startswith("idx"):
                continue
            aid, d, val = parts[0], parts[1], parts[2]
            sc = parse_score(extract_text(r["response"]["body"]))
            if sc is not None:
                data[(aid, d, val)].append(sc)
    return data


def load_multiturn(path, turn):
    data = defaultdict(list)
    suffix = f"|t{turn}"
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        cid = r["custom_id"]
        if not cid.endswith(suffix):
            continue
        parts = cid.split("|")
        aid, d, val = parts[0], parts[1], parts[2]
        sc = parse_score(extract_text(r["response"]["body"]))
        if sc is not None:
            data[(aid, d, val)].append(sc)
    return data


# ── plotting ──────────────────────────────────────────────────────────────────

def build_dataframe(data_by_pool, s0_per_art):
    """Build long-form DataFrame for one panel."""
    rows = []
    palette = {}
    order = []
    for aid in ARTEFACT_ORDER_BT:
        s0_scores = s0_per_art.get(aid, [])
        for label, cond_dir, cond_val, fill in CONDITIONS:
            row_id = f"{aid}|{label}"
            pts = s0_scores if cond_dir is None else data_by_pool.get((aid, cond_dir, cond_val), [])
            for sc in pts:
                rows.append({"score": sc, "row": row_id})
            palette[row_id] = fill
            order.append(row_id)
    return pd.DataFrame(rows), palette, order


def plot_panel(ax, shot_label, data_by_pool, s0_per_art):
    df, palette, order = build_dataframe(data_by_pool, s0_per_art)
    n_rows = len(ARTEFACT_ORDER_BT) * N_STRIPS

    sns.boxenplot(
        data=df, x="score", y="row", order=order,
        palette=palette, hue="row", legend=False,
        orient="h", ax=ax, linewidth=0.7, width=0.7,
    )

    ax.set_xlim(0, 101)
    # Pin y-limits to the exact data range (no default margin) so the artefact
    # blocks fill the axis and the separator lines split it into even thirds.
    # First category (M02) is at the top, so the axis runs high -> low.
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_xlabel("Score (1–100)", fontsize=8)
    ax.set_title(shot_label, fontsize=9, fontweight="bold")
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", length=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="x", lw=0.35, alpha=0.4)

    # y-tick only at the neutral strip of each artefact (no S0 in label)
    tick_pos, tick_lab = [], []
    for art_i, aid in enumerate(ARTEFACT_ORDER_BT):
        neutral_pos = art_i * N_STRIPS + NEUTRAL_I
        tick_pos.append(neutral_pos)
        tick_lab.append(aid)

    # S0 annotation sitting on the separator line at the bottom of each block.
    for art_i, aid in enumerate(ARTEFACT_ORDER_BT):
        s0_scores = s0_per_art.get(aid, [])
        if not s0_scores:
            continue
        s0_med = round(float(np.median(s0_scores)))
        y_line = (art_i + 1) * N_STRIPS - 0.5   # separator at the bottom of this cell
        if s0_med < 50:
            ax.text(98, y_line, f"S₀={s0_med}", fontsize=7, ha="right", va="bottom",
                    color="#444444", clip_on=False)
        else:
            ax.text(2, y_line, f"S₀={s0_med}", fontsize=7, ha="left", va="bottom",
                    color="#444444", clip_on=False)
    ax.set_yticks(tick_pos)
    ax.set_yticklabels(tick_lab, fontsize=8.5)

    # separator lines between artefact groups
    for i in range(1, len(ARTEFACT_ORDER_BT)):
        ax.axhline(i * N_STRIPS - 0.5, color="#cccccc", lw=0.8, zorder=0)

    # S0 reference line per artefact
    for art_i, aid in enumerate(ARTEFACT_ORDER_BT):
        s0_scores = s0_per_art.get(aid, [])
        if s0_scores:
            s0_med = float(np.median(s0_scores))
            ax.vlines(s0_med,
                      art_i * N_STRIPS - 0.5,
                      (art_i + 1) * N_STRIPS - 0.5,
                      color="#555555", lw=0.7, linestyle="--", alpha=0.4, zorder=0)


def add_figure_legend(fig):
    """Shared horizontal legend below the panels, clear of the data area."""
    handles = [
        mpatches.Patch(facecolor="#d62728", label="lower invalid"),
        mpatches.Patch(facecolor="#2ca02c", label="lower valid"),
        mpatches.Patch(facecolor="#bbbbbb", label="neutral"),
        mpatches.Patch(facecolor="#98df8a", label="raise valid"),
        mpatches.Patch(facecolor="#ff9896", label="raise invalid"),
    ]
    fig.legend(
        handles=handles, ncol=5,
        loc="upper center", bbox_to_anchor=(0.5, 0.06),
        fontsize=7.5, frameon=False,
        handlelength=1.0, handleheight=0.7, handletextpad=0.4,
        columnspacing=1.4,
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t1-orig", required=True)
    ap.add_argument("--t1-opp",  required=True)
    ap.add_argument("--t2",      required=True)
    ap.add_argument("--t3",      required=True)
    ap.add_argument("--runlog",  default=None)
    a = ap.parse_args()

    s0_raw = load_s0_from_runlog(a.runlog)
    s0_per_art = {aid: list(runs.values()) for aid, runs in s0_raw.items()}

    # 1-shot panel = original-direction single-shot isolation results + the
    # opposite-direction one-shot batch (both idx0/1/2, three-argument).
    data_1shot = load_singleshot(a.t1_orig, a.t1_opp)
    data_2shot = load_multiturn(a.t2, turn=2)
    data_3shot = load_multiturn(a.t3, turn=3)

    panels = [("1-shot", data_1shot), ("2-shot", data_2shot), ("3-shot", data_3shot)]

    n_rows = len(ARTEFACT_ORDER_BT) * N_STRIPS
    fig_height = (n_rows * 0.38 + 1.0) / 2

    fig, axes = plt.subplots(1, 3, figsize=(13, fig_height),
                             sharey=True, sharex=True)

    for ax, (label, data) in zip(axes, panels):
        plot_panel(ax, label, data, s0_per_art)

    axes[0].set_ylabel("")
    plt.subplots_adjust(wspace=0.08, bottom=0.16)
    add_figure_legend(fig)

    out = os.path.join(RESULTS_DIR, "figure1.png")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
