"""Initial-scoring (S0) distribution per artefact, 22 artefacts x 20 runs.

Horizontal boxplot + jittered points per artefact, sorted by median S0, coloured
by tier (L/M/S). Reads the single-shot S0 batch output for the given model.

    python 17_plot_s0_distributions.py --tag gpt55
    python 17_plot_s0_distributions.py --tag o4mini
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TIERCOL = {"L": "#1f77b4", "M": "#ff7f0e", "S": "#2ca02c"}


def txt(b):
    return "".join(c.get("text", "") for it in b.get("output", [])
                   if it.get("type") == "message"
                   for c in it.get("content", []) if c.get("type") == "output_text")


def score(t):
    m = re.search(r"-?\d+", t)
    return int(m.group()) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="gpt55")
    a = ap.parse_args()

    folder = os.path.join(ROOT, "results", "singleshot", a.tag)
    s0f = None
    for p in sorted(glob.glob(os.path.join(folder, "*output*.jsonl"))):
        cid = json.loads(open(p).readline())["custom_id"]
        if "|" not in cid and "_run" in cid:
            s0f = p
            break
    if s0f is None:
        raise SystemExit(f"no S0 output found under {folder}")

    data = defaultdict(list)
    for l in open(s0f, encoding="utf-8"):
        r = json.loads(l)
        aid, _ = r["custom_id"].split("_run")
        v = score(txt(r["response"]["body"]))
        if v is not None:
            data[aid].append(v)
    arts = sorted(data, key=lambda x: np.median(data[x]))      # sort by median S0

    fig, ax = plt.subplots(figsize=(10, 8))
    rng = np.random.default_rng(0)
    for i, aid in enumerate(arts):
        vals = np.array(data[aid]); col = TIERCOL[aid[0]]
        bp = ax.boxplot(vals, positions=[i], widths=0.6, patch_artist=True,
                        orientation="horizontal", showfliers=False,
                        medianprops=dict(color="black", lw=1.2))
        bp["boxes"][0].set(facecolor=col, alpha=0.35, edgecolor=col)
        for w in bp["whiskers"] + bp["caps"]:
            w.set(color=col, lw=1)
        ax.scatter(vals, i + rng.uniform(-0.18, 0.18, len(vals)), s=14, color=col,
                   alpha=0.7, edgecolor="none", zorder=3)
        ax.text(101, i, f" n={len(vals)} med={int(np.median(vals))}", va="center",
                fontsize=7, color="#555")
    ax.set_yticks(range(len(arts))); ax.set_yticklabels(arts, fontsize=9)
    ax.set_xlim(0, 100); ax.set_ylim(-0.6, len(arts) - 0.4)
    ax.set_xlabel("initial score S₀ (1–100)")
    ax.axvline(50, color="gray", ls=":", lw=0.8)
    ax.set_title(f"Initial-rating (S₀) distributions — 22 artefacts × 20 runs [{a.tag}]\n"
                 "sorted by median; colour = tier (L/M/S)", fontsize=11)
    ax.legend(handles=[Patch(facecolor=TIERCOL[t], alpha=0.5, label=f"tier {t}") for t in "LMS"],
              fontsize=8, loc="lower right")
    ax.grid(axis="x", lw=0.3, alpha=0.4)
    fig.tight_layout()

    suffix = "" if a.tag == "gpt55" else f"_{a.tag}"
    out = os.path.join(ROOT, "bt_global", "results", f"s0_distributions_22{suffix}.png")
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    print("S0 std range: %.2f .. %.2f" % (
        min(np.std(v, ddof=1) for v in data.values()),
        max(np.std(v, ddof=1) for v in data.values())))


if __name__ == "__main__":
    main()
