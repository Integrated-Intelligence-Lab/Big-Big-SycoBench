"""Pushback score-SHIFT distribution figure overlaying BOTH arms, same format as
02_plot_distribution.py: one per-artefact panel, per-integer bins, dashed means,
one shared legend.

Plots the paired per-run shift Δ = Sₖ − S0 (not raw scores). The dotted line at 0
is the S0 baseline (no movement); each arm gets its own colour family with a
light->dark gradient over the pushback turns:
  valid   -> greens (Δ1 light ... Δ3 dark)   = responsiveness
  invalid -> reds   (Δ1 light ... Δ3 dark)   = sycophancy
Single-shot core-arguments are intentionally not shown here.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re

import numpy as np
import matplotlib.pyplot as plt

PUSH_DIR = "Marthe/results/pushback"
S0_OUTPUT = "Marthe/results/initial_scores/initial_default_output.jsonl"
CYCLE_OUTPUTS = [
    "neutral_cycle1_output.jsonl",  # cycle 1 -> S1
    "neutral_cycle2_output.jsonl",  # cycle 2 -> S2
    "neutral_cycle3_output.jsonl",  # cycle 3 -> S3
]
ARTEFACTS = ["L01", "M02", "S02"]
N_RUNS = 20

ARM_CMAP = {"valid": plt.cm.Greens, "invalid": plt.cm.Reds}
SHADES = np.linspace(0.45, 0.95, 3)  # Δ1 .. Δ3 (light -> dark)


def extract_text(body):
    return "\n".join(
        c.get("text", "")
        for it in body.get("output", [])
        if it.get("type") == "message"
        for c in it.get("content", [])
        if c.get("type") == "output_text"
    )


def parse_score(text):
    m = re.search(r"-?\d+", text)
    return int(m.group()) if m else None


def load_s0(path):
    d = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        aid, run = r["custom_id"].split("_run")
        d[(aid, int(run))] = parse_score(extract_text(r["response"]["body"]))
    return d


def load_cycle(path):
    d = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        aid, val, run, c = r["custom_id"].split("|")
        d[(aid, val, int(run[1:]), int(c[1:]))] = parse_score(extract_text(r["response"]["body"]))
    return d


def main():
    s0 = load_s0(S0_OUTPUT)
    cyc = {k: load_cycle(os.path.join(PUSH_DIR, f)) for k, f in enumerate(CYCLE_OUTPUTS, start=1)}

    def delta(aid, arm, k):
        return np.array([cyc[k][(aid, arm, r, k)] - s0[(aid, r)] for r in range(N_RUNS)], dtype=float)

    # shared per-integer Δ bins across all panels/arms
    allv = np.concatenate([delta(a, arm, k) for a in ARTEFACTS for arm in ARM_CMAP for k in (1, 2, 3)])
    lo, hi = int(np.floor(allv.min())) - 1, int(np.ceil(allv.max())) + 1
    bins = np.arange(lo - 0.5, hi + 1.5, 1)

    fig, axes = plt.subplots(len(ARTEFACTS), 1, figsize=(10, 2.6 * len(ARTEFACTS)), sharex=True)

    for ax, aid in zip(axes, ARTEFACTS):
        for arm in ["valid", "invalid"]:
            cmap = ARM_CMAP[arm]
            for k in (1, 2, 3):
                vals = delta(aid, arm, k)
                color = cmap(SHADES[k - 1])
                ax.hist(vals, bins=bins, color=color, alpha=0.45,
                        histtype="stepfilled", edgecolor=color, linewidth=1.0,
                        label=f"{arm} Δ{k}")
                ax.axvline(vals.mean(), color=color, linestyle="--", linewidth=1.0)
        ax.axvline(0, color="gray", linestyle=":", linewidth=1.0)  # S0 baseline
        ax.set_xlim(lo, hi)
        ax.set_ylabel("count")
        ax.yaxis.get_major_locator().set_params(integer=True)
        ax.set_title(aid, fontsize=10, fontweight="bold")

    axes[0].legend(fontsize=7, loc="upper left", ncol=2)
    axes[-1].set_xlabel("score shift  Δ = Sₖ − S0  (per-integer bins; 0 = no movement)")
    fig.suptitle("Pushback score-shift distribution per artefact — valid vs invalid, Δ1→Δ3 gradient "
                 "(neutral/default, N=20)", fontsize=11)
    fig.tight_layout()

    out = os.path.join(PUSH_DIR, "pushback_distribution_arms.png")
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
