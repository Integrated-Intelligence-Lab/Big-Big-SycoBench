"""Pushback score-SHIFT distribution figure, same format as 02_plot_distribution.py:
one per-artefact panel, per-integer bins, dashed means, one shared legend.

Plots the paired per-run shift Δ = Sₖ − S0 (not raw scores): the multi-turn turns
Δ1 → Δ2 → Δ3 in a blue gradient (light → dark) plus the single-shot core-arguments
Δ in a contrasting colour. The dotted line at 0 is the S0 baseline (no movement).
Generated separately for the valid and invalid arms.
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
SINGLESHOT_OUTPUT = "neutral_singleshot_output.jsonl"  # method test (core args)
ARTEFACTS = ["L01", "M02", "S02"]
N_RUNS = 20

# Blue gradient for the cycle shifts Δ1..Δ3: light -> dark.
BLUES = plt.cm.Blues(np.linspace(0.45, 0.95, 3))
CORE_COLOR = "tab:orange"


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


def load_piped(path):
    """custom_id {aid}|{val}|r{run}[|c{k}] -> score, keyed by the full tuple."""
    d = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        parts = r["custom_id"].split("|")
        aid, val, run = parts[0], parts[1], int(parts[2][1:])
        k = int(parts[3][1:]) if len(parts) > 3 else 0
        d[(aid, val, run, k)] = parse_score(extract_text(r["response"]["body"]))
    return d


def main():
    s0 = load_s0(S0_OUTPUT)
    cyc = {k: load_piped(os.path.join(PUSH_DIR, f)) for k, f in enumerate(CYCLE_OUTPUTS, start=1)}
    single = load_piped(os.path.join(PUSH_DIR, SINGLESHOT_OUTPUT))

    def deltas(aid, arm):  # -> dict label -> per-run Δ array
        base = [s0[(aid, r)] for r in range(N_RUNS)]
        return {
            "Δ1 (turn 1)":  np.array([cyc[1][(aid, arm, r, 1)] - base[r] for r in range(N_RUNS)], float),
            "Δ2 (turn 2)":  np.array([cyc[2][(aid, arm, r, 2)] - base[r] for r in range(N_RUNS)], float),
            "Δ3 (turn 3)":  np.array([cyc[3][(aid, arm, r, 3)] - base[r] for r in range(N_RUNS)], float),
            "single-shot core args (Δ)":
                            np.array([single[(aid, arm, r, 0)] - base[r] for r in range(N_RUNS)], float),
        }
    COLORS = [BLUES[0], BLUES[1], BLUES[2], CORE_COLOR]

    for arm in ["invalid", "valid"]:
        # shared per-integer Δ bins across the three panels of this figure
        allv = np.concatenate([np.concatenate(list(deltas(a, arm).values())) for a in ARTEFACTS])
        lo, hi = int(np.floor(allv.min())) - 1, int(np.ceil(allv.max())) + 1
        bins = np.arange(lo - 0.5, hi + 1.5, 1)

        fig, axes = plt.subplots(len(ARTEFACTS), 1, figsize=(10, 2.6 * len(ARTEFACTS)), sharex=True)
        for ax, aid in zip(axes, ARTEFACTS):
            for (label, vals), color in zip(deltas(aid, arm).items(), COLORS):
                ax.hist(vals, bins=bins, color=color, alpha=0.5,
                        histtype="stepfilled", edgecolor=color, linewidth=1.1, label=label)
                ax.axvline(vals.mean(), color=color, linestyle="--", linewidth=1.2)
            ax.axvline(0, color="gray", linestyle=":", linewidth=1.0)  # S0 baseline
            ax.set_xlim(lo, hi)
            ax.set_ylabel("count")
            ax.yaxis.get_major_locator().set_params(integer=True)
            ax.set_title(aid, fontsize=10, fontweight="bold")

        axes[0].legend(fontsize=8, loc="upper left")
        axes[-1].set_xlabel("score shift  Δ = Sₖ − S0  (per-integer bins; 0 = no movement)")
        fig.suptitle(f"Pushback score-shift distribution per artefact — {arm} arguments "
                     f"(multi-turn Δ1→Δ3 + single-shot, neutral/default, N=20)", fontsize=11)
        fig.tight_layout()

        out = os.path.join(PUSH_DIR, f"pushback_distribution_{arm}.png")
        fig.savefig(out, dpi=150)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
