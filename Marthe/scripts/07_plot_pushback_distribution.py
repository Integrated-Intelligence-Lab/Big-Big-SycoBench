"""Pushback score-distribution figure, same format as 02_plot_distribution.py:
one per-artefact panel, per-integer bins on the 1-100 scale, dashed means, one
shared legend.

Overlays the multi-turn trajectory S0 -> S1 -> S2 -> S3 in a blue gradient
(light = initial, dark = after 3 turns) plus the single-shot core-arguments S1
in a contrasting colour. Generated separately for the valid and invalid arms.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re

import numpy as np
import matplotlib.pyplot as plt

PUSH_DIR = "Marthe/results/pushback"
S0_OUTPUT = "Marthe/results/initial_scores/batch_6a2ab6ba613c8190b307db0984f42a29_output.jsonl"
CYCLE_OUTPUTS = [
    "batch_6a3143b187688190a6f3c30737a4cfe6_output.jsonl",  # cycle 1 -> S1
    "batch_6a31522f8c608190b1d30a40bc800f95_output.jsonl",  # cycle 2 -> S2
    "batch_6a31610752ac81909e207c0e4d60a319_output.jsonl",  # cycle 3 -> S3
]
SINGLESHOT_OUTPUT = "batch_6a3137e98f1481908bb2c8de78ac8060_output.jsonl"  # method test (core args)
ARTEFACTS = ["L01", "M02", "S02"]
N_RUNS = 20

# Blue gradient for the cycle turns: light (S0) -> dark (S3).
BLUES = plt.cm.Blues(np.linspace(0.35, 0.95, 4))
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
    cyc = {}
    for k, f in enumerate(CYCLE_OUTPUTS, start=1):
        cyc[k] = load_piped(os.path.join(PUSH_DIR, f))
    single = load_piped(os.path.join(PUSH_DIR, SINGLESHOT_OUTPUT))

    bins = np.arange(0.5, 101.5, 1)

    for arm in ["invalid", "valid"]:
        fig, axes = plt.subplots(len(ARTEFACTS), 1, figsize=(10, 2.6 * len(ARTEFACTS)), sharex=True)
        for ax, aid in zip(axes, ARTEFACTS):
            series = [
                ("S0 (initial)", [s0[(aid, r)] for r in range(N_RUNS)], BLUES[0]),
                ("S1 (turn 1)", [cyc[1][(aid, arm, r, 1)] for r in range(N_RUNS)], BLUES[1]),
                ("S2 (turn 2)", [cyc[2][(aid, arm, r, 2)] for r in range(N_RUNS)], BLUES[2]),
                ("S3 (turn 3)", [cyc[3][(aid, arm, r, 3)] for r in range(N_RUNS)], BLUES[3]),
                ("single-shot core args (S1)", [single[(aid, arm, r, 0)] for r in range(N_RUNS)], CORE_COLOR),
            ]
            for label, scores, color in series:
                scores = np.array(scores, dtype=float)
                ax.hist(scores, bins=bins, color=color, alpha=0.5,
                        histtype="stepfilled", edgecolor=color, linewidth=1.1, label=label)
                ax.axvline(scores.mean(), color=color, linestyle="--", linewidth=1.2)
            ax.set_xlim(1, 100)
            ax.set_ylabel("count")
            ax.yaxis.get_major_locator().set_params(integer=True)
            ax.set_title(aid, fontsize=10, fontweight="bold")

        axes[0].legend(fontsize=8, loc="upper left")
        axes[-1].set_xlabel("score (1-100, per-integer bins)")
        axes[-1].set_xticks(range(0, 101, 10))
        fig.suptitle(f"Pushback score distribution per artefact — {arm} arguments "
                     f"(multi-turn S0→S3 + single-shot, neutral/default, N=20)", fontsize=11)
        fig.tight_layout()

        out = os.path.join(PUSH_DIR, f"pushback_distribution_{arm}.png")
        fig.savefig(out, dpi=150)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
