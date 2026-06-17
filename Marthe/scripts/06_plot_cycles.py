"""Plot the multi-turn pushback trajectory S0 -> S1 -> S2 -> S3 per artefact.

Turn 0 is the default/neutral S0 (original prompt); turns 1-3 come from the
chained cycle outputs. One panel per artefact, two lines (valid = responsiveness
control, invalid = sycophancy signal), mean +/- 1 sd band over the 20 runs.
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
# Chained cycle outputs in turn order (1, 2, 3).
CYCLE_OUTPUTS = [
    "batch_6a3143b187688190a6f3c30737a4cfe6_output.jsonl",  # cycle 1 -> S1
    "batch_6a31522f8c608190b1d30a40bc800f95_output.jsonl",  # cycle 2 -> S2
    "batch_6a31610752ac81909e207c0e4d60a319_output.jsonl",  # cycle 3 -> S3
]
ARTEFACTS = ["L01", "M02", "S02"]
N_RUNS = 20
ARM_STYLE = {
    "valid":   {"color": "tab:green", "label": "valid args (responsiveness)"},
    "invalid": {"color": "tab:red",   "label": "invalid args (sycophancy)"},
}
DIRECTION = {"L01": "good → push down", "M02": "bad → push up", "S02": "good → push down"}


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
    cyc = [load_cycle(os.path.join(PUSH_DIR, f)) for f in CYCLE_OUTPUTS]

    fig, axes = plt.subplots(1, len(ARTEFACTS), figsize=(13, 4.2), sharey=False)
    turns = [0, 1, 2, 3]

    for ax, aid in zip(axes, ARTEFACTS):
        for arm, sty in ARM_STYLE.items():
            # rows: run, cols: turn; turn 0 = S0, turns 1-3 = cycle outputs
            mat = np.array([
                [s0[(aid, r)]] + [cyc[k][(aid, arm, r, k + 1)] for k in range(3)]
                for r in range(N_RUNS)
            ], dtype=float)
            mean, sd = mat.mean(0), mat.std(0)
            ax.plot(turns, mean, "-o", color=sty["color"], label=sty["label"], lw=2, ms=5)
            ax.fill_between(turns, mean - sd, mean + sd, color=sty["color"], alpha=0.15)
            ax.annotate(f"{mean[-1]:.1f}", (turns[-1], mean[-1]),
                        textcoords="offset points", xytext=(6, 0), fontsize=8, color=sty["color"])
        ax.axhline(np.mean([s0[(aid, r)] for r in range(N_RUNS)]),
                   color="gray", ls=":", lw=0.8)  # S0 reference
        ax.set_title(f"{aid}  ({DIRECTION[aid]})", fontsize=10, fontweight="bold")
        ax.set_xlabel("pushback turn")
        ax.set_xticks(turns)
        ax.set_ylim(1, 100)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("score (1–100)")
    axes[0].legend(fontsize=8, loc="center left")
    fig.suptitle("Multi-turn pushback: score trajectory by argument validity (neutral/default, N=20)", fontsize=12)
    fig.tight_layout()

    out = os.path.join(PUSH_DIR, "cycle_trajectory.png")
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
