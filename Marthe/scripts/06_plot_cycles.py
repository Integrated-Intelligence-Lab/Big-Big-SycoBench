"""Plot the multi-turn pushback SHIFT trajectory per artefact.

Per run we take the paired delta Δ = Sₖ − S0 (S0 is that run's own initial score),
so the baseline is removed and only the pushback-induced movement is shown. One
panel per artefact, two lines (valid = responsiveness control, invalid =
sycophancy signal): mean Δ with a 95% bootstrap CI band, anchored at (turn 0, 0).
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


def mean_ci(mat):
    """Per-column mean and 95% bootstrap CI for a runs×turns matrix (NaN-safe)."""
    rng = np.random.default_rng(0)
    means, los, his = [], [], []
    for j in range(mat.shape[1]):
        v = mat[:, j]
        v = v[~np.isnan(v)]
        if len(v) == 0:
            means += [np.nan]; los += [np.nan]; his += [np.nan]; continue
        m = v.mean(); means.append(m)
        if np.allclose(v, v[0]):                 # degenerate (e.g. turn 0 = all zeros)
            los.append(m); his.append(m)
        else:
            boot = rng.choice(v, size=(2000, len(v)), replace=True).mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            los.append(lo); his.append(hi)
    return np.array(means), np.array(los), np.array(his)


def delta_matrix(s0, cyc, aid, arm):
    """rows=run, cols=[Δ0=0, Δ1, Δ2, Δ3] with Δk = Sk − S0 (paired per run)."""
    rows = []
    for r in range(N_RUNS):
        s = [s0[(aid, r)]] + [cyc[k][(aid, arm, r, k + 1)] for k in range(3)]
        rows.append([sk - s[0] for sk in s])
    return np.array(rows, dtype=float)


def main():
    s0 = load_s0(S0_OUTPUT)
    cyc = [load_cycle(os.path.join(PUSH_DIR, f)) for f in CYCLE_OUTPUTS]

    fig, axes = plt.subplots(1, len(ARTEFACTS), figsize=(13, 4.2), sharey=False)
    turns = [0, 1, 2, 3]

    for ax, aid in zip(axes, ARTEFACTS):
        for arm, sty in ARM_STYLE.items():
            dmat = delta_matrix(s0, cyc, aid, arm)
            mean, lo, hi = mean_ci(dmat)
            ax.plot(turns, mean, "-o", color=sty["color"], label=sty["label"], lw=2, ms=5)
            ax.fill_between(turns, lo, hi, color=sty["color"], alpha=0.15)
            ax.annotate(f"{mean[-1]:+.1f}", (turns[-1], mean[-1]),
                        textcoords="offset points", xytext=(6, 0), fontsize=8, color=sty["color"])
        ax.axhline(0, color="gray", ls=":", lw=0.8)  # S0 baseline
        ax.set_title(f"{aid}  ({DIRECTION[aid]})", fontsize=10, fontweight="bold")
        ax.set_xlabel("pushback turn")
        ax.set_xticks(turns)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("score shift  Δ = Sₖ − S0")
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle("Multi-turn pushback: mean score shift Δ from S0 by argument validity "
                 "(neutral/default, N=20, 95% CI)", fontsize=12)
    fig.tight_layout()

    out = os.path.join(PUSH_DIR, "cycle_trajectory.png")
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
