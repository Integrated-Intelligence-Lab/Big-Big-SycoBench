"""Multi-turn pushback shift trajectory per artefact, for ALL 22 artefacts.

Same idea as scripts/06_plot_cycles.py (the pushback cycle_trajectory) but driven
by the joined challenge-direction table results/trajectories_challenge_22.csv and
extended to every artefact. Per (run, ordering) we take Δ = Sₖ − S0 (paired, so the
baseline is removed); each arm pools the 20 runs × 3 orderings = 60 trajectories.
One panel per artefact (small multiples, sorted by S0), two lines:
  valid   = responsiveness control (green)
  invalid = sycophancy signal     (red)
mean Δ with a 95% bootstrap CI band, anchored at (turn 0, 0).

    python 07_plot_cycle_trajectory_22.py
"""
import csv
import math
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "results", "trajectories_challenge_22.csv")
ARM_STYLE = {
    "valid":   {"color": "tab:green", "label": "valid args (responsiveness)"},
    "invalid": {"color": "tab:red",   "label": "invalid args (sycophancy)"},
}


def mean_ci(mat):
    """Per-column mean and 95% bootstrap CI for a units×turns matrix (NaN-safe)."""
    rng = np.random.default_rng(0)
    means, los, his = [], [], []
    for j in range(mat.shape[1]):
        v = mat[:, j]; v = v[~np.isnan(v)]
        if len(v) == 0:
            means += [np.nan]; los += [np.nan]; his += [np.nan]; continue
        m = v.mean(); means.append(m)
        if np.allclose(v, v[0]):
            los.append(m); his.append(m)
        else:
            boot = rng.choice(v, size=(2000, len(v)), replace=True).mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            los.append(lo); his.append(hi)
    return np.array(means), np.array(los), np.array(his)


def main():
    # rows[(aid, validity)] -> list of [Δ0=0, Δ1, Δ2, Δ3]
    rows = defaultdict(list)
    s0_by_art = defaultdict(list)
    dir_by_art = {}
    for r in csv.DictReader(open(CSV, encoding="utf-8")):
        aid, val = r["artefact"], r["validity"]
        s0 = int(r["S0"])
        d = [0.0] + [int(r[t]) - s0 for t in ("t1", "t2", "t3")]
        rows[(aid, val)].append(d)
        s0_by_art[aid].append(s0)
        dir_by_art[aid] = r["direction"]

    s0med = {a: int(np.median(v)) for a, v in s0_by_art.items()}
    arts = sorted(s0med, key=lambda a: -s0med[a])      # high S0 (push down) first
    dir_label = {"lower": "good → push down", "raise": "bad → push up"}

    ncols = 5
    nrows = math.ceil(len(arts) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.7 * nrows),
                             sharey=False)
    axes = axes.ravel()
    turns = [0, 1, 2, 3]

    for i, aid in enumerate(arts):
        ax = axes[i]
        for arm, sty in ARM_STYLE.items():
            mat = np.array(rows[(aid, arm)], dtype=float)
            mean, lo, hi = mean_ci(mat)
            ax.plot(turns, mean, "-o", color=sty["color"], lw=1.8, ms=4)
            ax.fill_between(turns, lo, hi, color=sty["color"], alpha=0.15)
            ax.annotate(f"{mean[-1]:+.1f}", (turns[-1], mean[-1]),
                        textcoords="offset points", xytext=(4, 0),
                        fontsize=7, color=sty["color"])
        ax.axhline(0, color="gray", ls=":", lw=0.8)
        ax.set_title(f"{aid}  ({dir_label[dir_by_art[aid]]}, S₀≈{s0med[aid]})",
                     fontsize=8.5, fontweight="bold")
        ax.set_xticks(turns)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
        if i % ncols == 0:
            ax.set_ylabel("Δ = Sₖ − S0", fontsize=8)
        if i // ncols == nrows - 1 or i + ncols >= len(arts):
            ax.set_xlabel("pushback turn", fontsize=8)

    for j in range(len(arts), len(axes)):       # hide unused panels
        axes[j].axis("off")

    # one shared legend in the first empty slot (or figure-level if none)
    handles = [plt.Line2D([], [], color=s["color"], marker="o", lw=1.8, label=s["label"])
               for s in ARM_STYLE.values()]
    if len(arts) < len(axes):
        axes[len(arts)].legend(handles=handles, loc="center", fontsize=9, frameon=False)
    else:
        fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9)

    fig.suptitle("Multi-turn pushback: mean score shift Δ = Sₖ − S0 by argument validity  "
                 "[gpt-5.5, 22 artefacts, challenge direction, 20 runs × 3 orderings, 95% CI]",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(HERE, "results", "cycle_trajectory_22.png")
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
