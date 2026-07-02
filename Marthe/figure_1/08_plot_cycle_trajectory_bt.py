"""Pushback shift trajectory with the x-axis = cumulative BT of arguments pushed.

Like 07_plot_cycle_trajectory_22.py, but instead of the discrete pushback turn,
the x-position of each step is the cumulative BT validity of the arguments shown
so far (turn k -> sum of bt_rating of the first k arguments in that ordering).
y stays Δ = Sₖ − S0. So a line that climbs steeply at small |x| means the model
moves a lot for little argument validity (sycophancy); a shallow line extending
far right means the shift is proportional to the validity pushed (responsiveness).

BT ratings: bt_global/results/arguments_bt_global.parquet  (per artefact|dir|val|idx).
Trajectories: results/trajectories_challenge_22.csv (challenge direction, 22 artefacts).

    python 08_plot_cycle_trajectory_bt.py
"""
import csv
import math
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CSV = os.path.join(HERE, "results", "trajectories_challenge_22.csv")
PARQUET = os.path.join(ROOT, "bt_global", "results", "arguments_bt_global.parquet")
ARM_STYLE = {
    "valid":   {"color": "tab:green", "label": "valid args (responsiveness)"},
    "invalid": {"color": "tab:red",   "label": "invalid args (sycophancy)"},
}


def mean_ci(mat):
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
    bt_df = pd.read_parquet(PARQUET)
    bt = {(r.artefact_id, r.direction, r.validity, int(r.idx)): float(r.bt_rating)
          for r in bt_df.itertuples()}

    # per (aid, val): lists of cumulative-BT arrays and Δ arrays (one per run×ordering)
    cumbt = defaultdict(list)
    deltas = defaultdict(list)
    s0_by_art = defaultdict(list)
    dir_by_art = {}
    for r in csv.DictReader(open(CSV, encoding="utf-8")):
        aid, val, d = r["artefact"], r["validity"], r["direction"]
        dir_by_art[aid] = d
        s0 = int(r["S0"]); s0_by_art[aid].append(s0)
        idxs = [int(ch) for ch in r["ordering"]]
        bseq = [bt[(aid, d, val, i)] for i in idxs]
        cumbt[(aid, val)].append([0.0, bseq[0], bseq[0] + bseq[1], sum(bseq)])
        deltas[(aid, val)].append([0.0] + [int(r[t]) - s0 for t in ("t1", "t2", "t3")])

    s0med = {a: int(np.median(v)) for a, v in s0_by_art.items()}
    arts = sorted(s0med, key=lambda a: -s0med[a])
    dir_label = {"lower": "good → push down", "raise": "bad → push up"}

    ncols = 5
    nrows = math.ceil(len(arts) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.7 * nrows))
    axes = axes.ravel()

    for i, aid in enumerate(arts):
        ax = axes[i]
        for arm, sty in ARM_STYLE.items():
            x = np.array(cumbt[(aid, arm)]).mean(axis=0)        # mean cumBT per turn
            ymean, lo, hi = mean_ci(np.array(deltas[(aid, arm)], dtype=float))
            ax.plot(x, ymean, "-o", color=sty["color"], lw=1.8, ms=4)
            ax.fill_between(x, lo, hi, color=sty["color"], alpha=0.15)
            # mark turn numbers along the line
            for k in (1, 2, 3):
                ax.annotate(str(k), (x[k], ymean[k]), textcoords="offset points",
                            xytext=(3, 3), fontsize=6, color=sty["color"])
        ax.axhline(0, color="gray", ls=":", lw=0.8)
        ax.axvline(0, color="gray", ls=":", lw=0.8)
        ax.set_title(f"{aid}  ({dir_label[dir_by_art[aid]]}, S₀≈{s0med[aid]})",
                     fontsize=8.5, fontweight="bold")
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
        if i % ncols == 0:
            ax.set_ylabel("Δ = Sₖ − S0", fontsize=8)
        if i + ncols >= len(arts):
            ax.set_xlabel("cumulative BT pushed", fontsize=8)

    for j in range(len(arts), len(axes)):
        axes[j].axis("off")

    handles = [plt.Line2D([], [], color=s["color"], marker="o", lw=1.8, label=s["label"])
               for s in ARM_STYLE.values()]
    if len(arts) < len(axes):
        axes[len(arts)].legend(handles=handles, loc="center", fontsize=9, frameon=False,
                               title="markers = turn 1/2/3")
    else:
        fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9)

    fig.suptitle("Pushback shift Δ vs cumulative BT of arguments pushed  "
                 "[gpt-5.5, 22 artefacts, challenge direction, 20 runs × 3 orderings, 95% CI]",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(HERE, "results", "cycle_trajectory_bt_22.png")
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
