"""cycle_trajectory_bt_22 with the shift made positive (|Δ|), 22 panels.

Identical to 08_plot_cycle_trajectory_bt.py (per-artefact panels, x = cumulative BT
pushed, markers = turn 1/2/3) except y is the magnitude of the shift so that
push-down and push-up artefacts are on the same (positive) axis and can be compared:

  y = sign(dir) · (Sₖ − S0)        (= |Sₖ − S0| at the panel-mean level; flips the
                                     push-down panels up without per-run rectification)

    python 12_plot_cycle_trajectory_bt_absshift.py
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
SIGN = {"lower": -1, "raise": +1}
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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="gpt55")
    a = ap.parse_args()
    suffix = "" if a.tag == "gpt55" else f"_{a.tag}"
    csv_path = os.path.join(HERE, "results", f"trajectories_challenge_22{suffix}.csv")

    bt = {(r.artefact_id, r.direction, r.validity, int(r.idx)): float(r.bt_rating)
          for r in pd.read_parquet(PARQUET).itertuples()}

    cumbt = defaultdict(list)
    deltas = defaultdict(list)
    s0_by_art = defaultdict(list)
    dir_by_art = {}
    for r in csv.DictReader(open(csv_path, encoding="utf-8")):
        aid, val, d = r["artefact"], r["validity"], r["direction"]
        dir_by_art[aid] = d
        s0 = int(r["S0"]); s0_by_art[aid].append(s0)
        bseq = [bt[(aid, d, val, int(c))] for c in r["ordering"]]
        cumbt[(aid, val)].append([0.0, bseq[0], bseq[0] + bseq[1], sum(bseq)])
        deltas[(aid, val)].append([0.0] + [SIGN[d] * (int(r[t]) - s0) for t in ("t1", "t2", "t3")])

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
            x = np.array(cumbt[(aid, arm)]).mean(axis=0)
            ymean, lo, hi = mean_ci(np.array(deltas[(aid, arm)], dtype=float))
            ax.plot(x, ymean, "-o", color=sty["color"], lw=1.8, ms=4)
            ax.fill_between(x, lo, hi, color=sty["color"], alpha=0.15)
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
            ax.set_ylabel("|Δ| = |Sₖ − S0|", fontsize=8)
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

    fig.suptitle(f"Pushback shift MAGNITUDE |Δ| vs cumulative BT pushed  "
                 f"[{a.tag}, 22 artefacts, challenge direction, push-down mirrored up, 95% CI]",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(HERE, "results", f"cycle_trajectory_bt_22_absshift{suffix}.png")
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
