"""Absolute-value pushback trajectory, aggregated to compare raise vs lower pushes.

cycle_trajectory_bt_22 plots raw signed Δ per artefact, so push-down (negative) and
push-up (positive) can't be overlaid. Here we take |Δ| = |Sₖ − S0| (magnitude of
movement) and |cumulative BT| so both directions land in the same positive quadrant,
then aggregate across artefacts within each (direction, validity) group:

  x = mean |cumulative BT pushed|   y = mean |Sₖ − S0|  (±95% bootstrap CI)

Two panels (valid / invalid); within each, lower-pushes vs raise-pushes overlaid, so
you can see whether the model responds symmetrically to upward and downward pushes.

NOTE: lower- and raise-challenge artefacts are different populations (high-S0 vs
low-S0), so a gap between the curves mixes a direction effect with that selection.

    python 11_plot_cycle_trajectory_bt_abs.py
"""
import csv
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CSV = os.path.join(HERE, "results", "trajectories_challenge_22.csv")
PARQUET = os.path.join(ROOT, "bt_global", "results", "arguments_bt_global.parquet")
DCOL = {"lower": "tab:blue", "raise": "tab:orange"}
DLAB = {"lower": "push down (good artefacts)", "raise": "push up (bad artefacts)"}


def boot_ci(v):
    v = np.asarray(v, float)
    if len(v) < 2 or np.allclose(v, v[0]):
        return v.mean(), v.mean(), v.mean()
    rng = np.random.default_rng(0)
    b = rng.choice(v, size=(2000, len(v)), replace=True).mean(axis=1)
    return v.mean(), *np.percentile(b, [2.5, 97.5])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="gpt55")
    a = ap.parse_args()
    suffix = "" if a.tag == "gpt55" else f"_{a.tag}"
    csv_path = os.path.join(HERE, "results", f"trajectories_challenge_22{suffix}.csv")

    bt = {(r.artefact_id, r.direction, r.validity, int(r.idx)): float(r.bt_rating)
          for r in pd.read_parquet(PARQUET).itertuples()}

    # group (direction, validity) -> per turn: lists of |Δ| (y) and |cumBT| (x)
    absdelta = defaultdict(lambda: defaultdict(list))
    abscumbt = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(csv_path, encoding="utf-8")):
        aid, val, d = r["artefact"], r["validity"], r["direction"]
        s = [int(r["S0"]), int(r["t1"]), int(r["t2"]), int(r["t3"])]
        idxs = [int(c) for c in r["ordering"]]
        bseq = [bt[(aid, d, val, i)] for i in idxs]
        cb = [0.0, bseq[0], bseq[0] + bseq[1], sum(bseq)]
        for k in (1, 2, 3):
            absdelta[(d, val)][k].append(abs(s[k] - s[0]))
            abscumbt[(d, val)][k].append(abs(cb[k]))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharex=True, sharey=True)
    for ax, val in zip(axes, ("valid", "invalid")):
        for d in ("lower", "raise"):
            xs, ys, los, his = [0.0], [0.0], [0.0], [0.0]
            for k in (1, 2, 3):
                xs.append(np.mean(abscumbt[(d, val)][k]))
                m, lo, hi = boot_ci(absdelta[(d, val)][k])
                ys.append(m); los.append(lo); his.append(hi)
            ls = "-" if d == "lower" else "--"
            ax.plot(xs, ys, ls, marker="o", color=DCOL[d], lw=2, ms=5, label=DLAB[d])
            ax.fill_between(xs, los, his, color=DCOL[d], alpha=0.15)
            for k in (1, 2, 3):
                ax.annotate(str(k), (xs[k], ys[k]), textcoords="offset points",
                            xytext=(4, 3), fontsize=7, color=DCOL[d])
        ax.set_title(f"{val} arguments", fontsize=11, fontweight="bold")
        ax.set_xlabel("|cumulative BT pushed|")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("mean |Sₖ − S0|  (magnitude of shift)")

    fig.suptitle(f"Absolute pushback magnitude vs |cumulative BT|: raise vs lower pushes  "
                 f"[{a.tag}, 22 artefacts, 95% CI]", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(HERE, "results", f"cycle_trajectory_bt_abs_22{suffix}.png")
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
