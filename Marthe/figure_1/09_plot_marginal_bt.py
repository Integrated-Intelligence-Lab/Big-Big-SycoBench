"""Marginal pushback effect vs the BT of the single argument applied (no cumulation).

For every push step we take the *marginal* shift it caused,
    δ = sign(dir) · (Sₖ − Sₖ₋₁)      (argued direction: + = moved as pushed)
and place it at x = bt_rating of the argument shown at that turn (not a running
sum). Pooled over all 22 artefacts (the 3 arguments within one artefact have
near-identical BT, so the spread is between artefacts/validity, not within).

Each point = mean over 20 runs for one (artefact, validity, idx, turn). Left panel
colours by validity (the dose-response); right panel colours by turn (to show
whether a push's effect shrinks at later turns at the same BT = saturation).

BT: bt_global/results/arguments_bt_global.parquet ; trajectories: results/trajectories_challenge_22.csv

    python 09_plot_marginal_bt.py
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
SIGN = {"lower": -1, "raise": +1}
VCOL = {"valid": "tab:green", "invalid": "tab:red"}
TCOL = {1: "#440154", 2: "#21918c", 3: "#5ec962"}


def binned(xs, ys, nb=7):
    xs, ys = np.asarray(xs), np.asarray(ys)
    order = np.argsort(xs); bz = max(1, len(xs) // nb)
    bx, by, be = [], [], []
    for i in range(nb):
        idx = order[i * bz:(i + 1) * bz if i < nb - 1 else len(xs)]
        if len(idx) == 0:
            continue
        bx.append(xs[idx].mean()); by.append(ys[idx].mean())
        be.append(1.96 * ys[idx].std(ddof=1) / np.sqrt(len(idx)) if len(idx) > 1 else 0)
    return np.array(bx), np.array(by), np.array(be)


def main():
    bt_df = pd.read_parquet(PARQUET)
    bt = {(r.artefact_id, r.direction, r.validity, int(r.idx)): float(r.bt_rating)
          for r in bt_df.itertuples()}

    # accumulate marginal shifts per (aid, dir, val, idx, turn)
    acc = defaultdict(list)
    for r in csv.DictReader(open(CSV, encoding="utf-8")):
        aid, val, d = r["artefact"], r["validity"], r["direction"]
        s = [int(r["S0"]), int(r["t1"]), int(r["t2"]), int(r["t3"])]
        for k in (1, 2, 3):
            idx = int(r["ordering"][k - 1])
            acc[(aid, d, val, idx, k)].append(SIGN[d] * (s[k] - s[k - 1]))

    pts = []  # (bt, mean_marginal, validity, turn)
    for (aid, d, val, idx, k), v in acc.items():
        pts.append((bt[(aid, d, val, idx)], float(np.mean(v)), val, k))

    fig, (axV, axT) = plt.subplots(1, 2, figsize=(13.5, 5.4), sharex=True, sharey=True)

    # Panel A: colour by validity
    for val in ("valid", "invalid"):
        xs = [p[0] for p in pts if p[2] == val]; ys = [p[1] for p in pts if p[2] == val]
        axV.scatter(xs, ys, c=VCOL[val], s=26, alpha=0.5, edgecolor="none", label=f"{val} arg")
    bx, by, be = binned([p[0] for p in pts], [p[1] for p in pts])
    axV.errorbar(bx, by, yerr=be, color="black", lw=1.6, marker="o", ms=5, capsize=3,
                 zorder=5, label="binned mean ±95% CI")
    axV.legend(fontsize=8, loc="upper left")
    axV.set_title("A · marginal effect vs argument BT (by validity)", fontsize=10)
    axV.set_ylabel("marginal shift  δ = Sₖ − Sₖ₋₁  (argued direction)")

    # Panel B: colour by turn, per-turn binned trend
    for k in (1, 2, 3):
        xs = [p[0] for p in pts if p[3] == k]; ys = [p[1] for p in pts if p[3] == k]
        axT.scatter(xs, ys, c=TCOL[k], s=20, alpha=0.4, edgecolor="none")
        bx, by, _ = binned(xs, ys, nb=6)
        axT.plot(bx, by, "-o", color=TCOL[k], lw=1.8, ms=4, label=f"turn {k}")
    axT.legend(fontsize=8, loc="upper left", title="push step")
    axT.set_title("B · same, coloured by turn (saturation check)", fontsize=10)

    for ax in (axV, axT):
        ax.axhline(0, color="black", lw=0.6); ax.axvline(0, color="gray", ls=":", lw=0.8)
        ax.set_xlabel("BT validity of the argument pushed (not cumulative)")
        ax.grid(alpha=0.25)

    fig.suptitle("Marginal pushback effect vs single-argument BT  "
                 "[gpt-5.5, 22 artefacts, challenge direction, 20 runs]", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(HERE, "results", "marginal_bt_22.png")
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
