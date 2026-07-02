"""Normalised marginal / cumulative pushback effect vs single-argument BT.

Like marginal_bt_22 but the shift is divided by σ₀, the std of that artefact's
initial-score (S0) distribution (σ₀=0 -> the minimum nonzero σ₀ across the 22).

  left  : marginal   (Sₖ − Sₖ₋₁)/σ₀   vs argument BT, coloured by validity
  right : cumulative (Sₖ − S0)/σ₀     vs argument BT, coloured by turn

Both in the argued direction (+ = moved as pushed). x = BT of the single argument
applied at that turn (not cumulative). Each point = mean over 20 runs for one
(artefact, validity, idx, turn).

    python 10_plot_marginal_bt_norm.py
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
    bt = {(r.artefact_id, r.direction, r.validity, int(r.idx)): float(r.bt_rating)
          for r in pd.read_parquet(PARQUET).itertuples()}

    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    # sigma0 per artefact (std of S0 over the 20 runs), floor = min nonzero
    s0run = defaultdict(dict)
    for r in rows:
        s0run[r["artefact"]][int(r["run"])] = int(r["S0"])
    sig = {a: np.std(list(v.values()), ddof=1) for a, v in s0run.items()}
    floor = min(s for s in sig.values() if s > 0)
    sig = {a: (s if s > 0 else floor) for a, s in sig.items()}
    print(f"sigma0 floor (min nonzero) = {floor:.4f}; floored artefacts: "
          f"{[a for a, v in s0run.items() if np.std(list(v.values()), ddof=1) == 0]}")

    marg = defaultdict(list)   # (aid,d,val,idx,turn) -> [marginal/σ0]
    cum = defaultdict(list)    # (aid,d,val,idx,turn) -> [cumulative/σ0]
    for r in rows:
        aid, val, d = r["artefact"], r["validity"], r["direction"]
        s = [int(r["S0"]), int(r["t1"]), int(r["t2"]), int(r["t3"])]
        for k in (1, 2, 3):
            idx = int(r["ordering"][k - 1])
            marg[(aid, d, val, idx, k)].append(SIGN[d] * (s[k] - s[k - 1]) / sig[aid])
            cum[(aid, d, val, idx, k)].append(SIGN[d] * (s[k] - s[0]) / sig[aid])

    pm = [(bt[(a, d, v, i)], float(np.mean(x)), v, k) for (a, d, v, i, k), x in marg.items()]
    pc = [(bt[(a, d, v, i)], float(np.mean(x)), v, k) for (a, d, v, i, k), x in cum.items()]

    fig, (axM, axC) = plt.subplots(1, 2, figsize=(13.5, 5.4), sharex=True)

    # left: marginal, by validity
    for val in ("valid", "invalid"):
        xs = [p[0] for p in pm if p[2] == val]; ys = [p[1] for p in pm if p[2] == val]
        axM.scatter(xs, ys, c=VCOL[val], s=24, alpha=0.5, edgecolor="none", label=f"{val} arg")
    bx, by, be = binned([p[0] for p in pm], [p[1] for p in pm])
    axM.errorbar(bx, by, yerr=be, color="black", lw=1.6, marker="o", ms=5, capsize=3,
                 zorder=5, label="binned mean ±95% CI")
    axM.legend(fontsize=8, loc="upper left")
    axM.set_title("marginal  (Sₖ − Sₖ₋₁) / σ₀", fontsize=10)
    axM.set_ylabel("normalized shift  (σ₀ units, argued direction)")

    # right: cumulative, by turn
    for k in (1, 2, 3):
        xs = [p[0] for p in pc if p[3] == k]; ys = [p[1] for p in pc if p[3] == k]
        axC.scatter(xs, ys, c=TCOL[k], s=18, alpha=0.4, edgecolor="none")
        bx, by, _ = binned(xs, ys, nb=6)
        axC.plot(bx, by, "-o", color=TCOL[k], lw=1.8, ms=4, label=f"turn {k}")
    axC.legend(fontsize=8, loc="upper left", title="push step")
    axC.set_title("cumulative  (Sₖ − S0) / σ₀", fontsize=10)

    for ax in (axM, axC):
        ax.axhline(0, color="black", lw=0.6); ax.axvline(0, color="gray", ls=":", lw=0.8)
        ax.set_xlabel("BT validity of the argument pushed")
        ax.grid(alpha=0.25)

    fig.suptitle("Normalised pushback effect vs single-argument BT  "
                 "[gpt-5.5, 22 artefacts, Δ ÷ σ₀ of initial distribution]", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(HERE, "results", "marginal_bt_norm_22.png")
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
