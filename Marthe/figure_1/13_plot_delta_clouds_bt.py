"""The point clouds the (α, β) regression will be fit to, one panel per horizon.

z = d · (Sₖ − S0) / σ₀   (argued/push direction, normalised by the artefact's
baseline-score std) vs argument BT, for k = 1, 2, 3.

One bold point per argument = mean over the 20 runs (this is what the line is fit
to); faint points behind = the 20 individual runs. x is the BT of the FIRST argument
in the trajectory (within an artefact-validity the three args have ~equal BT, so the
same 132 points just rise vertically as the horizon grows). Pooled over all 22
artefacts, challenge direction only. A reference OLS line + slope are drawn per panel.

σ₀ = std of the 20 baseline scores; σ₀=0 -> min nonzero σ₀ across the 22 (M08).

    python 13_plot_delta_clouds_bt.py
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


def main():
    bt = {(r.artefact_id, r.direction, r.validity, int(r.idx)): float(r.bt_rating)
          for r in pd.read_parquet(PARQUET).itertuples()}
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))

    s0run = defaultdict(dict)
    for r in rows:
        s0run[r["artefact"]][int(r["run"])] = int(r["S0"])
    sig = {a: np.std(list(v.values()), ddof=1) for a, v in s0run.items()}
    floor = min(s for s in sig.values() if s > 0)
    sig = {a: (s if s > 0 else floor) for a, s in sig.items()}

    # per (aid, val, first_idx) -> per turn list of z over runs; plus x=BT(first arg)
    runz = {1: defaultdict(list), 2: defaultdict(list), 3: defaultdict(list)}
    xval, vval = {}, {}
    for r in rows:
        aid, val, d = r["artefact"], r["validity"], r["direction"]
        s = [int(r["S0"]), int(r["t1"]), int(r["t2"]), int(r["t3"])]
        fi = int(r["ordering"][0])
        key = (aid, val, fi)
        xval[key] = bt[(aid, d, val, fi)]; vval[key] = val
        for k in (1, 2, 3):
            runz[k][key].append(SIGN[d] * (s[k] - s[0]) / sig[aid])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    for k, ax in zip((1, 2, 3), axes):
        # faint run-level + bold per-argument means
        mx, my, mv = [], [], []
        for key, zs in runz[k].items():
            x = xval[key]
            ax.scatter([x] * len(zs), zs, c=VCOL[vval[key]], s=8, alpha=0.10, edgecolor="none")
            mx.append(x); my.append(float(np.mean(zs))); mv.append(vval[key])
        mx, my = np.array(mx), np.array(my)
        for val in ("valid", "invalid"):
            m = np.array([v == val for v in mv])
            ax.scatter(mx[m], my[m], c=VCOL[val], s=34, alpha=0.85, edgecolor="white",
                       linewidth=0.3, label=f"{val} arg (mean/arg)")
        ax.axhline(0, color="black", lw=0.6); ax.axvline(0, color="gray", ls=":", lw=0.8)
        ax.set_title(f"turn {k}:  z = d·(S{k} − S0)/σ₀", fontsize=10, fontweight="bold")
        ax.set_xlabel("argument BT (quality proxy)")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.25)
        print(f"turn {k}: n={len(mx)} args")
    axes[0].set_ylabel("z = d·(Sₖ − S0) / σ₀   (push direction)")

    # clip view to the bulk (M08, σ₀-floored, is off-scale)
    allz = np.concatenate([np.array([np.mean(z) for z in runz[k].values()]) for k in (1, 2, 3)])
    lo, hi = np.percentile(allz, [1, 99])
    pad = 0.15 * (hi - lo)
    axes[0].set_ylim(lo - pad, hi + pad)

    fig.suptitle("Normalised push-direction shift vs argument BT, by horizon  "
                 "[gpt-5.5, 22 artefacts, 132 args, 20 runs each]", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(HERE, "results", "delta_clouds_bt_22.png")
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
