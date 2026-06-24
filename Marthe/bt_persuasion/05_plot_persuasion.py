"""Does BT-recovered persuasion strength rise with Arne's persuasion level?

The persuasion axis is `persuasion_load` (operator count). If the operators do
what they claim, BT_persuasion should increase with load -- a manipulation
check. Splitting by validity (GOOD/BAD) also shows whether substance still
matters once you control for rhetoric (e.g. does a high-load BAD argument
out-persuade a low-load GOOD one?).

BT is pool-centered, so per-pool Spearman(load, bt) is the clean within-pool
signal; we report the mean across pools plus the pooled view.

Outputs:
  results/persuasion_by_load.csv  -- mean BT per (validity, persuasion_load)
  results/bt_persuasion.png       -- BT vs persuasion_load, colored by validity
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
COLOR = {"GOOD": "tab:green", "BAD": "tab:red"}


def main():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "bt_scores.csv"))

    # per-pool Spearman(load, bt), averaged (pool-centered scale)
    per_pool = [spearmanr(g.persuasion_load, g.bt_rating).statistic
                for _, g in df.groupby("pool_id") if g.persuasion_load.nunique() > 1]
    per_pool = [r for r in per_pool if not np.isnan(r)]
    pooled = spearmanr(df.persuasion_load, df.bt_rating).statistic
    print(f"Spearman(persuasion_load, BT): mean per-pool {np.mean(per_pool):.2f} "
          f"(n={len(per_pool)} pools) | pooled {pooled:.2f}")
    for v in ("GOOD", "BAD"):
        s = df[df.validity == v]
        print(f"  within {v}: pooled {spearmanr(s.persuasion_load, s.bt_rating).statistic:.2f}")

    tab = df.groupby(["validity", "persuasion_load"]).bt_rating.mean().unstack("validity")
    tab.to_csv(os.path.join(RESULTS_DIR, "persuasion_by_load.csv"))
    print("mean BT by persuasion_load:")
    print(tab.round(2).to_string())

    fig, ax = plt.subplots(figsize=(8, 5.5))
    rng = np.random.default_rng(0)
    for v in ("GOOD", "BAD"):
        s = df[df.validity == v]
        x = s.persuasion_load + rng.uniform(-0.12, 0.12, len(s))
        ax.scatter(x, s.bt_rating, c=COLOR[v], s=16, alpha=0.5, edgecolor="none", label=v)
        m = s.groupby("persuasion_load").bt_rating.mean()
        ax.plot(m.index, m.values, c=COLOR[v], lw=2, marker="o", label=f"{v} mean")
    ax.axhline(0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("persuasion_load (rhetoric operators added)")
    ax.set_ylabel("BT persuasion rating (pool-centered)")
    ax.set_title("Does measured persuasiveness scale with Arne's persuasion operators?\n"
                 f"mean per-pool $\\rho$(load, BT) = {np.mean(per_pool):.2f}  |  "
                 f"5 artefacts x 2 directions")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "bt_persuasion.png")
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
