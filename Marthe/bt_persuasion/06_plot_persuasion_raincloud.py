"""Raincloud of BT persuasion ratings, in the bt_validation house style, but
showing the two competing groupings side by side:

  left  : grouped by VALIDITY (GOOD/BAD)        -> separates strongly
  right : grouped by PERSUASION level (L0-L4)    -> overlaps (no separation)

The contrast is the finding: asked "which argument is more persuasive", the
judge ranks by substance, not by the rhetoric operators Arne layered on.
BT is pool-centered so all 700 items share one axis.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, spearmanr

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
VAL_COLOR = {"GOOD": "tab:green", "BAD": "tab:red"}
LVL_CMAP = plt.cm.viridis


def auc(pos, neg):
    p = np.asarray(pos)[:, None]
    n = np.asarray(neg)[None, :]
    return ((p > n).sum() + 0.5 * (p == n).sum()) / (p.size * n.size)


def raincloud(ax, df, groups, colors, labels):
    xs = np.linspace(df.bt_rating.min() - 0.5, df.bt_rating.max() + 0.5, 400)
    kde_max = max(gaussian_kde(df[m].bt_rating)(xs).max() for _, m in groups)
    strip_h = 0.05 * kde_max
    rng = np.random.default_rng(0)
    for i, (key, mask) in enumerate(groups):
        vals = df[mask].bt_rating.values
        kde = gaussian_kde(vals)
        ys = kde(xs)
        ax.plot(xs, ys, color=colors[i], lw=1.4, zorder=3)
        ax.fill_between(xs, 0, ys, color=colors[i], alpha=0.35, zorder=2,
                        label=f"{labels[i]} (n={len(vals)})")
        m = vals.mean()
        ax.vlines(m, 0, np.interp(m, xs, ys), color=colors[i], ls=(0, (2, 1)), lw=0.9, zorder=4)
        ybase = -0.05 * kde_max - i * strip_h * 1.4
        ax.scatter(vals, ybase + rng.uniform(0, strip_h, len(vals)),
                   color=colors[i], s=5, alpha=0.5, edgecolor="none", zorder=2)
    ax.axhline(0, color="black", lw=0.6)
    ax.axvline(0, color="gray", ls=":", lw=0.8)
    ax.set_yticks([])
    ax.set_xlabel("BT persuasion rating (pool-centered)")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=7, loc="upper left", frameon=False)


def main():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "bt_scores.csv"))
    a = auc(df[df.validity == "GOOD"].bt_rating, df[df.validity == "BAD"].bt_rating)
    rho = spearmanr(df.persuasion_load, df.bt_rating).statistic

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    raincloud(axes[0], df,
              [("GOOD", df.validity == "GOOD"), ("BAD", df.validity == "BAD")],
              [VAL_COLOR["GOOD"], VAL_COLOR["BAD"]], ["GOOD", "BAD"])
    axes[0].set_title(f"by validity  ->  pooled AUC(GOOD>BAD) = {a:.2f}")

    loads = sorted(df.persuasion_load.unique())
    raincloud(axes[1], df,
              [(l, df.persuasion_load == l) for l in loads],
              [LVL_CMAP(i / (len(loads) - 1)) for i in range(len(loads))],
              [f"L{l}" for l in loads])
    axes[1].set_title(f"by persuasion level  ->  $\\rho$(load, BT) = {rho:.2f}")

    fig.suptitle("BT 'persuasion' ratings (Arne's args): the judge ranks by substance, "
                 "not by the persuasion operators", fontsize=12)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "bt_persuasion_raincloud.png")
    fig.savefig(out, dpi=150)
    print(f"Saved {out}  | AUC(GOOD>BAD)={a:.3f}  rho(load,BT)={rho:.3f}")


if __name__ == "__main__":
    main()
