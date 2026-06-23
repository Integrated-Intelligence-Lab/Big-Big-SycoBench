"""Validation diagnostics: do the BT ratings actually separate the valid and
invalid labels, and which arguments (if any) are crossover outliers?

Outputs:
  results/separation_summary.csv  -- per-pool clean-separation flag + per-pool AUC
  results/outliers.csv            -- invalid items rated above their pool's
                                      weakest valid item (or vice versa)
  results/bt_separation.png       -- pooled BT-rating distribution (valid vs
                                      invalid) with a raincloud strip of every
                                      item below it

A pool is "cleanly separated" when all of its valid items outrank all of its
invalid items (min(valid) > max(invalid)) -- the BT analogue of "no overlap"
from the original human-audit conversation. AUC = P(a random valid item
outranks a random invalid item from the same pool), 1.0 = perfect, 0.5 =
chance; it's computed directly from the bt_rating ranks rather than via
scikit-learn, since that's not otherwise a dependency of this repo.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
COLOR = {"valid": "tab:green", "invalid": "tab:red"}


def pool_auc(valid_ratings, invalid_ratings):
    """P(valid > invalid) over all valid x invalid pairs in a pool, ties = 0.5."""
    v = np.asarray(valid_ratings)[:, None]
    iv = np.asarray(invalid_ratings)[None, :]
    wins = (v > iv).sum() + 0.5 * (v == iv).sum()
    return wins / (v.size * iv.size)


def main():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "bt_scores.csv"))

    summary_rows = []
    outlier_rows = []
    for pool_id, g in df.groupby("pool_id"):
        valid = g[g.validity == "valid"]
        invalid = g[g.validity == "invalid"]
        clean = valid.bt_rating.min() > invalid.bt_rating.max()
        auc = pool_auc(valid.bt_rating.values, invalid.bt_rating.values)
        summary_rows.append({"pool_id": pool_id, "clean_separation": clean, "auc": auc})

        crossover_invalid = invalid[invalid.bt_rating > valid.bt_rating.min()]
        crossover_valid = valid[valid.bt_rating < invalid.bt_rating.max()]
        for _, row in pd.concat([crossover_invalid, crossover_valid]).iterrows():
            outlier_rows.append(row.to_dict())

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(RESULTS_DIR, "separation_summary.csv"), index=False)
    outliers = pd.DataFrame(outlier_rows)
    outliers.to_csv(os.path.join(RESULTS_DIR, "outliers.csv"), index=False)

    overall_auc = pool_auc(
        df[df.validity == "valid"].bt_rating.values,
        df[df.validity == "invalid"].bt_rating.values,
    )
    print(f"{len(summary)} pools | {summary.clean_separation.mean():.0%} cleanly separated | "
          f"mean per-pool AUC {summary.auc.mean():.3f} | pooled AUC {overall_auc:.3f}")
    if len(outliers):
        print(f"{len(outliers)} crossover items flagged in results/outliers.csv:")
        for _, r in outliers.iterrows():
            print(f"  [{r.validity}] {r.pool_id} idx={r.idx} bt={r.bt_rating:.2f}  {r.text[:90]}")

    # Distribution + scatter, in the style of the difficulty-ranking project's
    # Figure 4: an overlaid filled KDE per label group, a dashed line at each
    # group mean with a +/-1 std shaded band, and every individual item as a
    # jittered point in a strip just below the x-axis. BT ratings are
    # pool-centered, so all 264 items share one scale.
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(6.5, 9 / 5 * 1.6), dpi=300)
    xs = np.linspace(df.bt_rating.min() - 0.5, df.bt_rating.max() + 0.5, 400)
    kde_max = 0.0
    for validity in ("invalid", "valid"):
        vals = df[df.validity == validity].bt_rating.values
        kde = gaussian_kde(vals)
        ys = kde(xs)
        kde_max = max(kde_max, ys.max())
        ax.plot(xs, ys, color=COLOR[validity], linewidth=1, zorder=3)
        ax.fill_between(xs, 0, ys, facecolor=COLOR[validity], alpha=0.5,
                        zorder=2, label=f"{validity} (n={len(vals)})")
        # mean line + ±1 std band under the curve
        mean, std = vals.mean(), vals.std()
        height = np.interp(mean, xs, ys)
        ax.vlines(mean, 0, height, color=COLOR[validity],
                  linestyle=(0, (2, 1)), linewidth=0.8, zorder=4)
        mask = (xs >= mean - std) & (xs <= mean + std)
        xb = np.concatenate([[mean - std], xs[mask], [mean + std]])
        yb = np.concatenate([[np.interp(mean - std, xs, ys)], ys[mask],
                             [np.interp(mean + std, xs, ys)]])
        ax.fill_between(xb, 0, yb, facecolor=COLOR[validity], alpha=0.4, zorder=2)

    # points in a strip just below 0, jittered both horizontally and vertically
    y_base, y_jit = -0.012 * kde_max, 0.006 * kde_max
    x_jit = 0.06 * (df.bt_rating.max() - df.bt_rating.min())
    for validity in ("invalid", "valid"):
        sub = df[df.validity == validity]
        x = sub.bt_rating + rng.uniform(-x_jit, x_jit, len(sub))
        y = y_base + rng.uniform(-y_jit, y_jit, len(sub))
        ax.scatter(x, y, s=8, alpha=0.9, color=COLOR[validity],
                   edgecolor="none", zorder=3)

    ax.set_yticks([])
    ax.set_ylabel("")
    ax.set_xlabel("BT rating (log-strength, pool-centered)", fontsize=7)
    ax.tick_params(axis="x", labelsize=7)
    ax.set_title("Argument-validity BT ratings, valid vs invalid\n"
                 f"pooled AUC {overall_auc:.3f}, "
                 f"{summary.clean_separation.mean():.0%} of pools cleanly separated",
                 fontsize=8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.legend(fontsize=7, loc="upper center", frameon=False)
    fig.tight_layout()

    out = os.path.join(RESULTS_DIR, "bt_separation.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
