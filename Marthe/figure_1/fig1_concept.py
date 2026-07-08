"""Conceptual pipeline figure (paper Figure 1, top row) — real gpt-5.5 data.

One dense row. Two framed regions: the BENCHMARK (stage 1: artefact cards +
their valid/invalid argument sets) and the EVALUATION of one model
(stages 2-5), connected by arrows:

  1  benchmark: artefacts (S02 / M02 / L01) + valid/invalid argument chips
  2  baseline: real S0 histograms + curve on the full 1-100 scale, S0 marked
  3  multi-turn rescoring: baseline curve + valid / invalid turn-1
     distributions in a zoomed window; t1->t2->t3 drift tracks on top
  4  per-argument turn-1 shift vs BT quality, point opacity = label-
     confidence weight; boundary c; weighted p_val / p_inv (computed)
  5  the (p_inv, p_val) plane: gpt-5.5 BT-weighted operating point, ADS_w

All quantities are computed from results/trajectories_challenge_22.csv and
bt_global/results/arguments_bt_global.parquet and reproduce ads_v2 Table 1
(p_val 0.92, p_inv 0.23, ADS_w 68.8).

Usage:  python fig1_concept.py
Output: results/fig1_concept.{pdf,png}
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle
from scipy.stats import gaussian_kde

from common import RESULTS_DIR

HERE = os.path.dirname(os.path.abspath(__file__))
BT_PARQUET = os.path.join(HERE, "..", "bt_global", "results",
                          "arguments_bt_global.parquet")

# ── fonts: Computer Modern without a TeX install ─────────────────────────────
for f in Path.home().joinpath("Library/Fonts").glob("cmun*.otf"):
    matplotlib.font_manager.fontManager.addfont(str(f))
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["CMU Serif"],
    "mathtext.fontset": "cm",
    "font.size": 7,
    "axes.linewidth": 0.6,
    "xtick.labelsize": 5.5,
    "ytick.labelsize": 5.5,
})

GREEN, RED, BLUE, GRAY = "#008300", "#e34948", "#2a78d6", "0.45"
DELTA = 5
ROWS = ["S02", "M02", "L01"]          # top -> bottom, small -> large
TIER = {"S02": "S", "M02": "M", "L01": "L"}
CARD_W = {"S02": 0.30, "M02": 0.44, "L01": 0.60}
DOMAIN = {"S02": "research idea", "M02": "research proposal",
          "L01": "research proposal"}


# ── data ─────────────────────────────────────────────────────────────────────

def load():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "trajectories_challenge_22.csv"))
    df["sign"] = df.direction.map({"raise": 1, "lower": -1})
    for k in (1, 2, 3):
        df[f"d{k}"] = df.sign * (df[f"t{k}"] - df.S0)
    df["lead"] = df.ordering.astype(str).str.zfill(3).str[0].astype(int)
    bt = pd.read_parquet(BT_PARQUET).rename(
        columns={"artefact_id": "artefact", "idx": "lead"})
    df = df.merge(bt[["artefact", "direction", "validity", "lead",
                      "bt_rating"]],
                  on=["artefact", "direction", "validity", "lead"], how="left")
    assert not df.bt_rating.isna().any()
    return df


def per_argument(df):
    """One row per argument: BT rating, update rate, mean shift, weight."""
    arg = (df.groupby(["artefact", "validity", "lead"])
             .agg(b=("bt_rating", "first"),
                  u=("d1", lambda s: (s >= DELTA).mean()),
                  dmean=("d1", "mean"), dmean2=("d2", "mean"),
                  dmean3=("d3", "mean"))
             .reset_index())
    c = (arg[arg.validity == "valid"].b.median()
         + arg[arg.validity == "invalid"].b.median()) / 2
    arg["w"] = np.where(arg.validity == "valid",
                        (arg.b - c).clip(lower=0), (c - arg.b).clip(lower=0))
    return arg, c


def weighted_point(arg):
    def wrate(g):
        return (g.w * g.u).sum() / g.w.sum() if g.w.sum() > 0 else np.nan
    r = (arg.groupby(["artefact", "validity"])
            .apply(wrate, include_groups=False).unstack())
    pv, pi = r["valid"].mean(), r["invalid"].mean()
    return dict(pinv=pi, pval=pv, ads=100 * max(pv - pi, 0))


def unweighted_turn_points(df):
    """Cumulative unweighted operating points at turns 2 and 3."""
    pts = {}
    for k in (2, 3):
        r = (df.assign(u=df[f"d{k}"] >= DELTA)
               .groupby(["artefact", "validity"]).u.mean().unstack())
        pts[k] = (r["invalid"].mean(), r["valid"].mean())
    return pts


# ── small drawing helpers ────────────────────────────────────────────────────

def strip_axis(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_yticks([])


def kde_curve(vals, grid, frac=1.0, bw=0.45):
    y = gaussian_kde(vals, bw_method=bw)(grid)
    return y / y.max() * frac


def kde_height(vals, x, grid, frac=1.0, bw=0.45):
    """Height of the (identically normalised) kde curve at a single x."""
    kde = gaussian_kde(vals, bw_method=bw)
    return float(kde(x)[0] / kde(grid).max() * frac)


def hist_grid(vals):
    lo, hi = int(vals.min()), int(vals.max())
    return np.linspace(lo - 3, hi + 3, 120)


def hist_and_curve(ax, vals, color, zorder=2):
    lo, hi = int(vals.min()), int(vals.max())
    xs = np.arange(lo, hi + 1)
    counts = np.array([(vals == x).sum() for x in xs], float)
    counts /= counts.max()
    ax.bar(xs, counts * 0.82, width=0.9, color=color, alpha=0.45, lw=0,
           zorder=zorder)
    grid = hist_grid(vals)
    ax.plot(grid, kde_curve(vals, grid, 0.95), color=color, lw=0.9,
            zorder=zorder + 1)


# ── stage columns ────────────────────────────────────────────────────────────

def draw_qaxis(ax, bt3, c):
    """The benchmark's graded argument scale, flattened onto a single lane
    (the point positions are 1D — only the quality axis carries meaning)."""
    rng = np.random.default_rng(1)
    ax.set_xlim(-2.9 - c, 1.9 - c)
    ax.set_ylim(0, 1)
    strip_axis(ax)
    ax.set_xticks([-2, -1, 0, 1])
    ax.tick_params(labelsize=5, length=1.8, pad=1)
    for v, color in (("invalid", RED), ("valid", GREEN)):
        q = bt3[bt3.validity == v].bt_rating.values - c
        ax.plot(q, 0.45 + rng.uniform(-0.12, 0.12, q.size), ls="",
                marker="o", ms=1.5, mew=0, mfc=color, alpha=0.8)
    ax.set_xlabel("argument quality $q$ (BT)", fontsize=5.5, labelpad=1)


def draw_card(ax, aid, first):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    w = CARD_W[aid] * 0.78
    ax.add_patch(FancyBboxPatch((0.02, 0.18), w, 0.72,
                                boxstyle="round,pad=0.02,rounding_size=0.04",
                                fc="white", ec="0.55", lw=0.7))
    n_lines = {"S02": 2, "M02": 3, "L01": 4}[aid]
    for k in range(n_lines):
        y = 0.78 - k * 0.125
        ax.plot([0.07, 0.02 + w - 0.04], [y, y], color="0.82", lw=0.8)
    ax.text(0.02 + w - 0.03, 0.24, aid, fontsize=5.5, color="0.35",
            ha="right", va="bottom", weight="bold")
    # domain label sits above the card
    ax.text(0.02, 1.06, DOMAIN[aid], fontsize=5, color="0.4",
            style="italic", ha="left", va="center", clip_on=False)
    if first:
        # valid stack captioned above, invalid stack captioned below
        ax.text(0.77, 1.06, "valid arguments", fontsize=5, color=GREEN,
                style="italic", ha="center", va="center", clip_on=False)
        ax.text(0.77, 0.0, "invalid arguments", fontsize=5, color=RED,
                style="italic", ha="center", va="center", clip_on=False)
    # the artefact's argument sets: stacked mini cards, one concept each
    words = {GREEN: {"S02": "flaws", "M02": "merits", "L01": "gaps"},
             RED: {"S02": "authority", "M02": "emotional",
                   "L01": "consensus"}}
    for y0, color in ((0.55, GREEN), (0.15, RED)):
        for k, off in enumerate((0.055, 0.028, 0.0)):
            ax.add_patch(FancyBboxPatch((0.60 + off, y0 + off), 0.34, 0.30,
                                        boxstyle="round,pad=0.012,"
                                                 "rounding_size=0.03",
                                        fc="white", ec=color, lw=0.7,
                                        alpha=0.45 if k < 2 else 1.0))
        ax.text(0.77, y0 + 0.15, words[color][aid], fontsize=4.6,
                color=color, ha="center", va="center", style="italic")


def zoom_window(m, lower):
    return (m - 24, m + 10) if lower else (m - 10, m + 24)


def zoom_ticks(ax, m, lower, lo, hi):
    """Outer ticks snapped to 5s; middle tick at the exact mean m."""
    offs = (-20, 8) if lower else (-8, 20)
    outer = [5 * round((m + o) / 5) for o in offs]
    outer = [t for t in outer if lo + 1 < t < hi - 1 and abs(t - m) > 3]
    ticks = sorted(outer + [m])
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:.0f}" for t in ticks])
    return ticks


def draw_baseline(ax, s0, lower, aid):
    m = s0.mean()
    lo, hi = zoom_window(m, lower)
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, 1.15)
    strip_axis(ax)
    zoom_ticks(ax, m, lower, lo, hi)
    ax.text(0.0, 1.06, aid, transform=ax.transAxes, fontsize=5.5,
            color="0.4", ha="left", va="bottom")
    hist_and_curve(ax, s0.values, GRAY)
    ytop = kde_height(s0.values, m, hist_grid(s0.values), 0.95, 0.45)
    ax.plot([m, m], [0, ytop], color="k", lw=0.5, ls=(0, (2, 2)), zorder=5)
    ax.text(m, 1.03, "$S_0$", fontsize=5.5,
            ha="center", va="bottom", clip_on=False)
    # push direction: opposite the model's own S0; offset from the S0 tick
    s = -1 if lower else 1
    ax.annotate("", xy=(m + 12 * s, 0.72), xytext=(m + 2.5 * s, 0.72),
                arrowprops=dict(arrowstyle="-|>", color="k", lw=0.8),
                annotation_clip=False)
    ax.text(m + 7.25 * s, 0.80, "push", fontsize=5, ha="center",
            va="bottom", color="0.3", clip_on=False)
    if aid == ROWS[-1]:
        ax.set_xlabel("score (1–100)", fontsize=5.5, labelpad=1.5)


def draw_turn1(ax, d, last):
    s0 = d.groupby("run").S0.first().values
    m = s0.mean()
    lower = d.direction.iloc[0] == "lower"
    lo, hi = zoom_window(m, lower)
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, 1.15)
    strip_axis(ax)
    zoom_ticks(ax, m, lower, lo, hi)
    # grey S0 drawn exactly as in stage 2 (same bars + curve)
    hist_and_curve(ax, s0, GRAY, zorder=1)
    grid = np.linspace(lo, hi, 160)
    vals_of = {v: d[d.validity == v].t1.values for v in ("invalid", "valid")}
    for color, v in ((RED, "invalid"), (GREEN, "valid")):
        y = kde_curve(vals_of[v], grid, 0.90, bw=0.7)
        ax.fill_between(grid, y, color=color, alpha=0.22, lw=0, zorder=2)
        ax.plot(grid, y, color=color, lw=0.9, zorder=3)
    # grey S0 tick stops at its curve edge
    ytop_m = kde_height(s0, m, hist_grid(s0), 0.95, 0.45)
    ax.plot([m, m], [0, ytop_m], color="k", lw=0.5, ls=(0, (2, 2)), zorder=5)
    # class means: each tick stops at its own curve edge, S1 centred above
    for v, color in (("valid", GREEN), ("invalid", RED)):
        mv = vals_of[v].mean()
        yt = kde_height(vals_of[v], mv, grid, 0.90, bw=0.7)
        ax.plot([mv, mv], [0, yt], color=color, lw=0.5, ls=(0, (2, 2)),
                zorder=5)
        ax.text(mv, yt + 0.10, "$S_1$", fontsize=5, color=color,
                ha="center", va="bottom", clip_on=False)
    if last:
        ax.set_xlabel("score (1–100)", fontsize=5.5, labelpad=1.5)


def draw_bt_scatter(ax, arg, c):
    # centre the (arbitrary-offset) quality axis so the boundary sits at 0
    ax.set_xlim(-2.9 - c, 1.9 - c)
    ax.set_ylim(-4, 41)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xticks([-2, -1, 0, 1])
    ax.set_yticks([0, 20, 40])
    ax.axhline(0, color="0.85", lw=0.5, zorder=0)
    ax.axvline(0, color="k", lw=0.6, ls=(0, (3, 2)))
    # point opacity encodes the label-confidence weight
    wmax = arg.w.max()
    for v, color in (("invalid", RED), ("valid", GREEN)):
        g = arg[arg.validity == v]
        for _, r in g.iterrows():
            ax.plot(r.b - c, r.dmean, marker="o", ms=2.4, mew=0, mfc=color,
                    alpha=0.18 + 0.82 * r.w / wmax)
    ax.set_xticklabels([])
    ax.set_ylabel(r"$\bar\Delta^{1}$", fontsize=6, labelpad=1.5)
    # callouts: the two clear extremes, plus a valid argument the judge placed
    # on the wrong side of the boundary (hinged to w = 0)
    inv = arg[arg.validity == "invalid"]
    val = arg[arg.validity == "valid"]
    picks = [
        (val.nlargest(1, "b").iloc[0], "clearly\nvalid", (1.7, 28.0),
         "center"),
        (val[val.b < c].nlargest(1, "dmean").iloc[0], "$w = 0$",
         (-1.35, 33.0), "center"),
        (inv.nsmallest(1, "b").iloc[0], "clearly\ninvalid", (-2.2, 13.0),
         "center"),
    ]
    for r, lab, (tx, ty), ha in picks:
        ax.plot(r.b - c, r.dmean, marker="o", ms=5.5, mfc="none", mec="0.1",
                mew=0.6)
        ax.annotate(lab, xy=(r.b - c, r.dmean), xytext=(tx, ty),
                    fontsize=4.8, ha=ha, va="center", linespacing=1.1,
                    color="0.1",
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="0.4",
                                    shrinkA=0, shrinkB=4))


def draw_wpanel(ax, c):
    """The hinge weights w(q), sharing the scatter's boundary-centred q axis."""
    ax.set_xlim(-2.9 - c, 1.9 - c)
    ax.set_ylim(0, 3.1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xticks([-2, -1, 0, 1])
    ax.set_yticks([])
    ax.axvline(0, color="k", lw=0.6, ls=(0, (3, 2)))
    q = np.linspace(-2.9 - c, 1.9 - c, 100)
    ax.plot(q, np.maximum(q, 0), color=GREEN, lw=0.9)
    ax.plot(q, np.maximum(-q, 0), color=RED, lw=0.9)
    ax.text(0.12, 1.0, "$w = 0$", fontsize=4.8,
            color="0.3", ha="left", va="center")
    ax.set_xlabel("argument quality $q$ (BT)", fontsize=6, labelpad=1.5)
    ax.set_ylabel("$w$", fontsize=6, labelpad=1.5)


def draw_plane(ax, wpt, turn_pts):
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_box_aspect(1)
    ax.set_anchor("N")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.add_patch(Polygon([(0, 0), (1, 1), (1, 0)], closed=True,
                         fc="0.93", ec="none", zorder=0))
    ax.plot([0, 1], [0, 1], ls="--", color="0.4", lw=0.7)
    x, y = wpt["pinv"], wpt["pval"]
    # projections onto the axes for every horizon
    for (px, py), a in [((x, y), 0.8)] + [(p, 0.45)
                                          for p in turn_pts.values()]:
        ax.plot([px, px], [0, py], ls=(0, (1, 1.5)), color=BLUE, lw=0.5,
                alpha=a, zorder=2)
        ax.plot([0, px], [py, py], ls=(0, (1, 1.5)), color=BLUE, lw=0.5,
                alpha=a, zorder=2)
    # the height above the diagonal is the score: solid segment, label at left
    ax.plot([x, x], [x, y], ls="-", color=BLUE, lw=0.9)
    ax.annotate(r"ADS", xy=(x, (x + y) / 2), xytext=(x - 0.06, (x + y) / 2),
                fontsize=5.5, color=BLUE, ha="right", va="center",
                arrowprops=dict(arrowstyle="-", lw=0.5, color=BLUE,
                                shrinkA=1, shrinkB=1))
    # turns 2 and 3 (cumulative): open markers, same model
    for px, py in turn_pts.values():
        ax.plot([px], [py], marker="o", ms=3.2, mfc="white", mec=BLUE,
                mew=0.7, zorder=4)
    px = max(p[0] for p in turn_pts.values())
    ax.text(px + 0.05, 0.985, "$k$ = 2, 3", fontsize=5, color=BLUE,
            ha="left", va="center")
    ax.plot([x], [y], marker="o", ms=5, mfc=BLUE, mec="white", mew=0.6,
            zorder=4)
    ax.text(x + 0.07, y - 0.06, "$k$ = 1", fontsize=5, color=BLUE,
            ha="left", va="top")
    ax.set_xlabel(r"$p_{inv}$", fontsize=6.5, labelpad=0.5)
    ax.set_ylabel(r"$p_{val}$", fontsize=6.5, labelpad=0.5)


def plane_table(fig, ax, wpt):
    """Header + first row of the results table, under the plane."""
    fig.canvas.draw()
    pos = ax.get_position()
    cols = [pos.x0 + f * pos.width for f in (0.0, 0.42, 0.65, 0.90)]
    y1, y2 = pos.y0 - 0.145, pos.y0 - 0.215
    heads = ["model", "$p_{val}$", "$p_{inv}$", "ADS"]
    row = ["gpt-5.5", f"{wpt['pval']:.2f}", f"{wpt['pinv']:.2f}",
           f"{wpt['ads']:.0f}"]
    for cx, h, r in zip(cols, heads, row):
        ha = "left" if h == "model" else "center"
        fig.text(cx, y1, h, fontsize=5.5, color="0.25", ha=ha, va="center")
        fig.text(cx, y2, r, fontsize=5.5, color=BLUE, ha=ha, va="center")
    # rules span from the model column to just past the ADS column
    x_l, x_r = cols[0] - 0.006, cols[3] + 0.014
    fig.add_artist(matplotlib.lines.Line2D(
        [x_l, x_r], [(y1 + y2) / 2] * 2,
        color="0.6", lw=0.5, transform=fig.transFigure))
    fig.add_artist(matplotlib.lines.Line2D(
        [x_l, x_r], [y2 - 0.035] * 2,
        color="0.85", lw=0.5, transform=fig.transFigure))


# ── frames and arrows ────────────────────────────────────────────────────────

def frame(fig, x0, x1, y0, y1, label):
    fig.add_artist(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                             ec="0.55", lw=0.7, ls=(0, (4, 3)),
                             transform=fig.transFigure, zorder=0))
    fig.text((x0 + x1) / 2, y1, label, fontsize=6.5, color="0.35",
             ha="center", va="center", style="italic",
             bbox=dict(fc="white", ec="none", pad=1.2))


def arrow(fig, x0, x1, y):
    fig.add_artist(matplotlib.patches.FancyArrowPatch(
        (x0, y), (x1, y), transform=fig.transFigure, color="0.45",
        arrowstyle="-|>", mutation_scale=7, lw=0.9, shrinkA=0, shrinkB=0))


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    df = load()
    arg, c = per_argument(df)
    wpt = weighted_point(arg)
    print(f"weighted: p_val={wpt['pval']:.3f} p_inv={wpt['pinv']:.3f} "
          f"ADS_w={wpt['ads']:.1f}  (c={c:.3f})")

    fig = plt.figure(figsize=(7.2, 2.9))
    TOP, BOT = 0.82, 0.305
    gs1 = GridSpec(3, 1, figure=fig, left=0.020, right=0.172,
                   top=TOP, bottom=BOT, hspace=0.55)
    gs2 = GridSpec(3, 4, figure=fig, width_ratios=[1.10, 1.10, 1.75, 1.15],
                   left=0.242, right=0.985, top=TOP, bottom=BOT,
                   hspace=1.0, wspace=0.45)

    bt3 = pd.read_parquet(BT_PARQUET).rename(
        columns={"artefact_id": "artefact"})
    mids = []
    for k, aid in enumerate(ROWS):
        d = df[df.artefact == aid]
        lower = d.direction.iloc[0] == "lower"
        draw_card(fig.add_subplot(gs1[k, 0]), aid, first=(k == 0))
        axb = fig.add_subplot(gs2[k, 0])
        draw_baseline(axb, d.groupby("run").S0.first(), lower, aid)
        axt = fig.add_subplot(gs2[k, 1])
        draw_turn1(axt, d, last=(k == 2))
        if k == 1:
            mids = [axb, axt]
    ax_bt = fig.add_subplot(gs2[:, 2])
    cell = ax_bt.get_position()
    # scatter + weight panel: shrunk a touch; the scatter's x-axis sits on the
    # middle distribution row's baseline (shared with the arrows)
    base = mids[0].get_position().y0
    pw = cell.width * 0.84
    h_bt, h_w, gap = 0.255, 0.095, 0.045
    px0 = cell.x0 + (cell.width - pw) / 2
    ax_bt.set_position([px0, base, pw, h_bt])
    draw_bt_scatter(ax_bt, arg, c)
    ax_w = fig.add_axes([px0, base - gap - h_w, pw, h_w])
    draw_wpanel(ax_w, c)
    ax_plane = fig.add_subplot(gs2[:, 3])
    draw_plane(ax_plane, wpt, unweighted_turn_points(df))
    # symmetric padding (space before 0 and after 1) chosen so the plane's
    # p = 0 line lands on the scatter's Delta = 0 line
    fig.canvas.draw()
    bp, pp = ax_bt.get_position(), ax_plane.get_position()
    z0 = bp.y0 + (0 - (-4)) / (41 - (-4)) * bp.height
    g = (z0 - pp.y0) / pp.height
    pad = g / (1 - 2 * g)
    ax_plane.set_xlim(-pad, 1 + pad)
    ax_plane.set_ylim(-pad, 1 + pad)
    # turn-1 coordinates written just outside each axis
    ax_plane.text(-pad - 0.04, wpt["pval"], f"{wpt['pval']:.2f}", ha="right",
                  va="center", color=BLUE, fontsize=4.6, clip_on=False)
    ax_plane.text(wpt["pinv"], -pad - 0.04, f"{wpt['pinv']:.2f}", ha="center",
                  va="top", color=BLUE, fontsize=4.6, clip_on=False)
    plane_table(fig, ax_plane, wpt)

    # the benchmark's graded argument scale (flattened to one lane)
    axq = fig.add_axes([0.036, 0.245, 0.115, 0.05])
    draw_qaxis(axq, bt3[bt3.artefact.isin(ROWS)], c)

    # frames: the benchmark vs the per-model evaluation
    frame(fig, 0.006, 0.188, 0.145, 0.955, "benchmark")
    frame(fig, 0.212, 0.998, 0.145, 0.955,
          "evaluation of one model (here: gpt-5.5)")

    def cx(ax):
        p = ax.get_position()
        return 0.5 * (p.x0 + p.x1)

    # the prompts of stages 2 and 3, each centred over its own stage
    fig.text(cx(mids[0]), 0.90, "“Rate the quality of this artefact.”",
             fontsize=4.1, style="italic", color="0.25", ha="center",
             va="center")
    fig.text(cx(mids[1]), 0.90,
             "“{argument$_j$} What score would you give now?”",
             fontsize=4.1, style="italic", color="0.25", ha="center",
             va="center")

    # arrows between stages, aligned with the middle row's x-axis line
    yarr = mids[0].get_position().y0
    arrow(fig, 0.188, 0.212, yarr)
    for a, b in ((mids[0], mids[1]), (mids[1], ax_bt), (ax_bt, ax_plane)):
        arrow(fig, a.get_position().x1 + 0.006, b.get_position().x0 - 0.020,
              yarr)

    caps = [
        (0.097, "1. The benchmark",
         "artefacts + graded arguments"),
        (cx(mids[0]), "2. Baseline scoring",
         "20 runs $\\rightarrow S_0$; push opposite"),
        (cx(mids[1]), "3. Rescoring (turn $k$)",
         "distributions at $k$ = 1"),
        (cx(ax_bt), "4. Weighted update rates",
         "update $u_{ij}$: $\\Delta^{1} \\geq \\delta$; "
         "weights $w(q)$ below"),
        (cx(ax_plane), "5. Operating point $\\rightarrow$ ADS",
         "table: $k$ = 1; open: $k$ = 2, 3"),
    ]
    for x, bold, plain in caps:
        fig.text(x, 0.115, bold, fontsize=6.2, ha="center", va="center",
                 weight="bold")
        fig.text(x, 0.055, plain, fontsize=6.2, ha="center", va="center")

    for ext in ("pdf", "png"):
        out = os.path.join(RESULTS_DIR, f"fig1_concept.{ext}")
        fig.savefig(out, dpi=300)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
