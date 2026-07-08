"""Paper Figure 1 — concept + methodology + model operating points.

Three panels:
  (a) full-width methodology strip: artefacts -> baseline scoring ->
      endogenous challenge direction -> valid/invalid arguments ->
      turn-1 update event defining p_val / p_inv
  (b) annotated concept plane: quadrant map with archetype ghost points,
      indiscriminate diagonal, ADS iso-lines, ADS-as-height demo,
      update-propensity arrow
  (c) the same plane with the benchmarked models' BT-weighted turn-1
      operating points and 95% cluster-bootstrap whiskers

Placeholder data: ads_v2 Table 1 (gpt-5.5, o4-mini). Add models to MODELS
as the scaled benchmark lands.

Usage:
    python fig1_paper.py

Output: results/fig1_paper.pdf + .png
"""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, Polygon

from common import RESULTS_DIR

# ── data (ads_v2 Table 1, BT-weighted turn-1) ────────────────────────────────

MODELS = [
    dict(name="gpt-5.5", pval=0.92, pval_ci=(0.82, 0.98),
         pinv=0.23, pinv_ci=(0.11, 0.36), ads=68.8, color="#2a78d6"),
    dict(name="o4-mini", pval=0.90, pval_ci=(0.81, 0.97),
         pinv=0.53, pinv_ci=(0.38, 0.69), ads=37.0, color="#c98500"),
    # add benchmarked models here as results land
]

# ── style tokens ─────────────────────────────────────────────────────────────

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SHADE = "#f0efec"
VALID = "#008300"
INVALID = "#e34948"
GHOST = "#a5a49e"

plt.rcParams.update({
    "font.size": 7.5,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.labelcolor": INK,
    "text.color": INK,
})


# ── panel (a): methodology strip ─────────────────────────────────────────────

def _card(ax, x, y, w, h, edge, face="white", lw=0.9):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.15,rounding_size=0.35",
        fc=face, ec=edge, lw=lw, mutation_aspect=1 / ax._strip_aspect))


def _mini_scale(ax, x0, x1, y, marks=True):
    """Horizontal 0-100 scale; returns score -> x mapper."""
    ax.plot([x0, x1], [y, y], color=AXIS, lw=0.9, solid_capstyle="butt")
    if marks:
        for s, lab in ((0, "0"), (100, "100")):
            xs = x0 + (x1 - x0) * s / 100
            ax.plot([xs, xs], [y - 0.22, y + 0.22], color=AXIS, lw=0.9)
            ax.text(xs, y - 0.55, lab, fontsize=5.5, color=MUTED,
                    ha="center", va="top")
    return lambda s: x0 + (x1 - x0) * s / 100


def _stage_caption(ax, xc, num, title, sub):
    ax.text(xc, 1.55, f"{num} {title}", fontsize=7, fontweight="bold",
            color=INK, ha="center", va="top")
    ax.text(xc, 0.65, sub, fontsize=6, color=INK2, ha="center",
            va="top", linespacing=1.35)


def _arrow(ax, x0, x1, y=6.0):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.0,
                                shrinkA=0, shrinkB=0))


def draw_strip(ax):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 10)
    ax.axis("off")
    # aspect of data units, used to keep rounded corners round
    fw, fh = ax.figure.get_size_inches()
    bb = ax.get_position()
    ax._strip_aspect = (100 / (fw * bb.width)) / (10 / (fh * bb.height))

    # ── 1: artefacts ──
    for i in range(3):
        _card(ax, 2.2 + i * 1.5, 3.9 + i * 1.1, 9.5, 3.4, AXIS)
    ax.text(13.9, 9.2, "S ×600\nM ×300\nL ×200", fontsize=5.8,
            color=INK2, ha="right", va="top", linespacing=1.55)
    _stage_caption(ax, 8.5, "①", "artefacts",
                   "600 S / 300 M / 200 L\n(~100 / 500 / 2000 words)\n"
                   "ground truth 1–100, ~uniform")
    _arrow(ax, 15.5, 18.5)

    # ── 2: baseline scoring ──
    to_x = _mini_scale(ax, 20.5, 33.5, 5.6)
    rng = np.random.default_rng(0)
    s0 = 78
    xs = to_x(np.clip(rng.normal(s0, 5, 14), 0, 100))
    ax.plot(xs, 6.25 + rng.uniform(-0.28, 0.28, xs.size), ls="",
            marker="o", ms=2.2, mfc=INK2, mec="none", alpha=0.55)
    ax.plot([to_x(s0)], [5.6], marker="v", ms=4.5, color=INK, mec="none")
    ax.text(to_x(s0), 7.3, "$S_0$ = 78", fontsize=6.5, color=INK,
            ha="center", va="bottom")
    _stage_caption(ax, 27, "②", "baseline scoring",
                   "subject model scores the\nartefact in independent\n"
                   "runs → initial score $S_0$")
    _arrow(ax, 35, 38)

    # ── 3: challenge direction ──
    to_x = _mini_scale(ax, 40, 53, 5.6)
    ax.plot([to_x(s0)], [5.6], marker="v", ms=4.5, color=INK, mec="none")
    ax.annotate("", xy=(to_x(44), 5.6), xytext=(to_x(74), 5.6),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))
    ax.text(to_x(59), 7.3, "push opposite $S_0$", fontsize=6.5,
            color=INK, ha="center", va="bottom")
    _stage_caption(ax, 46.5, "③", "challenge direction",
                   "raise & lower sets are\ngenerated; the set opposing\n"
                   "the model's own $S_0$ is used")
    _arrow(ax, 54.5, 57.5)

    # ── 4: arguments ──
    _card(ax, 59.5, 6.1, 16, 2.5, VALID)
    ax.text(60.4, 7.35, "valid: identifies a real flaw", fontsize=6,
            color=VALID, ha="left", va="center")
    _card(ax, 59.5, 3.0, 16, 2.5, INVALID)
    ax.text(60.4, 4.25, "invalid: “experts agree …”\n"
            "(authority fallacy)", fontsize=6, color=INVALID,
            ha="left", va="center", linespacing=1.3)
    _stage_caption(ax, 67.5, "④", "challenge arguments",
                   "20 per artefact:\n{valid, invalid} × {raise, lower}\n"
                   "shown one at a time")
    _arrow(ax, 77, 80)

    # ── 5: update event ──
    ax.text(90.0, 8.9, "update = shift ≥ δ\ntoward the request",
            fontsize=6.5, color=INK, ha="center", va="center",
            linespacing=1.3)
    _card(ax, 82.0, 5.4, 16, 1.9, VALID)
    ax.text(90.0, 6.35, "$p_{val}$ = P(update | valid)", fontsize=6.0,
            color=VALID, ha="center", va="center")
    _card(ax, 82.0, 2.8, 16, 1.9, INVALID)
    ax.text(90.0, 3.75, "$p_{inv}$ = P(update | invalid)", fontsize=6.0,
            color=INVALID, ha="center", va="center")
    _stage_caption(ax, 90.0, "⑤", "turn-1 update event",
                   "one argument per observation\n→ the axes of (b) and (c)")


# ── panels (b)/(c): the discernment plane ────────────────────────────────────

def draw_plane_base(ax, iso_alpha=1.0):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.add_patch(Polygon([(0, 0), (1, 1), (1, 0)], closed=True,
                         fc=SHADE, ec="none", zorder=0))
    ax.plot([0, 1], [0, 1], ls=(0, (5, 3)), color=MUTED, lw=1.0, zorder=1)
    for k in (0.25, 0.50, 0.75):
        ax.plot([0, 1 - k], [k, 1], ls=(0, (1, 2)), color=MUTED,
                lw=0.8, alpha=0.85 * iso_alpha, zorder=1)
        ax.text(0.012, k + 0.032, f"ADS = {int(k * 100)}", fontsize=5.8,
                color=MUTED, alpha=iso_alpha, rotation=45,
                rotation_mode="anchor", ha="left", va="bottom")
    ax.set_xlabel("$p_{inv}$ = P(update | invalid argument)", fontsize=8)
    ax.set_ylabel("$p_{val}$ = P(update | valid argument)", fontsize=8)


def draw_concept(ax):
    draw_plane_base(ax)

    # archetype ghost points + quadrant labels
    corners = [(0, 1, "Discerning", "left", "top", (0.035, -0.035)),
               (1, 1, "Sycophantic", "right", "top", (-0.035, -0.035)),
               (0, 0, "Stubborn", "left", "bottom", (0.035, 0.035)),
               (1, 0, "Anti-discerning", "right", "bottom", (-0.035, 0.035))]
    for x, y, lab, ha, va, (dx, dy) in corners:
        ax.plot([x], [y], marker="o", ms=6, mfc=GHOST, mec="white",
                mew=1.0, clip_on=False, zorder=4)
        ax.text(x + dx, y + dy, lab, fontsize=7.5, fontweight="bold",
                color=INK2, ha=ha, va=va, zorder=4)
    ax.plot([0.5], [0.5], marker="o", ms=6, mfc=GHOST, mec="white",
            mew=1.0, zorder=4)
    ax.text(0.665, 0.685, "indiscriminate (ADS = 0)", fontsize=5.8,
            color=MUTED, rotation=45, rotation_mode="anchor",
            ha="center", va="bottom")

    # ADS = height above the diagonal (demo point)
    px, py = 0.15, 0.72
    ax.plot([px], [py], marker="o", ms=6.5, mfc=INK2, mec="white",
            mew=1.0, zorder=4)
    ax.plot([px, px], [px, py], ls=(0, (2, 2)), color=INK2, lw=1.1, zorder=3)
    ax.annotate("ADS = height above\nthe diagonal",
                xy=(px + 0.005, (px + py) / 2), xytext=(0.225, 0.435),
                fontsize=6.2, color=INK2, ha="left", va="center",
                linespacing=1.3,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7,
                                shrinkA=2, shrinkB=3))

    # update propensity along the diagonal (deliberately not scored)
    off = 0.08
    ax.annotate("",
                xy=(0.95, 0.95 - off), xytext=(0.38, 0.38 - off),
                arrowprops=dict(arrowstyle="<|-|>", color=MUTED, lw=1.0))
    ax.text(0.665 + 0.04, 0.665 - off - 0.04,
            "update propensity (not scored)", fontsize=5.8, color=MUTED,
            rotation=45, rotation_mode="anchor", ha="center", va="center")

    # clipped region
    ax.text(0.80, 0.24, "score clips to 0", fontsize=5.8, color=MUTED,
            ha="center", va="center", rotation=45, rotation_mode="anchor")


def draw_models(ax):
    draw_plane_base(ax, iso_alpha=0.6)
    label_off = {"gpt-5.5": (0.05, 0.035, "left"),
                 "o4-mini": (0.06, -0.055, "left")}
    for m in MODELS:
        xerr = [[m["pinv"] - m["pinv_ci"][0]], [m["pinv_ci"][1] - m["pinv"]]]
        yerr = [[m["pval"] - m["pval_ci"][0]], [m["pval_ci"][1] - m["pval"]]]
        ax.errorbar(m["pinv"], m["pval"], xerr=xerr, yerr=yerr,
                    fmt="o", ms=7, mfc=m["color"], mec="white", mew=1.2,
                    ecolor=m["color"], elinewidth=1.1, capsize=2.2,
                    capthick=1.1, zorder=4)
        dx, dy, ha = label_off.get(m["name"], (0.05, 0.03, "left"))
        ax.text(m["pinv"] + dx, m["pval"] + dy,
                f"{m['name']}\nADS$_w$ = {m['ads']:.0f}",
                fontsize=6.8, color=m["color"], fontweight="bold",
                ha=ha, va="center", linespacing=1.3, zorder=5)
    ax.text(0.97, 0.06, "whiskers: 95% artefact-cluster bootstrap",
            fontsize=5.8, color=MUTED, ha="right", va="bottom")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    fig = plt.figure(figsize=(7.2, 6.1))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[0.72, 2.0],
                  left=0.075, right=0.985, top=0.965, bottom=0.075,
                  hspace=0.28, wspace=0.30)

    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    draw_strip(ax_a)
    draw_concept(ax_b)
    draw_models(ax_c)

    for ax, letter in ((ax_a, "a"), (ax_b, "b"), (ax_c, "c")):
        x = -0.01 if letter == "a" else -0.16
        ax.text(x, 1.06, letter, transform=ax.transAxes, fontsize=11,
                fontweight="bold", ha="left", va="top")
    ax_b.set_title("The discernment plane", fontsize=8.5, pad=8)
    ax_c.set_title("Benchmarked models (turn-1, BT-weighted)",
                   fontsize=8.5, pad=8)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        out = os.path.join(RESULTS_DIR, f"fig1_paper.{ext}")
        fig.savefig(out, dpi=300)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
