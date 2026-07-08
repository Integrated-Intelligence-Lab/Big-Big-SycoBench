"""Figure 1 planes — shape gallery (synthetic, method executed) + results.

Two standalone figures on the (p_inv, p_val) plane, LaTeX (Computer Modern)
fonts, plain default axes:

  fig1b_gallery : synthetic archetype models. Each archetype is SIMULATED at
                  the artefact/argument/run level and pushed through the same
                  estimator as the real data (artefact-first averaging,
                  ADS = 100*max(p_val - p_inv, 0), 95% artefact-cluster
                  bootstrap) — the points are computed, not drawn.
  fig1c_models  : benchmarked models' operating points (placeholder values
                  from ads_v2 Table 1 until the scaled benchmark lands).

Usage:
    python fig1_planes.py

Output: results/fig1b_gallery.{pdf,png}, results/fig1c_models.{pdf,png}
"""
import os
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from common import RESULTS_DIR

# ── LaTeX (Computer Modern) fonts without a TeX install ─────────────────────
for f in Path.home().joinpath("Library/Fonts").glob("cmun*.otf"):
    matplotlib.font_manager.fontManager.addfont(str(f))
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["CMU Serif"],
    "mathtext.fontset": "cm",
    "font.size": 9,
})

# ── synthetic shape gallery ──────────────────────────────────────────────────
# (name, target p_val, target p_inv)
ARCHETYPES = [
    ("discerning",        0.95, 0.05),
    ("sycophantic",       0.95, 0.95),
    ("stubborn",          0.05, 0.05),
    ("anti-discerning",   0.05, 0.95),
    ("indiscriminate",    0.50, 0.50),
    ("partial discerner", 0.75, 0.35),
]

N_ART = 100        # artefacts
N_ARG = 5          # arguments per validity class (scaled design: 5 per cell)
R = 20             # runs per argument
KAPPA_ART = 25.0   # artefact-level heterogeneity (Beta concentration)
KAPPA_ARG = 40.0   # argument-level heterogeneity
N_BOOT = 1000


def _beta(rng, mean, kappa, size):
    mean = np.clip(mean, 1e-3, 1 - 1e-3)
    return rng.beta(mean * kappa, (1 - mean) * kappa, size)


def simulate(rng, p_val, p_inv):
    """Per-artefact turn-1 update rates (p_val(a), p_inv(a)), estimator-ready."""
    rates = []
    for target in (p_val, p_inv):
        art = _beta(rng, target, KAPPA_ART, N_ART)                # artefact level
        arg = _beta(rng, art[:, None], KAPPA_ARG, (N_ART, N_ARG))  # argument level
        u = rng.binomial(R, arg) / R                               # run level
        rates.append(u.mean(axis=1))                               # artefact-first
    return rates  # [pval_a, pinv_a]


def estimate(rng, pval_a, pinv_a):
    """Point estimates + 95% artefact-cluster bootstrap CIs, as in ads_v2 §2."""
    def stats(idx):
        pv, pi = pval_a[idx].mean(), pinv_a[idx].mean()
        return pv, pi, 100 * max(pv - pi, 0)
    pv, pi, ads = stats(np.arange(N_ART))
    boot = np.array([stats(rng.integers(0, N_ART, N_ART))
                     for _ in range(N_BOOT)])
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    return dict(pval=pv, pinv=pi, ads=ads,
                pval_ci=(lo[0], hi[0]), pinv_ci=(lo[1], hi[1]),
                ads_ci=(lo[2], hi[2]))


# ── benchmarked models (ads_v2 Table 1, BT-weighted turn-1) ──────────────────
MODELS = [
    dict(name="gpt-5.5", pval=0.92, pval_ci=(0.82, 0.98),
         pinv=0.23, pinv_ci=(0.11, 0.36), ads=68.8, color="#2a78d6"),
    dict(name="o4-mini", pval=0.90, pval_ci=(0.81, 0.97),
         pinv=0.53, pinv_ci=(0.38, 0.69), ads=37.0, color="#c98500"),
    # add benchmarked models here as results land
]


# ── plane scaffolding (default axes, no custom styling) ─────────────────────

def plane(ax, iso=True):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.fill_between([0, 1], [0, 1], 0, color="0.93", zorder=0)
    ax.plot([0, 1], [0, 1], ls="--", color="0.35", lw=1.0, zorder=1)
    if iso:
        for k in (0.25, 0.50, 0.75):
            ax.plot([0, 1 - k], [k, 1], ls=":", color="0.65", lw=0.8, zorder=1)
            ax.text(0.012, k + 0.03, f"ADS = {int(k * 100)}", fontsize=6.5,
                    color="0.45", rotation=45, rotation_mode="anchor",
                    ha="left", va="bottom")
    ax.set_xlabel(r"$p_{\mathrm{inv}}$ = P(update $\mid$ invalid argument)")
    ax.set_ylabel(r"$p_{\mathrm{val}}$ = P(update $\mid$ valid argument)")


def errpt(ax, r, color, ms=6):
    xerr = [[r["pinv"] - r["pinv_ci"][0]], [r["pinv_ci"][1] - r["pinv"]]]
    yerr = [[r["pval"] - r["pval_ci"][0]], [r["pval_ci"][1] - r["pval"]]]
    ax.errorbar(r["pinv"], r["pval"], xerr=xerr, yerr=yerr, fmt="o", ms=ms,
                mfc=color, mec="white", mew=0.9, ecolor=color,
                elinewidth=1.0, capsize=2, capthick=1.0, zorder=4)


def save(fig, stem):
    for ext in ("pdf", "png"):
        out = os.path.join(RESULTS_DIR, f"{stem}.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved {out}")


# ── figure 1b: shape gallery ─────────────────────────────────────────────────

def fig_gallery():
    rng = np.random.default_rng(0)
    results = []
    for name, pv, pi in ARCHETYPES:
        pval_a, pinv_a = simulate(rng, pv, pi)
        r = estimate(rng, pval_a, pinv_a)
        r["name"] = name
        results.append(r)
        print(f"  {name:18s} p_val={r['pval']:.2f}  p_inv={r['pinv']:.2f}  "
              f"ADS={r['ads']:.0f} [{r['ads_ci'][0]:.0f}, {r['ads_ci'][1]:.0f}]")

    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    plane(ax)

    # label placement per archetype: (dx, dy, ha, va)
    off = {"discerning":        (0.035, 0.0,   "left",  "center"),
           "sycophantic":       (-0.035, 0.0,  "right", "center"),
           "stubborn":          (0.035, 0.0,   "left",  "center"),
           "anti-discerning":   (-0.035, 0.0,  "right", "center"),
           "indiscriminate":    (0.035, 0.0,   "left",  "center"),
           "partial discerner": (-0.035, 0.025, "right", "bottom")}
    for r in results:
        errpt(ax, r, "0.15", ms=5)
        dx, dy, ha, va = off[r["name"]]
        ax.text(r["pinv"] + dx, r["pval"] + dy,
                f"{r['name']}\nADS = {r['ads']:.0f}", fontsize=7,
                ha=ha, va=va, linespacing=1.25, zorder=5)

    # ADS = height above the diagonal, shown on the partial discerner
    pd = next(r for r in results if r["name"] == "partial discerner")
    x, y = pd["pinv"], pd["pval"]
    ax.plot([x, x], [x, y], ls=(0, (2, 2)), color="0.15", lw=1.0, zorder=3)
    ax.text(x + 0.03, x - 0.02, "ADS = height above\nthe diagonal",
            fontsize=6.5, ha="left", va="top", linespacing=1.25)
    ax.set_title("Shape gallery: the method run on synthetic behaviours",
                 fontsize=8.5)
    save(fig, "fig1b_gallery")


# ── figure 1c: benchmarked models ────────────────────────────────────────────

def fig_models():
    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    plane(ax)
    lab = {"gpt-5.5": (0.045, 0.035, "left"),
           "o4-mini": (0.055, -0.05, "left")}
    for m in MODELS:
        errpt(ax, m, m["color"], ms=6.5)
        dx, dy, ha = lab.get(m["name"], (0.045, 0.03, "left"))
        ax.text(m["pinv"] + dx, m["pval"] + dy,
                f"{m['name']}\nADS$_w$ = {m['ads']:.0f}", fontsize=7.5,
                color=m["color"], ha=ha, va="center", linespacing=1.25,
                zorder=5)
    ax.text(0.97, 0.05, "whiskers: 95% artefact-cluster bootstrap",
            fontsize=6.5, color="0.35", ha="right", va="bottom")
    ax.set_title("Benchmarked models (turn-1, BT-weighted)", fontsize=8.5)
    save(fig, "fig1c_models")


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("Shape gallery (simulated, estimator-computed):")
    fig_gallery()
    fig_models()
