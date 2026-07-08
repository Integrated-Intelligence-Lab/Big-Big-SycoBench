"""Shape gallery: real 132 BT values on x, synthetic normalized shifts z in a
library of behaviors on y. Overlays the *constrained* sigmoid the benchmark uses
(floor = 0, take-off pinned at BT = 0; free hi, s). No R^2 / scoring yet — this is
just to eyeball how each behavior's cloud looks and how the pinned sigmoid tracks it.

    z = hi * expit(s*q - 2)     # lo=0, q0=2/s  =>  take-off = q0 - 2/s = 0

    python 01_shape_gallery.py
"""
import csv
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import expit

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CSV = os.path.join(ROOT, "figure_1", "results", "trajectories_challenge_22.csv")
PARQUET = os.path.join(ROOT, "bt_global", "results", "arguments_bt_global.parquet")
VCOL = {"valid": "tab:green", "invalid": "tab:red"}
RNG = np.random.default_rng(0)
NOISE = 0.20          # realistic per-arg mean noise (real: SD~0.9 / sqrt(20 runs))


def load_real_x():
    """The 132 argument BT ratings + validity used in figure_1 (all included)."""
    bt = {(r.artefact_id, r.direction, r.validity, int(r.idx)): float(r.bt_rating)
          for r in pd.read_parquet(PARQUET).itertuples()}
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    seen = {}
    for r in rows:
        aid, val, d = r["artefact"], r["validity"], r["direction"]
        fi = int(r["ordering"][0]); key = (aid, val, fi)
        seen[key] = (bt[(aid, d, val, fi)], val)
    q = np.array([v[0] for v in seen.values()])
    val = np.array([v[1] for v in seen.values()])
    order = np.argsort(q)
    return q[order], val[order]


def logistic(q, lo, A, s, c):
    return lo + A * expit(s * (q - c))


def constrained_sigmoid(q, hi, s):
    """The benchmark's family: lo=0, take-off pinned at q=0 (q0=2/s)."""
    return hi * expit(s * q - 2.0)


def fit_constrained(q, z):
    p0 = [max(np.percentile(z, 90), 1.0), 2.0]
    try:
        popt, _ = curve_fit(constrained_sigmoid, q, z, p0=p0, maxfev=20000,
                            bounds=([0.0, 0.05], [10 * (abs(z).max() + 1), 60]))
        return popt
    except Exception:
        return None


def build_shapes(A0):
    """z = f(q) behaviors, in z-units, take-offs/floors expressed on the BT scale."""
    def toff(t, s):  # inflection c that puts the tangent take-off at BT=t
        return t + 2.0 / s
    return {
        "calibrated (ideal)":        lambda q: logistic(q, 0.0,     A0, 3, toff(0.0, 3)),
        "sycophant: floor>0":        lambda q: logistic(q, 0.4*A0,  A0, 3, toff(0.0, 3)),
        "sycophant: early take-off": lambda q: logistic(q, 0.0,     A0, 3, toff(-1.0, 3)),
        "skeptic: late take-off":    lambda q: logistic(q, 0.0,     A0, 3, toff(0.7, 3)),
        "stubborn (z~0)":            lambda q: np.zeros_like(q),
        "pushover (flat-high)":      lambda q: np.full_like(q, A0),
        "super-sensitive":           lambda q: logistic(q, 0.0,   2*A0, 3, toff(0.0, 3)),
        "linear":                    lambda q: 0.3*A0 + 0.6*A0 * q,
        # true sycophant: moves for invalid args exactly as for valid ones -> the
        # calibrated curve mirrored across BT=0 (response tracks |q|, not validity).
        "true sycophant (valid=invalid)": lambda q: logistic(np.abs(q), 0.0, A0, 3, toff(0.0, 3)),
        "contrarian (decreasing)":   lambda q: logistic(q, 0.0,     A0, -3, -0.5),
        "bump (mid-quality)":        lambda q: A0 * np.exp(-(q - 0.0)**2 / (2*0.4**2)),
        "calibrated but noisy":      lambda q: logistic(q, 0.0,     A0, 3, toff(0.0, 3)),
    }


# one distinct color per shape, shared with 02_score_scatter.py
_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
            "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#393b79", "#e7ba52"]
SHAPE_COLORS = {name: _PALETTE[i % len(_PALETTE)]
                for i, name in enumerate(build_shapes(1.0).keys())}


def main():
    q, val = load_real_x()

    # anchor amplitude A0 to a plausible real scale: use ~4 z-units (a few sigma move)
    A0 = 4.0
    shapes = build_shapes(A0)

    ncol = 4
    nrow = int(np.ceil(len(shapes) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2*ncol, 3.5*nrow),
                             sharex=True, sharey=True)
    axes = axes.ravel()
    xl = np.linspace(q.min() - 0.1, q.max() + 0.1, 300)

    for ax, (name, f) in zip(axes, shapes.items()):
        sd = 1.0 if name == "calibrated but noisy" else NOISE
        z = f(q) + RNG.normal(0, sd, size=q.shape)
        ax.scatter(q, z, color=SHAPE_COLORS[name], s=22, alpha=0.8,
                   edgecolor="white", linewidth=0.3, label="data")
        ax.plot(xl, f(xl), color="0.5", ls="--", lw=1.3, label="true shape", zorder=1)
        popt = fit_constrained(q, z)
        if popt is not None:
            hi_f, s_f = popt
            s_disp = 0.0 if hi_f < 0.05 else s_f          # slope is meaningless when hi=0
            ax.plot(xl, constrained_sigmoid(xl, *popt), color="black", lw=2.2,
                    label=f"pinned fit\nhi={hi_f:.1f}, s={s_disp:.1f}", zorder=3)
        ax.axvline(0, color="black", ls=":", lw=0.8)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.legend(fontsize=6.5, loc="upper left", framealpha=0.9)
        ax.grid(alpha=0.25)

    for ax in axes[len(shapes):]:
        ax.set_visible(False)
    for ax in axes[-ncol:]:
        ax.set_xlabel("q = argument BT (unscaled; take-off pinned at 0)")
    for r in range(nrow):
        axes[r*ncol].set_ylabel("z = normalized shift")

    fig.suptitle("Shape gallery — real 132 BT values (x), synthetic shifts (y), "
                 "pinned sigmoid overlay  [floor=0, take-off@BT=0]", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(HERE, "results", "shape_gallery.png")
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
