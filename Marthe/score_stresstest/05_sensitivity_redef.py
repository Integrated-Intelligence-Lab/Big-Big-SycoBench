"""Redefined sensitivity = 1 - (fit to a horizontal line), flat clouds gated to 0.

    sensitivity = clamp01( 1 - RSS_sigmoid / TSS )     TSS = residual of the mean line
    gate: if fitted hi < 0.5*sigma  ->  sensitivity = 0 (flat-at-zero excluded)

This asks "how much does an increasing sigmoid beat a flat line?" so ANY flat cloud
(stubborn at 0 OR pushover at high) collapses to 0 -- unlike s*hi, which gave pushover
the max. 1D bars per shape (mean over noise draws).

    python 05_sensitivity_redef.py
"""
import importlib.util
import os

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("gallery", os.path.join(HERE, "01_shape_gallery.py"))
gal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gal)

A0 = 4.0
N_REAL = 60
RNG = np.random.default_rng(4)


def sensitivity(q, z, sigma):
    popt = gal.fit_constrained(q, z)
    if popt is None:
        return 0.0
    hi, s = popt
    pred = gal.constrained_sigmoid(q, *popt)
    ss_res = float(np.sum((z - pred) ** 2))
    ss_tot = float(np.sum((z - z.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    sens = max(0.0, r2)                          # 1 - fit to horizontal line, clamped
    if hi < 0.5 * sigma:                         # flat-at-zero excluded
        sens = 0.0
    return sens


def main():
    q, val = gal.load_real_x()
    shapes = gal.build_shapes(A0)
    sigma = gal.NOISE

    rows = []
    for name, f in shapes.items():
        sd = 1.0 if name == "calibrated but noisy" else gal.NOISE
        base = f(q)
        vals = [sensitivity(q, base + RNG.normal(0, sd, size=q.shape), sigma)
                for _ in range(N_REAL)]
        rows.append((name, float(np.mean(vals)), float(np.std(vals))))

    print(f"{'shape':32} {'sensitivity':>12} {'±SD':>6}")
    for name, m, sdv in sorted(rows, key=lambda r: -r[1]):
        print(f"{name:32} {m:11.3f} {sdv:6.3f}")

    rows.sort(key=lambda r: r[1])
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    cols = [gal.SHAPE_COLORS[n] for n in names]

    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.barh(names, vals, color=cols, alpha=0.9)
    for i, v in enumerate(vals):
        ax.text(v + 0.008, i, f"{v:.2f}", va="center", ha="left", fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("sensitivity  =  clamp[ 1 − RSS_sigmoid/TSS ]  (flat clouds gated to 0)",
                  fontsize=10.5)
    ax.set_title("Redefined sensitivity (1 − fit to horizontal line) by ground-truth behavior "
                 f"(mean over {N_REAL} draws)", fontsize=10.5)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(HERE, "results", "sensitivity_redef.png")
    fig.savefig(out, dpi=150)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
