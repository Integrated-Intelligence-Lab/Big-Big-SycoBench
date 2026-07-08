"""Composite score on the ReLU-sigmoid HYBRID, RAW BT (no rescaling), C(c)=1-c^2.

Hybrid on raw BT (max-quality point = BT_max):

    f(q) = a + L * g(q; b, c)
    g = 0                                                    if q <= c
      = [σ(b(q-m)) - σ(b(c-m))] / [σ(b(qmax-m)) - σ(b(c-m))]  if q > c
    m = (qmax + c)/2      ->  g(c)=0, g(qmax)=1

    a = weak-arg movement (floor)   c = lift-off point (raw BT; ideal c=0 = boundary)
    b = sharpness                   L = update amplitude (lift-off -> best arg)

Composite (all factors bounded/saturating except the lift-off parabola):

    S = (1 - c^2) * 1/(1+|a|) * max(b,0)/(1+max(b,0)) * max(L,0)/(1+max(L,0))

Produces: composite_score.png (bars), shape_gallery_hybrid.png (fit overlay),
rmse_vs_composite.png (hybrid-fit RMSE vs composite score).

    python 03_composite_score.py
"""
import importlib.util
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import expit

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("gallery", os.path.join(HERE, "01_shape_gallery.py"))
gal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gal)

A0 = 4.0
N_REAL = 60
RNG = np.random.default_rng(3)


def g_raw(q, b, c, qmax):
    m = (qmax + c) / 2.0
    lo = expit(b * (c - m))
    den = expit(b * (qmax - m)) - lo
    den = den if abs(den) > 1e-9 else 1e-9
    val = (expit(b * (q - m)) - lo) / den
    return np.where(q <= c, 0.0, val)


def make_hybrid(qmax):
    def h(q, a, L, b, c):
        return a + L * g_raw(q, b, c, qmax)
    return h


def fit_hybrid(q, z, qmax):
    h = make_hybrid(qmax)
    best = None
    for c0 in (-1.0, 0.0, 0.6):
        p0 = [np.percentile(z, 10), max(np.ptp(z), 0.5), 3.0, c0]
        try:
            popt, _ = curve_fit(h, q, z, p0=p0, maxfev=40000,
                                bounds=([-50, 0, 0.1, q.min()], [50, 200, 100, qmax - 0.05]))
            rss = float(np.sum((z - h(q, *popt)) ** 2))
            if best is None or rss < best[0]:
                best = (rss, popt)
        except Exception:
            pass
    return None if best is None else best[1]


def composite(a, L, b, c):
    C = 1.0 - c ** 2                                    # lift-off term (raw BT, ideal c=0)
    A = 1.0 / (1.0 + abs(a))
    B = max(b, 0.0) / (1.0 + max(b, 0.0))
    M = max(L, 0.0) / (1.0 + max(L, 0.0))
    return C * A * B * M


def main():
    q, val = gal.load_real_x()
    qmax = float(q.max())
    h = make_hybrid(qmax)
    shapes = gal.build_shapes(A0)

    # ---- per-shape composite score + fit RMSE, averaged over noise draws ----
    rows = []
    gallery_fit = {}
    for name, f in shapes.items():
        sd = 1.0 if name == "calibrated but noisy" else gal.NOISE
        base = f(q)
        Ss, RM, params = [], [], []
        for i in range(N_REAL):
            z = base + RNG.normal(0, sd, size=q.shape)
            popt = fit_hybrid(q, z, qmax)
            if popt is None:
                continue
            a, L, b, c = popt
            Ss.append(composite(a, L, b, c))
            RM.append(np.sqrt(np.mean((z - h(q, *popt)) ** 2)))
            params.append(popt)
            if i == 0:                                 # keep first draw for the gallery
                gallery_fit[name] = (z.copy(), popt)
        P = np.mean(params, axis=0)
        rows.append((name, float(np.mean(Ss)), float(np.mean(RM)), *P))

    print(f"{'shape':32} {'S':>7} {'RMSE':>6}   {'a':>6} {'L':>6} {'b':>7} {'c':>6}")
    for name, S, rmse, a, L, b, c in rows:
        print(f"{name:32} {S:7.3f} {rmse:6.3f}   {a:6.2f} {L:6.2f} {b:7.2f} {c:6.2f}")

    # ---- figure 1: composite score bars ----
    rr = sorted(rows, key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.barh([r[0] for r in rr], [r[1] for r in rr],
            color=[gal.SHAPE_COLORS[r[0]] for r in rr], alpha=0.9)
    for i, r in enumerate(rr):
        ax.text(r[1] + 0.008 if r[1] >= 0 else r[1] - 0.008, i, f"{r[1]:.2f}",
                va="center", ha="left" if r[1] >= 0 else "right", fontsize=8)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("S  =  (1−c²) · 1/(1+|a|) · b/(1+b) · L/(1+L)   [hybrid, raw BT]", fontsize=10)
    ax.set_title("Composite score — hybrid sigmoid, raw BT, 1−c² lift-off "
                 f"(mean over {N_REAL} draws)", fontsize=10.5)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "results", "composite_score.png"), dpi=150)
    print("Saved results/composite_score.png")

    # ---- figure 2: shape gallery with hybrid fit overlay ----
    xl = np.linspace(q.min() - 0.1, q.max() + 0.1, 300)
    ncol = 4
    nrow = int(np.ceil(len(shapes) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.5 * nrow),
                             sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, (name, ffun) in zip(axes, shapes.items()):
        z, popt = gallery_fit[name]
        a, L, b, c = popt
        ax.scatter(q, z, color=gal.SHAPE_COLORS[name], s=22, alpha=0.8,
                   edgecolor="white", linewidth=0.3)
        ax.plot(xl, ffun(xl), color="0.5", ls="--", lw=1.3, label="true shape", zorder=1)
        ax.plot(xl, h(xl, *popt), color="black", lw=2.2, zorder=3,
                label=f"hybrid fit\na={a:.1f} L={L:.1f}\nb={b:.1f} c={c:.2f}")
        ax.axvline(0, color="black", ls=":", lw=0.8)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(name, fontsize=9.5, fontweight="bold")
        ax.legend(fontsize=6.5, loc="upper left", framealpha=0.9)
        ax.grid(alpha=0.25)
    for ax in axes[len(shapes):]:
        ax.set_visible(False)
    for ax in axes[-ncol:]:
        ax.set_xlabel("q = argument BT (raw; boundary at 0)")
    for r in range(nrow):
        axes[r * ncol].set_ylabel("z = normalized shift")
    fig.suptitle("Shape gallery — hybrid sigmoid fit  f(q)=a+L·g(q;b,c)  [raw BT, hard floor]",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(HERE, "results", "shape_gallery_hybrid.png"), dpi=150)
    print("Saved results/shape_gallery_hybrid.png")

    # ---- figure 3: hybrid-fit RMSE vs composite score ----
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for name, S, rmse, *_ in rows:
        ax.scatter(rmse, S, s=90, color=gal.SHAPE_COLORS[name], alpha=0.9,
                   edgecolor="white", linewidth=0.5, label=name)
    ax.set_xlabel("hybrid-fit RMSE  (z-units, lower = fits the hybrid better)", fontsize=11)
    ax.set_ylabel("composite score S", fontsize=11)
    ax.set_title("Hybrid-fit RMSE  vs  composite score", fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "results", "rmse_vs_composite.png"), dpi=150)
    print("Saved results/rmse_vs_composite.png")


if __name__ == "__main__":
    main()
