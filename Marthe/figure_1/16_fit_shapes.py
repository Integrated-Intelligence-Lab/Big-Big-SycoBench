"""Fit linear / ReLU-hinge / sigmoid to the gpt-5.5 quality cloud, per horizon.

z = d*(Sk-S0)/sigma0  vs  q = min-max rescaled BT,  per-argument means, M08 excluded.

  linear :  z = a + b*q                              (2 params)  -> the colleague's a,b
  ReLU   :  z = a + b*max(q - q0, 0)                 (3 params)  -> a=floor(sycophancy),
                                                                   q0=activation threshold,
                                                                   b=slope above threshold
  sigmoid:  z = lo + (hi-lo)/(1+exp(-s*(q-q0)))      (4 params)

Reports AIC for each (lower = better); ReLU is the headline. AIC = n*ln(RSS/n)+2p.

    python 16_fit_shapes.py
"""
import argparse
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
CSV = os.path.join(HERE, "results", "trajectories_challenge_22.csv")
PARQUET = os.path.join(ROOT, "bt_global", "results", "arguments_bt_global.parquet")
SIGN = {"lower": -1, "raise": +1}
VCOL = {"valid": "tab:green", "invalid": "tab:red"}


def aic(rss, n, p):
    return n * np.log(rss / n) + 2 * p


def fit_linear(q, z):
    b, a = np.polyfit(q, z, 1)
    pred = a + b * q
    return dict(name="linear", p=2, rss=float(((z - pred) ** 2).sum()),
                f=lambda x: a + b * x, txt=f"a={a:.1f}, b={b:.1f}")


def fit_relu(q, z):
    best = None
    for q0 in np.linspace(q.min() + 0.02, q.max() - 0.02, 100):
        X = np.column_stack([np.ones_like(q), np.maximum(q - q0, 0.0)])
        coef, *_ = np.linalg.lstsq(X, z, rcond=None)
        rss = float(((z - X @ coef) ** 2).sum())
        if best is None or rss < best[0]:
            best = (rss, coef[0], coef[1], q0)
    rss, a, b, q0 = best
    return dict(name="ReLU", p=3, rss=rss, q0=q0, a=a, b=b,
                f=lambda x: a + b * np.maximum(x - q0, 0.0),
                txt=f"a={a:.1f}, q₀={q0:.2f}, b={b:.1f}")


def fit_sigmoid(q, z):
    def g(x, lo, hi, s, q0):
        return lo + (hi - lo) * expit(s * (x - q0))
    p0 = [0.0, np.percentile(z, 90), 8.0, 0.4]
    try:
        popt, _ = curve_fit(g, q, z, p0=p0, maxfev=20000,
                            bounds=([-10, 0, 0.1, 0], [20, 200, 60, 1]))
        lo, hi, s, q0 = popt
        pred = g(q, *popt)
        return dict(name="sigmoid", p=4, rss=float(((z - pred) ** 2).sum()),
                    f=lambda x: g(x, *popt), lo=lo, hi=hi, s=s, q0=q0,
                    txt=f"lo={lo:.1f} hi={hi:.1f}\ns={s:.0f} q₀={q0:.2f}")
    except Exception:
        return dict(name="sigmoid", p=4, rss=np.nan, f=None, txt="fit failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="gpt55")
    a = ap.parse_args()
    suffix = "" if a.tag == "gpt55" else f"_{a.tag}"
    csv_path = os.path.join(HERE, "results", f"trajectories_challenge_22{suffix}.csv")

    bt = {(r.artefact_id, r.direction, r.validity, int(r.idx)): float(r.bt_rating)
          for r in pd.read_parquet(PARQUET).itertuples()}
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    s0run = defaultdict(dict)
    for r in rows:
        s0run[r["artefact"]][int(r["run"])] = int(r["S0"])
    sig = {a: np.std(list(v.values()), ddof=1) for a, v in s0run.items()}
    floored = {aid for aid, s in sig.items() if s == 0}          # σ₀=0 artefacts: excluded

    runz = {1: defaultdict(list), 2: defaultdict(list), 3: defaultdict(list)}
    meta = {}
    for r in rows:
        aid, val, d = r["artefact"], r["validity"], r["direction"]
        if aid in floored:
            continue
        s = [int(r["S0"]), int(r["t1"]), int(r["t2"]), int(r["t3"])]
        fi = int(r["ordering"][0]); key = (aid, val, fi)
        meta[key] = (bt[(aid, d, val, fi)], val)
        for k in (1, 2, 3):
            runz[k][key].append(SIGN[d] * (s[k] - s[0]) / sig[aid])
    btv = np.array([m[0] for m in meta.values()]); bmin, bmax = btv.min(), btv.max()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), sharex=True, sharey=True)
    print(f"{'horizon':8} {'α=z(0)':>7} {'z(1)':>6} {'earned':>7} {'Φ':>5} {'B(λ1)':>6}   "
          f"ReLU(a,q₀,b) | sigmoid(lo,hi,s,q₀)")
    handles = None
    for k, ax in zip((1, 2, 3), axes):
        Q, Z, V = [], [], []
        for key, zs in runz[k].items():
            Q.append((meta[key][0] - bmin) / (bmax - bmin)); Z.append(float(np.mean(zs)))
            V.append(meta[key][1])
        Q, Z, V = np.array(Q), np.array(Z), np.array(V)
        relu, sig = fit_relu(Q, Z), fit_sigmoid(Q, Z)

        def score_of(f):
            z0, z1 = float(f(0.0)), float(f(1.0))
            alpha = max(z0, 0.0); earned = z1 - z0
            phi = alpha / z1 if z1 > 0 else np.nan
            return z0, z1, earned, phi, earned - alpha           # z0, z1, earned, Phi, B
        rz0, rz1, rE, rPhi, rB = score_of(relu["f"])
        sz0, sz1, sE, sPhi, sB = (score_of(sig["f"]) if sig["f"] is not None
                                  else (np.nan,) * 5)
        print(f"turn {k}: ReLU z0={rz0:.2f} z1={rz1:.1f} Φ={rPhi:.2f} B={rB:.1f} "
              f"(q₀={relu['q0']:.2f},β={relu['b']:.1f}) | "
              f"sigm z0={sz0:.2f} z1={sz1:.1f} Φ={sPhi:.2f} B={sB:.1f} "
              f"(s={sig.get('s', float('nan')):.0f},q₀={sig.get('q0', float('nan')):.2f})")

        for val in ("valid", "invalid"):
            m = V == val
            ax.scatter(Q[m], Z[m], c=VCOL[val], s=30, alpha=0.7, edgecolor="white",
                       linewidth=0.3, label=f"{val} arg")
        xl = np.linspace(0, 1, 200)
        ax.plot(xl, relu["f"](xl), color="black", ls="-", lw=2.4, label="ReLU fit")
        if sig["f"] is not None:
            ax.plot(xl, sig["f"](xl), color="tab:blue", ls=":", lw=2.0, label="sigmoid fit")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(f"turn {k}", fontsize=11, fontweight="bold")
        ax.set_xlabel("q = rescaled BT (0=worst, 1=best)")
        box = (f"{'':9}{'ReLU':>6} {'sigm':>6}\n"
               f"{'α=z(0)':9}{rz0:6.1f} {sz0:6.1f}\n"
               f"{'z(1)':9}{rz1:6.1f} {sz1:6.1f}\n"
               f"{'earned':9}{rE:6.1f} {sE:6.1f}\n"
               f"{'Φ':9}{rPhi:6.2f} {sPhi:6.2f}\n"
               f"{'B (λ=1)':9}{rB:6.1f} {sB:6.1f}\n"
               f"shape: ReLU q₀={relu['q0']:.2f}, β={relu['b']:.1f}\n"
               f"       sigm s={sig.get('s', float('nan')):.0f}, q₀={sig.get('q0', float('nan')):.2f}")
        ax.text(0.03, 0.975, box, transform=ax.transAxes, va="top", fontsize=7.3,
                family="monospace",
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.93))
        ax.grid(alpha=0.25)
        if handles is None:
            handles = ax.get_legend_handles_labels()
    axes[0].set_ylabel("z = d·(Sₖ − S0) / σ₀")
    allz = np.concatenate([[np.mean(z) for z in runz[k].values()] for k in (1, 2, 3)])
    lo, hi = np.percentile(allz, [0, 99]); axes[0].set_ylim(lo - 2, hi + 5)

    fig.legend(*handles, loc="lower center", ncol=4, fontsize=9, frameon=False)
    excl = ",".join(sorted(floored)) or "none"
    fig.suptitle(f"ReLU & sigmoid fits to the {a.tag} quality cloud, per horizon   "
                 f"Φ = max(α,0)/z(1),  B = (z(1)−α) − |α|   [σ₀=0 excluded: {excl}]", fontsize=12)
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    out = os.path.join(HERE, "results", f"shape_fits_22{suffix}.png")
    fig.savefig(out, dpi=150)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
