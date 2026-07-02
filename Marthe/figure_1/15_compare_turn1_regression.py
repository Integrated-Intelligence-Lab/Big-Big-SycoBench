"""Turn-1 quality regression z = alpha + beta*q, gpt-5.5 vs o4-mini, side by side.

Single-shot only (turn 1), built straight from each model's singleshot S0/args:
  z = d * (S1 - S0) / sigma0      sigma0 = std of that model's 20 baseline scores
  q = (BT - BT_min)/(BT_max - BT_min)   shared bt_global scale (model-agnostic judge)
One point per argument = mean over 20 runs. Artefacts with sigma0=0 are floored to
the model's min nonzero sigma0 and shown as rings, excluded from the fit.

    python 15_compare_turn1_regression.py
"""
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PARQUET = os.path.join(ROOT, "bt_global", "results", "arguments_bt_global.parquet")
TAGS = ["gpt55", "o4mini"]
SIGN = {"lower": -1, "raise": +1}
VCOL = {"valid": "tab:green", "invalid": "tab:red"}


def txt(b):
    return "".join(c.get("text", "") for it in b.get("output", [])
                   if it.get("type") == "message"
                   for c in it.get("content", []) if c.get("type") == "output_text")


def score(t):
    m = re.search(r"-?\d+", t)
    return int(m.group()) if m else None


def disc(tag, kind):
    folder = os.path.join(ROOT, "results", "singleshot", tag)
    for p in sorted(glob.glob(os.path.join(folder, "*output*.jsonl"))):
        cid = json.loads(open(p).readline())["custom_id"]
        if kind == "s0" and "|" not in cid and "_run" in cid:
            return p
        if kind == "args" and "|" in cid:
            return p


def load_tag(tag):
    s0 = defaultdict(dict)
    for l in open(disc(tag, "s0"), encoding="utf-8"):
        r = json.loads(l); aid, run = r["custom_id"].split("_run")
        s0[aid][int(run)] = score(txt(r["response"]["body"]))
    sig0 = {a: np.std([v for v in d.values() if v is not None], ddof=1) for a, d in s0.items()}
    floored = {a for a, s in sig0.items() if s == 0}
    floor = min(s for s in sig0.values() if s > 0)
    sig = {a: (s if s > 0 else floor) for a, s in sig0.items()}

    perarg = defaultdict(list)   # (aid,dir,val,idx) -> [z over runs]
    for l in open(disc(tag, "args"), encoding="utf-8"):
        r = json.loads(l); aid, d, val, idx, run = r["custom_id"].split("|")
        s1 = score(txt(r["response"]["body"])); s0v = s0[aid].get(int(run[1:]))
        if s1 is None or s0v is None:
            continue
        perarg[(aid, d, val, int(idx[3:]))].append(SIGN[d] * (s1 - s0v) / sig[aid])
    z = {k: float(np.mean(v)) for k, v in perarg.items()}
    return z, floored


def main():
    bt = {(r.artefact_id, r.direction, r.validity, int(r.idx)): float(r.bt_rating)
          for r in pd.read_parquet(PARQUET).itertuples()}
    data = {tag: load_tag(tag) for tag in TAGS}

    # shared q scale over the challenge-direction args actually used
    keys = set().union(*[set(z) for z, _ in data.values()])
    btv = np.array([bt[k] for k in keys]); bmin, bmax = btv.min(), btv.max()
    def q_of(k): return (bt[k] - bmin) / (bmax - bmin)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), sharex=True, sharey=True)
    print(f"{'model':8} {'beta':>6} {'alpha':>6} {'Phi':>5}")
    for ax, tag in zip(axes, TAGS):
        z, floored = data[tag]
        Q, Z, V, fl = [], [], [], []
        for k, zz in z.items():
            Q.append(q_of(k)); Z.append(zz); V.append(k[2]); fl.append(k[0] in floored)
        Q, Z, V, fl = np.array(Q), np.array(Z), np.array(V), np.array(fl)
        keep = ~fl
        beta, alpha = np.polyfit(Q[keep], Z[keep], 1)
        phi = abs(alpha) / (abs(alpha) + max(beta, 0))
        print(f"{tag:8} {beta:6.2f} {alpha:6.2f} {phi:5.2f}")
        for val in ("valid", "invalid"):
            m = (V == val) & keep
            ax.scatter(Q[m], Z[m], c=VCOL[val], s=34, alpha=0.8, edgecolor="white",
                       linewidth=0.3, label=f"{val} arg")
        if fl.any():
            ax.scatter(Q[fl], Z[fl], facecolors="none", edgecolors="gray", s=40,
                       label=f"σ₀ floored ({','.join(sorted(floored))})")
        xl = np.linspace(0, 1, 50)
        ax.plot(xl, alpha + beta * xl, "k--", lw=1.5)
        ax.axhline(0, color="black", lw=0.6); ax.axvline(0, color="gray", ls=":", lw=0.8)
        ax.set_title(f"{tag}   turn 1", fontsize=11, fontweight="bold")
        ax.set_xlabel("q = rescaled argument BT (0 = worst, 1 = best)")
        ax.text(0.04, 0.97, f"β={beta:.1f}\nα={alpha:.1f}\nΦ={phi:.2f}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
        ax.legend(fontsize=7.5, loc="lower right")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("z = d·(S1 − S0) / σ₀")

    allz = np.concatenate([[v for k, v in z.items() if k[0] not in fl0]
                           for tag, (z, fl0) in data.items()])
    lo, hi = np.percentile(allz, [1, 99]); pad = 0.15 * (hi - lo)
    axes[0].set_ylim(lo - pad, hi + pad)

    fig.suptitle("Turn-1 quality regression  z = α + β·q   gpt-5.5 vs o4-mini   "
                 "Φ = |α|/(|α|+max(β,0))   [shared bt_global q]", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(HERE, "results", "quality_regression_turn1_compare.png")
    fig.savefig(out, dpi=150)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
