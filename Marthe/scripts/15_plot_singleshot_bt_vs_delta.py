"""Single-shot BT validity vs isolated effect, per model.

Each cycle argument was fired once from S0 (stage 2), so its effect is clean of
turn-order / accumulation. We plot, per argument (one direction per artefact):

  left  : mean shift in argued direction (rescore - S0)         vs BT rating
  right : mean headroom-normalized shift (shift / room-to-edge) vs BT rating

The normalized panel divides out the floor/ceiling confound, so a positive slope
there means effect genuinely scales with argument validity. Spearman overall and
within valid/invalid printed + shown.

    python 15_plot_singleshot_bt_vs_delta.py --tag gpt55 \
        --s0-output <s0>.jsonl --args-output <args>.jsonl
"""
import argparse
import csv
import json
import os
import re
from collections import defaultdict
from statistics import mean

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BT_PATH = os.path.join(ROOT, "bt_validation", "results", "bt_scores.csv")
SIGN = {"lower": -1, "raise": +1}
COLOR = {"valid": "tab:green", "invalid": "tab:red"}


def extract_text(body):
    return "\n".join(
        c.get("text", "")
        for it in body.get("output", [])
        if it.get("type") == "message"
        for c in it.get("content", [])
        if c.get("type") == "output_text"
    )


def score(t):
    m = re.search(r"-?\d+", t)
    return int(m.group()) if m else None


def load_s0(path):
    out = {}
    for l in open(path, encoding="utf-8"):
        r = json.loads(l)
        aid, run = r["custom_id"].split("_run")
        out[(aid, int(run))] = score(extract_text(r["response"]["body"]))
    return out


def load_bt():
    bt = {}
    for r in csv.DictReader(open(BT_PATH, encoding="utf-8")):
        bt[(r["artefact_id"], r["direction"], r["validity"], int(r["idx"]))] = float(r["bt_rating"])
    return bt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--s0-output", required=True)
    ap.add_argument("--args-output", required=True)
    a = ap.parse_args()

    s0, bt = load_s0(a.s0_output), load_bt()
    raw = defaultdict(list)   # (aid,dir,val,idx) -> [(delta, norm)]
    tot = unparsed = 0
    for l in open(a.args_output, encoding="utf-8"):
        r = json.loads(l)
        tot += 1
        aid, d, val, idx, run = r["custom_id"].split("|")
        idx = int(idx[3:]); run = int(run[1:])
        sc = score(extract_text(r["response"]["body"]))
        s0v = s0.get((aid, run))
        if sc is None or s0v is None:
            unparsed += 1
            continue
        delta = SIGN[d] * (sc - s0v)
        room = (s0v - 1) if d == "lower" else (100 - s0v)
        norm = delta / room if room > 0 else None
        raw[(aid, d, val, idx)].append((delta, norm))
    print(f"[{a.tag}] args parsed: {tot-unparsed}/{tot} ({unparsed} unparsable)")

    pts = []  # (bt, mean_delta, mean_norm, validity)
    for key, vals in raw.items():
        aid, d, val, idx = key
        if key not in [(k[0], k[1], k[2], k[3]) for k in [key]]:
            pass
        b = bt.get((aid, d, val, idx))
        if b is None:
            continue
        md = mean([v[0] for v in vals])
        norms = [v[1] for v in vals if v[1] is not None]
        mn = mean(norms) if norms else float("nan")
        pts.append((b, md, mn, val))

    def rho(xs, ys):
        return spearmanr(xs, ys).statistic

    xb = [p[0] for p in pts]
    print(f"[{a.tag}] n={len(pts)} args | Spearman(BT, raw shift)={rho(xb,[p[1] for p in pts]):.2f} | "
          f"Spearman(BT, norm shift)={rho(xb,[p[2] for p in pts]):.2f}")
    for v in ("valid", "invalid"):
        sub = [p for p in pts if p[3] == v]
        print(f"    within {v} (n={len(sub)}): raw={rho([p[0] for p in sub],[p[1] for p in sub]):.2f} "
              f"norm={rho([p[0] for p in sub],[p[2] for p in sub]):.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, yi, ylab in ((axes[0], 1, "mean shift, argued direction (pts)"),
                         (axes[1], 2, "mean headroom-normalized shift")):
        xs = [p[0] for p in pts]
        ys = [p[yi] for p in pts]
        b, a0 = np.polyfit(xs, ys, 1)
        xl = np.linspace(min(xs) - 0.3, max(xs) + 0.3, 50)
        ax.plot(xl, a0 + b * xl, color="gray", ls="--", lw=1,
                label=f"fit  $\\rho$={rho(xs,ys):.2f}")
        for v in ("valid", "invalid"):
            sx = [p[0] for p in pts if p[3] == v]
            sy = [p[yi] for p in pts if p[3] == v]
            ax.scatter(sx, sy, c=COLOR[v], s=32, alpha=0.75, edgecolor="none", label=v)
        ax.axhline(0, color="black", lw=0.6)
        ax.axvline(0, color="gray", ls=":", lw=0.8)
        ax.set_xlabel("BT rating (argument validity, pool-centered)")
        ax.set_ylabel(ylab)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Single-shot isolation: does effect scale with BT validity?  [{a.tag}]  "
                 f"{len(pts)} arguments, 22 artefacts", fontsize=11)
    fig.tight_layout()
    out = os.path.join(ROOT, "results", "singleshot", a.tag, "bt_vs_delta.png")
    fig.savefig(out, dpi=150)
    print(f"[{a.tag}] saved {out}")


if __name__ == "__main__":
    main()
