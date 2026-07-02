"""Single-shot isolated effect vs GLOBAL BT validity rating.

Same analysis as ../scripts/15_plot_singleshot_bt_vs_delta.py, but the BT rating
on the x-axis is this folder's *single-pool* score (results/bt_scores.csv) -- one
common scale across all artefacts and directions -- instead of bt_validation's
per-pool, pool-centered score. That makes the slope a single global relationship
rather than a within-pool one.

Each cycle argument was fired once from S0 (single-shot stage 2), so its effect
is clean of turn-order / accumulation. Per argument (one direction per artefact)
we plot:

  left  : mean shift in argued direction (rescore - S0)         vs BT rating
  right : mean headroom-normalized shift (shift / room-to-edge) vs BT rating

The normalized panel divides out the floor/ceiling confound, so a positive slope
there means effect genuinely scales with argument validity. Spearman overall and
within valid/invalid are printed and shown.

    python 06_plot_singleshot_bt_vs_delta.py --tag gpt55     # SUT = gpt-5.5
    python 06_plot_singleshot_bt_vs_delta.py --tag o4mini
    # explicit files override auto-discovery:
    python 06_plot_singleshot_bt_vs_delta.py --tag gpt55 \
        --s0-output <s0>.jsonl --args-output <args>.jsonl
"""
import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict
from statistics import mean

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                                  # Marthe/
BT_PATH = os.path.join(HERE, "results", "bt_scores.csv")      # global scale
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


def first_custom_id(path):
    with open(path, encoding="utf-8") as f:
        return json.loads(f.readline())["custom_id"]


def discover(tag, kind):
    """Find the S0 (custom_id has '_run') or args (custom_id has '|') batch
    output in results/singleshot/<tag>/, by peeking at the first record."""
    folder = os.path.join(ROOT, "results", "singleshot", tag)
    for path in sorted(glob.glob(os.path.join(folder, "*output*.jsonl"))):
        cid = first_custom_id(path)
        if kind == "s0" and "|" not in cid and "_run" in cid:
            return path
        if kind == "args" and "|" in cid:
            return path
    raise SystemExit(f"could not find {kind} batch output under {folder}")


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
    ap.add_argument("--tag", default="gpt55")
    ap.add_argument("--s0-output", default=None)
    ap.add_argument("--args-output", default=None)
    a = ap.parse_args()

    s0_path = a.s0_output or discover(a.tag, "s0")
    args_path = a.args_output or discover(a.tag, "args")

    s0, bt = load_s0(s0_path), load_bt()
    raw = defaultdict(list)   # (aid,dir,val,idx) -> [(delta, norm)]
    tot = unparsed = 0
    for l in open(args_path, encoding="utf-8"):
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
    missing_bt = 0
    for (aid, d, val, idx), vals in raw.items():
        b = bt.get((aid, d, val, idx))
        if b is None:
            missing_bt += 1
            continue
        md = mean([v[0] for v in vals])
        norms = [v[1] for v in vals if v[1] is not None]
        mn = mean(norms) if norms else float("nan")
        pts.append((b, md, mn, val))
    if missing_bt:
        print(f"[{a.tag}] WARNING: {missing_bt} arguments had no global BT rating")

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
        ax.set_xlabel("BT rating (argument validity, one global scale)")
        ax.set_ylabel(ylab)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Single-shot isolation vs global BT validity  [{a.tag}]  "
                 f"{len(pts)} arguments, 22 artefacts", fontsize=11)
    fig.tight_layout()
    out = os.path.join(HERE, "results", f"bt_vs_delta_{a.tag}.png")
    fig.savefig(out, dpi=150)
    print(f"[{a.tag}] saved {out}")


if __name__ == "__main__":
    main()
