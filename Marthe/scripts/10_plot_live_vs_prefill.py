"""Compare the multi-turn pushback trajectory under the two state-carrying methods:

  * PREFILL  -- stateless replay via the Batch API (05_build_cycles_prefill.py):
               prior turns re-sent as bare assistant score-numbers, the model's
               real reasoning discarded each turn.
  * LIVE     -- Responses API with previous_response_id (09_run_cycles_live.py):
               the server carries the real prior turns, including the model's
               hidden reasoning, between turns.

This answers the methodological question: does carrying the model's actual
reasoning state move the scores differently than prefilling bare numbers?

The two runs use independently sampled S0 (live regenerates S0 fresh), so this is
a distribution-level comparison (mean +/- sd over the 20 runs per turn), not a
run-paired diff. One panel per artefact; solid = live, dashed = prefill;
green = valid arm (responsiveness), red = invalid arm (sycophancy). A numeric
table of per-arm S0->S3 totals and the method gap is printed to stdout.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re

import numpy as np
import matplotlib.pyplot as plt

PUSH_DIR = "Marthe/results/pushback"
LIVE_DIR = os.path.join(PUSH_DIR, "live")

# Prefill (Batch API) outputs, in turn order -- same files 06_plot_cycles.py uses.
PREFILL = {
    "s0": "Marthe/results/initial_scores/initial_default_output.jsonl",
    "cycles": [
        os.path.join(PUSH_DIR, "neutral_cycle1_output.jsonl"),  # S1
        os.path.join(PUSH_DIR, "neutral_cycle2_output.jsonl"),  # S2
        os.path.join(PUSH_DIR, "neutral_cycle3_output.jsonl"),  # S3
    ],
}
LIVE = {
    "s0": os.path.join(LIVE_DIR, "live_s0_output.jsonl"),
    "cycles": [os.path.join(LIVE_DIR, f"live_cycle{k}_output.jsonl") for k in (1, 2, 3)],
}

ARTEFACTS = ["L01", "M02", "S02"]
N_RUNS = 20
DIRECTION = {"L01": "good → push down", "M02": "bad → push up", "S02": "good → push down"}
ARM_COLOR = {"valid": "tab:green", "invalid": "tab:red"}
METHOD_STYLE = {"live": dict(ls="-", marker="o"), "prefill": dict(ls="--", marker="s")}


# ----------------------------- loaders --------------------------------------
def extract_text(body):
    return "\n".join(
        c.get("text", "")
        for it in body.get("output", [])
        if it.get("type") == "message"
        for c in it.get("content", [])
        if c.get("type") == "output_text"
    )


def parse_score(text):
    m = re.search(r"-?\d+", text or "")
    return int(m.group()) if m else None


def load_s0(path):
    d = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        aid, run = r["custom_id"].split("_run")
        d[(aid, int(run))] = parse_score(extract_text(r["response"]["body"]))
    return d


def load_cycle(path):
    d = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        aid, val, run, c = r["custom_id"].split("|")
        d[(aid, val, int(run[1:]), int(c[1:]))] = parse_score(extract_text(r["response"]["body"]))
    return d


def load_method(spec):
    s0 = load_s0(spec["s0"])
    cyc = [load_cycle(p) for p in spec["cycles"]]
    return s0, cyc


def trajectory_matrix(s0, cyc, aid, arm):
    """rows = run, cols = [S0, S1, S2, S3]; NaN where a score is missing."""
    rows = []
    for r in range(N_RUNS):
        row = [s0.get((aid, r))] + [cyc[k].get((aid, arm, r, k + 1)) for k in range(3)]
        rows.append([np.nan if v is None else v for v in row])
    return np.array(rows, dtype=float)


def mean_ci(mat):
    """Per-column mean and 95% bootstrap CI for a runs×turns matrix (NaN-safe)."""
    rng = np.random.default_rng(0)
    means, los, his = [], [], []
    for j in range(mat.shape[1]):
        v = mat[:, j]; v = v[~np.isnan(v)]
        if len(v) == 0:
            means += [np.nan]; los += [np.nan]; his += [np.nan]; continue
        m = v.mean(); means.append(m)
        if np.allclose(v, v[0]):
            los.append(m); his.append(m)
        else:
            boot = rng.choice(v, size=(2000, len(v)), replace=True).mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5]); los.append(lo); his.append(hi)
    return np.array(means), np.array(los), np.array(his)


# ----------------------------- main -----------------------------------------
def main():
    methods = {"live": load_method(LIVE), "prefill": load_method(PREFILL)}

    fig, axes = plt.subplots(1, len(ARTEFACTS), figsize=(13, 4.4), sharey=False)
    turns = [0, 1, 2, 3]
    table = []

    for ax, aid in zip(axes, ARTEFACTS):
        for method, (s0, cyc) in methods.items():
            for arm in ("valid", "invalid"):
                mat = trajectory_matrix(s0, cyc, aid, arm)
                mean = np.nanmean(mat, axis=0)          # raw, for the stdout table
                dmat = mat - mat[:, :1]                  # paired per-run shift Δ = Sk − S0
                dmean, dlo, dhi = mean_ci(dmat)
                sty = METHOD_STYLE[method]
                ax.plot(turns, dmean, color=ARM_COLOR[arm], lw=1.8, ms=4,
                        ls=sty["ls"], marker=sty["marker"], alpha=0.9)
                if method == "live":  # 95% CI band on live only, to keep the panel readable
                    ax.fill_between(turns, dlo, dhi, color=ARM_COLOR[arm], alpha=0.12)
                table.append((aid, arm, method, mean[0], mean[-1], mean[-1] - mean[0]))
        ax.axhline(0, color="gray", ls=":", lw=0.8)     # S0 baseline
        ax.set_title(f"{aid}  ({DIRECTION[aid]})", fontsize=10, fontweight="bold")
        ax.set_xlabel("pushback turn")
        ax.set_xticks(turns)
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("score shift  Δ = Sₖ − S0")
    # Legend: arm color + method linestyle.
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=ARM_COLOR["valid"], lw=2, label="valid arm"),
        Line2D([0], [0], color=ARM_COLOR["invalid"], lw=2, label="invalid arm"),
        Line2D([0], [0], color="black", lw=2, ls="-", marker="o", label="live (previous_response_id)"),
        Line2D([0], [0], color="black", lw=2, ls="--", marker="s", label="prefill (batch replay)"),
    ]
    axes[0].legend(handles=handles, fontsize=7.5, loc="best")
    fig.suptitle("Multi-turn pushback shift Δ from S0: live previous_response_id vs prefill replay "
                 "(N=20, 95% CI on live)", fontsize=12)
    fig.tight_layout()
    out = os.path.join(PUSH_DIR, "live_vs_prefill_trajectory.png")
    fig.savefig(out, dpi=150)
    print(f"Saved {out}\n")

    # ---- numeric comparison table ------------------------------------------
    print(f"{'artefact':8} {'arm':8} {'method':8} {'S0':>6} {'S3':>6} {'totalΔ':>8}")
    print("-" * 50)
    by_key = {}
    for aid, arm, method, s0m, s3m, tot in table:
        by_key[(aid, arm, method)] = (s0m, s3m, tot)
        print(f"{aid:8} {arm:8} {method:8} {s0m:6.1f} {s3m:6.1f} {tot:8.1f}")
    print("\nMethod gap in total Δ (live − prefill); large |gap| = method matters:")
    print(f"{'artefact':8} {'arm':8} {'Δ_live':>8} {'Δ_prefill':>10} {'gap':>8}")
    print("-" * 46)
    for aid in ARTEFACTS:
        for arm in ("valid", "invalid"):
            dl = by_key[(aid, arm, "live")][2]
            dp = by_key[(aid, arm, "prefill")][2]
            print(f"{aid:8} {arm:8} {dl:8.1f} {dp:10.1f} {dl - dp:8.1f}")


if __name__ == "__main__":
    main()
