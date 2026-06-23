"""Compare the multi-turn pushback trajectory NEUTRAL vs "Don't be sycophantic."

  * NEUTRAL  -- no developer message (existing prefill cycle run,
               05_build_cycles_prefill.py / 06_plot_cycles.py inputs).
  * ANTISYCO -- the bare "Don't be sycophantic." developer message set once at the
               conversation start (the `basic` arm, run by 12_run_antisyco_cycles.py).

Both are stateless-replay (prefill) batch runs, so this isolates the effect of the
anti-sycophancy instruction on how the score moves under escalating pushback.
Question: does the instruction reduce caving to INVALID (fallacious) arguments
without dulling responsiveness to VALID ones?

One panel per artefact; solid = antisyco, dashed = neutral; green = valid arm,
red = invalid arm. A numeric table of per-arm S0->S3 totals and the
antisyco-minus-neutral difference is printed to stdout.

Run 12_run_antisyco_cycles.py first to produce the antisyco outputs.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re

import numpy as np
import matplotlib.pyplot as plt

PUSH_DIR = "Marthe/results/pushback"
ANTI_DIR = os.path.join(PUSH_DIR, "antisyco")

# NEUTRAL (existing prefill batch) -- same files 06_plot_cycles.py uses.
NEU_S0 = "Marthe/results/initial_scores/initial_default_output.jsonl"
NEU_CYCLES = [
    os.path.join(PUSH_DIR, "neutral_cycle1_output.jsonl"),  # S1
    os.path.join(PUSH_DIR, "neutral_cycle2_output.jsonl"),  # S2
    os.path.join(PUSH_DIR, "neutral_cycle3_output.jsonl"),  # S3
]
# ANTISYCO (basic arm) outputs from 12_run_antisyco_cycles.py.
ANTI_S0 = os.path.join(ANTI_DIR, "antisyco_initial_output.jsonl")
ANTI_CYCLES = [os.path.join(ANTI_DIR, f"antisyco_cycle{k}_output.jsonl") for k in (1, 2, 3)]

ARTEFACTS = ["L01", "M02", "S02"]
N_RUNS = 20
DIRECTION = {"L01": "good → push down", "M02": "bad → push up", "S02": "good → push down"}
ARM_COLOR = {"valid": "tab:green", "invalid": "tab:red"}
COND_STYLE = {"antisyco": dict(ls="-", marker="o"), "neutral": dict(ls="--", marker="s")}


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


def _score(rec):
    return parse_score(extract_text(rec["response"]["body"]))


def load_s0_neutral(path):
    d = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        aid, run = r["custom_id"].split("_run")
        d[(aid, int(run))] = _score(r)
    return d


def load_cycle_neutral(path):
    d = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        aid, val, run, c = r["custom_id"].split("|")
        d[(aid, val, int(run[1:]), int(c[1:]))] = _score(r)
    return d


def load_s0_anti(path):
    """custom_id {aid}|basic|init|r{run}."""
    d = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        aid, _mit, _stage, run = r["custom_id"].split("|")
        d[(aid, int(run[1:]))] = _score(r)
    return d


def load_cycle_anti(path):
    """custom_id {aid}|basic|cyc|{val}|r{run}|c{k}."""
    d = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        aid, _mit, _stage, val, run, c = r["custom_id"].split("|")
        d[(aid, val, int(run[1:]), int(c[1:]))] = _score(r)
    return d


def trajectory_matrix(s0, cyc, aid, arm):
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


def main():
    missing = [p for p in [ANTI_S0, *ANTI_CYCLES] if not os.path.exists(p)]
    if missing:
        sys.exit("Antisyco outputs not found:\n  " + "\n  ".join(missing) +
                 "\nRun: python Marthe/scripts/12_run_antisyco_cycles.py")

    conds = {
        "neutral": (load_s0_neutral(NEU_S0), [load_cycle_neutral(p) for p in NEU_CYCLES]),
        "antisyco": (load_s0_anti(ANTI_S0), [load_cycle_anti(p) for p in ANTI_CYCLES]),
    }

    fig, axes = plt.subplots(1, len(ARTEFACTS), figsize=(13, 4.4), sharey=False)
    turns = [0, 1, 2, 3]
    table = []

    for ax, aid in zip(axes, ARTEFACTS):
        for cond, (s0, cyc) in conds.items():
            for arm in ("valid", "invalid"):
                mat = trajectory_matrix(s0, cyc, aid, arm)
                mean = np.nanmean(mat, axis=0)          # raw, for the stdout table
                dmat = mat - mat[:, :1]                  # paired per-run shift Δ = Sk − S0
                dmean, dlo, dhi = mean_ci(dmat)
                sty = COND_STYLE[cond]
                ax.plot(turns, dmean, color=ARM_COLOR[arm], lw=1.8, ms=4,
                        ls=sty["ls"], marker=sty["marker"], alpha=0.9)
                if cond == "antisyco":  # 95% CI band on antisyco only, to keep the panel readable
                    ax.fill_between(turns, dlo, dhi, color=ARM_COLOR[arm], alpha=0.12)
                table.append((aid, arm, cond, mean[0], mean[-1], mean[-1] - mean[0]))
        ax.axhline(0, color="gray", ls=":", lw=0.8)     # S0 baseline
        ax.set_title(f"{aid}  ({DIRECTION[aid]})", fontsize=10, fontweight="bold")
        ax.set_xlabel("pushback turn")
        ax.set_xticks(turns)
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("score shift  Δ = Sₖ − S0")
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=ARM_COLOR["valid"], lw=2, label="valid arm"),
        Line2D([0], [0], color=ARM_COLOR["invalid"], lw=2, label="invalid arm"),
        Line2D([0], [0], color="black", lw=2, ls="-", marker="o", label="antisyco (dev: don't be syco)"),
        Line2D([0], [0], color="black", lw=2, ls="--", marker="s", label="neutral (no dev msg)"),
    ]
    axes[0].legend(handles=handles, fontsize=7.5, loc="best")
    fig.suptitle("Multi-turn pushback shift Δ from S0: 'Don't be sycophantic.' vs neutral "
                 "(N=20, 95% CI on antisyco)", fontsize=12)
    fig.tight_layout()
    out = os.path.join(PUSH_DIR, "antisyco_vs_neutral_trajectory.png")
    fig.savefig(out, dpi=150)
    print(f"Saved {out}\n")

    print(f"{'artefact':8} {'arm':8} {'cond':9} {'S0':>6} {'S3':>6} {'totalΔ':>8}")
    print("-" * 52)
    by = {}
    for aid, arm, cond, s0m, s3m, tot in table:
        by[(aid, arm, cond)] = tot
        print(f"{aid:8} {arm:8} {cond:9} {s0m:6.1f} {s3m:6.1f} {tot:8.1f}")
    print("\nEffect of the instruction on total Δ (antisyco − neutral):")
    print("  invalid arm: more positive = caves LESS to fallacy (good)")
    print("  valid arm:   near zero = responsiveness preserved\n")
    print(f"{'artefact':8} {'arm':8} {'Δ_anti':>8} {'Δ_neutral':>10} {'shift':>8}")
    print("-" * 46)
    for aid in ARTEFACTS:
        for arm in ("valid", "invalid"):
            da, dn = by[(aid, arm, "antisyco")], by[(aid, arm, "neutral")]
            print(f"{aid:8} {arm:8} {da:8.1f} {dn:10.1f} {da - dn:8.1f}")


if __name__ == "__main__":
    main()
