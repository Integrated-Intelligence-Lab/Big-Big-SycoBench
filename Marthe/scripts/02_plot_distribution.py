import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "Marthe/results/initial_scores"

# Authorship-prime output: one file holding all three variants, distinguished by
# the custom_id ({artefact}_{variant}_run{n}). Filtered per variant below.
AUTHORSHIP_FILE = "initial_authorship_output.jsonl"

# Batches to overlay, in legend order. Each entry is one prompt variant. A
# "variant" key means the file mixes variants and only matching rows are kept.
BATCHES = [
    {
        "label": "original prompt",
        "file": "initial_default_output.jsonl",
        "color": "tab:blue",
    },
    {
        "label": 'anti-sycophancy ("don\'t be sycophantic")',
        "file": "initial_antisyco_output.jsonl",
        "color": "tab:orange",
    },
    {
        "label": "authorship: implied",
        "file": AUTHORSHIP_FILE,
        "variant": "implied",
        "color": "tab:green",
    },
    {
        "label": "authorship: stake",
        "file": AUTHORSHIP_FILE,
        "variant": "stake",
        "color": "tab:red",
    },
    {
        "label": "authorship: pride",
        "file": AUTHORSHIP_FILE,
        "variant": "pride",
        "color": "tab:purple",
    },
]


def extract_text(body):
    parts = []
    for item in body.get("output", []):
        if item.get("type") == "message":
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    parts.append(c.get("text", ""))
    return "\n".join(parts)


def parse_score(text):
    # Score is the leading integer; a justification may follow.
    m = re.search(r"-?\d+", text)
    return int(m.group()) if m else None


def load_scores(path, label, variant=None):
    records = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        custom_id = r["custom_id"]
        base = custom_id.split("_run")[0]            # "L01" or "L01_implied"
        if variant is not None:
            if not base.endswith("_" + variant):
                continue
            artefact = base[: -(len(variant) + 1)]   # strip "_<variant>" suffix
        else:
            artefact = base
        score = parse_score(extract_text(r["response"]["body"]))
        records.append({"batch": label, "artefact": artefact, "score": score})
    return pd.DataFrame(records)


def main():
    frames = []
    for b in BATCHES:
        df = load_scores(os.path.join(RESULTS_DIR, b["file"]), b["label"], b.get("variant"))
        n_missing = df["score"].isna().sum()
        if n_missing:
            print(f"WARNING: {n_missing} unparseable rows in {b['file']}")
        frames.append(df.dropna(subset=["score"]))
    data = pd.concat(frames, ignore_index=True)

    artefacts = sorted(data["artefact"].unique())

    summary = (
        data.groupby(["artefact", "batch"])["score"]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .round(2)
    )
    print(summary.to_string())

    # One panel per artefact, batches overlaid, all on the full 1-100 scale with
    # per-integer bins.
    fig, axes = plt.subplots(
        len(artefacts), 1, figsize=(10, 2.6 * len(artefacts)), sharex=True
    )
    if len(artefacts) == 1:
        axes = [axes]

    bins = np.arange(0.5, 101.5, 1)
    for ax, art in zip(axes, artefacts):
        for b in BATCHES:
            scores = data.loc[
                (data["artefact"] == art) & (data["batch"] == b["label"]), "score"
            ]
            if scores.empty:
                continue
            # Step outlines keep overlapping fills readable. Per-series μ/σ vary
            # by panel, so they live in the printed summary table, not the legend
            # (a single shared legend can only carry the prompt labels).
            ax.hist(
                scores, bins=bins, color=b["color"], alpha=0.45,
                histtype="stepfilled", edgecolor=b["color"], linewidth=1.1,
                label=b["label"],
            )
            ax.axvline(scores.mean(), color=b["color"], linestyle="--", linewidth=1.2)
        ax.set_xlim(1, 100)
        ax.set_ylabel("count")
        ax.yaxis.get_major_locator().set_params(integer=True)
        ax.set_title(art, fontsize=10, fontweight="bold")

    axes[0].legend(fontsize=8, loc="upper left")
    axes[-1].set_xlabel("score (1-100, per-integer bins)")
    axes[-1].set_xticks(range(0, 101, 10))
    fig.suptitle("Initial score distribution per artefact, by prompt", fontsize=12)
    fig.tight_layout()

    out_png = os.path.join(RESULTS_DIR, "score_distribution.png")
    fig.savefig(out_png, dpi=150)
    print(f"\nSaved plot to {out_png}")


if __name__ == "__main__":
    main()
