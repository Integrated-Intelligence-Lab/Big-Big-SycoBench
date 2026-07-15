import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Editable analysis settings
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "Francesca" / "results" / "vg_scoring"

# Leave empty to auto-detect batch_output_score_*.jsonl in RESULTS_DIR.
BATCH_OUTPUT_FILES = []

BASELINE_PROMPT = "domain_specific"
BASELINE_EVALUATOR = "neutral"


def extract_text(body):
    parts = []
    for item in body.get("output", []):
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    parts.append(content.get("text", ""))
    return "\n".join(parts).strip()


def parse_score(text):
    match = re.search(r"-?\d+", text)
    if not match:
        return None
    value = int(match.group())
    if 0 <= value <= 100:
        return value
    return None


def output_paths():
    if BATCH_OUTPUT_FILES:
        return [RESULTS_DIR / name for name in BATCH_OUTPUT_FILES]
    return sorted(RESULTS_DIR.glob("batch_output_score_*.jsonl"))


def load_output(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            parts = row["custom_id"].split("|")
            if len(parts) != 5 or parts[0] != "score":
                continue
            _, artefact, prompt_name, evaluator_name, run = parts
            text = extract_text(row["response"]["body"])
            rows.append(
                {
                    "source_file": path.name,
                    "artefact": artefact,
                    "prompt": prompt_name,
                    "evaluator": evaluator_name,
                    "run": int(run.removeprefix("run")),
                    "score": parse_score(text),
                    "raw_text": text,
                }
            )
    return pd.DataFrame(rows)


def save_summary(data):
    summary = (
        data.groupby(["artefact", "prompt", "evaluator"])["score"]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .round(2)
        .reset_index()
    )
    out_csv = RESULTS_DIR / "vg_score_summary.csv"
    summary.to_csv(out_csv, index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved {out_csv}")
    return summary


def save_shift_summary(summary):
    baseline = summary[
        (summary["prompt"] == BASELINE_PROMPT)
        & (summary["evaluator"] == BASELINE_EVALUATOR)
    ][["artefact", "mean"]].rename(columns={"mean": "baseline_mean"})

    shifts = summary.merge(baseline, on="artefact", how="left")
    shifts["mean_shift_vs_baseline"] = (
        shifts["mean"] - shifts["baseline_mean"]
    ).round(2)

    out_csv = RESULTS_DIR / "vg_score_shifts_vs_baseline.csv"
    shifts.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")
    return shifts


def plot_by_condition(data):
    condition_order = sorted(
        data[["prompt", "evaluator"]].drop_duplicates().itertuples(index=False, name=None)
    )
    labels = [f"{prompt}\n{evaluator}" for prompt, evaluator in condition_order]

    means = []
    errors = []
    for prompt, evaluator in condition_order:
        scores = data[(data["prompt"] == prompt) & (data["evaluator"] == evaluator)]["score"]
        means.append(scores.mean())
        errors.append(scores.std())

    fig, ax = plt.subplots(figsize=(max(9, 1.1 * len(labels)), 5))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=errors, color="tab:blue", alpha=0.75, capsize=3)
    ax.set_ylim(0, 100)
    ax.set_ylabel("score")
    ax.set_title("VG artefacts: overall score by prompt and evaluator instruction")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    fig.tight_layout()

    out_png = RESULTS_DIR / "vg_score_means_by_condition.png"
    fig.savefig(out_png, dpi=150)
    print(f"Saved {out_png}")


def plot_artifact_distributions(data):
    artefacts = sorted(data["artefact"].unique())
    fig, axes = plt.subplots(
        len(artefacts), 1, figsize=(11, max(2.0 * len(artefacts), 4)), sharex=True
    )
    if len(artefacts) == 1:
        axes = [axes]

    bins = np.arange(-0.5, 101.5, 1)
    baseline = data[
        (data["prompt"] == BASELINE_PROMPT)
        & (data["evaluator"] == BASELINE_EVALUATOR)
    ]

    for ax, artefact in zip(axes, artefacts):
        scores = baseline[baseline["artefact"] == artefact]["score"]
        if not scores.empty:
            ax.hist(scores, bins=bins, color="tab:green", alpha=0.65, edgecolor="white")
            ax.axvline(scores.mean(), color="black", linestyle="--", linewidth=1)
            ax.set_title(f"{artefact}: baseline mean={scores.mean():.1f}, sd={scores.std():.1f}", fontsize=9)
        else:
            ax.set_title(f"{artefact}: no baseline rows", fontsize=9)
        ax.set_ylabel("count")
        ax.yaxis.get_major_locator().set_params(integer=True)

    axes[-1].set_xlim(0, 100)
    axes[-1].set_xlabel("score")
    axes[-1].set_xticks(range(0, 101, 10))
    fig.suptitle("Baseline domain-specific neutral score distributions", fontsize=12)
    fig.tight_layout()

    out_png = RESULTS_DIR / "vg_baseline_distributions_by_artefact.png"
    fig.savefig(out_png, dpi=150)
    print(f"Saved {out_png}")


def main():
    paths = output_paths()
    if not paths:
        raise SystemExit(
            f"No scoring output files found in {RESULTS_DIR}. "
            "Expected batch_output_score_*.jsonl or edit BATCH_OUTPUT_FILES."
        )

    frames = []
    for path in paths:
        if not path.exists():
            print(f"Skipping missing file: {path}")
            continue
        frames.append(load_output(path))

    if not frames:
        raise SystemExit("No readable output files found.")

    data = pd.concat(frames, ignore_index=True)
    missing = data["score"].isna().sum()
    if missing:
        print(f"WARNING: {missing} rows had unparseable scores.")
    data.to_csv(RESULTS_DIR / "vg_score_rows.csv", index=False)
    data = data.dropna(subset=["score"])

    summary = save_summary(data)
    save_shift_summary(summary)
    plot_by_condition(data)
    plot_artifact_distributions(data)


if __name__ == "__main__":
    main()
