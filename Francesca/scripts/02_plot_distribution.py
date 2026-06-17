import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Editable analysis settings
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "Francesca" / "results" / "initial_scores"

# After downloading OpenAI Batch API outputs, put them in RESULTS_DIR and list
# them here. These names are placeholders; update them to match your files.
BATCHES = [
    {
        "label": "original prompt",
        "file": "batch_output_original.jsonl",
        "color": "tab:blue",
    },
    {
        "label": 'anti-sycophancy ("don\'t be sycophantic")',
        "file": "batch_output_anti_sycophantic.jsonl",
        "color": "tab:orange",
    },
]


def extract_text(body):
    parts = []
    for item in body.get("output", []):
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    parts.append(content.get("text", ""))
    return "\n".join(parts)


def parse_score(text):
    match = re.search(r"-?\d+", text)
    if not match:
        return None
    score = int(match.group())
    if 1 <= score <= 100:
        return score
    return None


def load_scores(path, label):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            custom_id = row["custom_id"]
            artefact = custom_id.split("_run")[0]
            text = extract_text(row["response"]["body"])
            score = parse_score(text)
            records.append(
                {
                    "batch": label,
                    "artefact": artefact,
                    "score": score,
                    "raw_text": text,
                }
            )
    return pd.DataFrame(records)


def main():
    frames = []
    for batch in BATCHES:
        path = RESULTS_DIR / batch["file"]
        if not path.exists():
            print(f"Skipping missing file: {path}")
            continue

        df = load_scores(path, batch["label"])
        n_missing = df["score"].isna().sum()
        if n_missing:
            print(f"WARNING: {n_missing} unparseable rows in {batch['file']}")
        frames.append(df.dropna(subset=["score"]))

    if not frames:
        raise SystemExit("No batch output files found. Update BATCHES first.")

    data = pd.concat(frames, ignore_index=True)
    artefacts = sorted(data["artefact"].unique())

    summary = (
        data.groupby(["artefact", "batch"])["score"]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .round(2)
    )
    print(summary.to_string())
    summary.to_csv(RESULTS_DIR / "score_summary.csv")

    fig, axes = plt.subplots(
        len(artefacts), 1, figsize=(10, 2.6 * len(artefacts)), sharex=True
    )
    if len(artefacts) == 1:
        axes = [axes]

    bins = np.arange(0.5, 101.5, 1)
    for ax, artefact in zip(axes, artefacts):
        title_bits = []
        for batch in BATCHES:
            scores = data.loc[
                (data["artefact"] == artefact)
                & (data["batch"] == batch["label"]),
                "score",
            ]
            if scores.empty:
                continue
            ax.hist(
                scores,
                bins=bins,
                color=batch["color"],
                alpha=0.6,
                edgecolor="white",
                label=batch["label"],
            )
            ax.axvline(scores.mean(), color=batch["color"], linestyle="--", linewidth=1.2)
            title_bits.append(
                f"{batch['label']}: mean={scores.mean():.1f}, sd={scores.std():.1f}"
            )

        ax.set_xlim(1, 100)
        ax.set_ylabel("count")
        ax.yaxis.get_major_locator().set_params(integer=True)
        ax.set_title(f"{artefact}   " + "   |   ".join(title_bits), fontsize=9)

    axes[0].legend(fontsize=8, loc="upper left")
    axes[-1].set_xlabel("score (1-100, per-integer bins)")
    axes[-1].set_xticks(range(0, 101, 10))
    fig.suptitle("Initial score distribution per artefact, by prompt", fontsize=12)
    fig.tight_layout()

    out_png = RESULTS_DIR / "score_distribution.png"
    fig.savefig(out_png, dpi=150)
    print(f"\nSaved plot to {out_png}")


if __name__ == "__main__":
    main()
