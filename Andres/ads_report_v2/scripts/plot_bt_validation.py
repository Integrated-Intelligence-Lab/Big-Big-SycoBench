import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / "Andres" / "ads_report_v2" / "outputs"
DEFAULT_POINTS = OUTPUT_DIR / "ads2_argument_points.csv"
DEFAULT_SUMMARY = OUTPUT_DIR / "ads2_summary.csv"
DEFAULT_OUTPUT = OUTPUT_DIR / "ads2_bt_validation"

MODEL_ORDER = ("gpt55", "o4mini")
MODEL_LABELS = {"gpt55": "gpt-5.5", "o4mini": "o4-mini"}
COLORS = {"valid": "#029e73", "invalid": "#d55e00"}
N_BINS = 8
BIN_MIN_POINTS = 3


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_points(path: Path) -> dict[str, list[dict[str, object]]]:
    out: dict[str, list[dict[str, object]]] = {}
    for row in read_csv(path):
        out.setdefault(row["model"], []).append({
            "bt": float(row["bt_rating"]),
            "z": float(row["z_mean"]),
            "validity": row["validity"],
        })
    return out


def load_weighted_summary(path: Path) -> dict[str, dict[str, float]]:
    out = {}
    for row in read_csv(path):
        if row["variant"] == "bt_weighted" and row["horizon"] == "t1":
            out[row["model"]] = {
                "tpr": float(row["tpr"]),
                "fpr": float(row["fpr"]),
                "ads": float(row["ads"]),
            }
    return out


def median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def boundary(points: list[dict[str, object]]) -> float:
    valid = [float(p["bt"]) for p in points if p["validity"] == "valid"]
    invalid = [float(p["bt"]) for p in points if p["validity"] == "invalid"]
    return 0.5 * (median(valid) + median(invalid))


def binned_means(points: list[dict[str, object]]) -> list[tuple[float, float, float]]:
    lo = min(float(p["bt"]) for p in points)
    hi = max(float(p["bt"]) for p in points)
    width = (hi - lo) / N_BINS
    bins: list[list[float]] = [[] for _ in range(N_BINS)]
    centers = [lo + (i + 0.5) * width for i in range(N_BINS)]
    for point in points:
        i = min(int((float(point["bt"]) - lo) / width), N_BINS - 1)
        bins[i].append(float(point["z"]))
    out = []
    for center, values in zip(centers, bins):
        if len(values) < BIN_MIN_POINTS:
            continue
        mean = sum(values) / len(values)
        sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
        out.append((center, mean, 1.96 * sd / math.sqrt(len(values))))
    return out


def style_axis(ax: plt.Axes, col_idx: int) -> None:
    ax.grid(True, color="#d9d9d9", linewidth=0.35, alpha=0.7)
    ax.axhline(0, color="#666666", linewidth=0.45)
    ax.tick_params(axis="both", labelsize=6, length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("argument BT validity score (log-odds)", fontsize=7)
    if col_idx == 0:
        ax.set_ylabel(r"mean shift toward request ($z$)", fontsize=7)


def make_figure(
    points: dict[str, list[dict[str, object]]],
    summary: dict[str, dict[str, float]],
) -> plt.Figure:
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["text.usetex"] = True
    fig, axes = plt.subplots(1, 2, figsize=(7.08, 3.1), dpi=300, sharex=True, sharey=True)
    for col_idx, model in enumerate(MODEL_ORDER):
        ax = axes[col_idx]
        rows = points[model]
        scores = summary[model]
        for validity in ("invalid", "valid"):
            subset = [row for row in rows if row["validity"] == validity]
            ax.scatter(
                [float(row["bt"]) for row in subset],
                [float(row["z"]) for row in subset],
                s=12,
                color=COLORS[validity],
                alpha=0.74,
                linewidths=0,
            )
        ax.axvline(boundary(rows), color="#666666", linewidth=0.6, linestyle=":")
        binned = binned_means(rows)
        ax.errorbar(
            [b[0] for b in binned],
            [b[1] for b in binned],
            yerr=[b[2] for b in binned],
            color="black",
            linewidth=1.1,
            marker="o",
            markersize=2.6,
            capsize=1.6,
            capthick=0.8,
            elinewidth=0.8,
        )
        ax.set_title(MODEL_LABELS[model], fontsize=7, fontweight="bold", pad=4)
        ax.text(
            -0.02,
            1.06,
            rf"\textbf{{{'ab'[col_idx]}}}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
        )
        ax.text(
            0.03,
            0.96,
            (
                rf"ADS$_w$={scores['ads']:.0f}" + "\n"
                rf"$p_{{\mathrm{{val}},w}}={scores['tpr']:.2f}$" + "\n"
                rf"$p_{{\mathrm{{inv}},w}}={scores['fpr']:.2f}$"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=5.5,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.88},
        )
        style_axis(ax, col_idx)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["valid"], markersize=3.5, label="valid argument"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["invalid"], markersize=3.5, label="invalid argument"),
        Line2D([0], [0], color="black", linewidth=1.1, marker="o", markersize=2.6, label=r"binned mean $\pm$95\% CI"),
        Line2D([0], [0], color="#666666", linewidth=0.6, linestyle=":", label="valid/invalid BT boundary"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6, frameon=False)
    fig.suptitle("Realized turn-1 updates against the graded BT validity scale", fontsize=9, y=0.995)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95), w_pad=0.9)
    return fig


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    try:
        fig.savefig(output_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
        fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    except RuntimeError:
        plt.rcParams["text.usetex"] = False
        for ax in fig.axes:
            for text in [ax.title, ax.xaxis.label, ax.yaxis.label]:
                text.set_usetex(False)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_usetex(False)
            for artist in ax.texts:
                artist.set_usetex(False)
        if fig._suptitle is not None:
            fig._suptitle.set_usetex(False)
        fig.savefig(output_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
        fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=Path, default=DEFAULT_POINTS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(f"points={args.points}")
    print(f"summary={args.summary}")
    print(f"output-path={args.output_path}")
    points = load_points(args.points)
    summary = load_weighted_summary(args.summary)
    fig = make_figure(points, summary)
    save_figure(fig, args.output_path)
    plt.close("all")


if __name__ == "__main__":
    main()
