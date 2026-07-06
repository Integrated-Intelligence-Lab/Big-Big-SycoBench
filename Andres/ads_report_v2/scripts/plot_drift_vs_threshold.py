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
DEFAULT_CURVES = OUTPUT_DIR / "ads2_turn_curves.csv"
DEFAULT_SUMMARY = OUTPUT_DIR / "ads2_summary.csv"
DEFAULT_OUTPUT = OUTPUT_DIR / "ads2_drift_vs_threshold"

DEFAULT_MODEL_ORDER = ("gpt55", "o4mini")
FULL_MODEL_ORDER = ("gpt55", "gpt55_prid", "gpt52_prid", "gpt5_prid", "o3_prid", "gpt41_prid", "o4mini")
MODEL_LABELS = {
    "gpt55": "gpt-5.5 original",
    "gpt55_prid": "gpt-5.5 PRID",
    "gpt52_prid": "gpt-5.2 PRID",
    "gpt5_prid": "gpt-5 PRID",
    "o3_prid": "o3 PRID",
    "gpt41_prid": "gpt-4.1 PRID",
    "o4mini": "o4-mini original",
}
COMPACT_LABELS = {"gpt55": "gpt-5.5", "o4mini": "o4-mini"}
VALIDITY_ORDER = ("valid", "invalid")
COLORS = {"valid": "#029e73", "invalid": "#d55e00"}
RATE_FIELDS = {
    "valid": ("tpr", "tpr_ci_low", "tpr_ci_high"),
    "invalid": ("fpr", "fpr_ci_low", "fpr_ci_high"),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_drift(path: Path) -> dict[tuple[str, str], list[dict[str, float]]]:
    out: dict[tuple[str, str], list[dict[str, float]]] = {}
    for row in read_csv(path):
        if row["variant"] != "unweighted":
            continue
        turn = int(row["turn"])
        if turn == 0:
            continue
        key = (row["model"], row["validity"])
        out.setdefault(key, []).append({
            "turn": turn,
            "mean": float(row["mean_z"]),
            "ci_low": float(row["ci_low"]),
            "ci_high": float(row["ci_high"]),
        })
    for values in out.values():
        values.sort(key=lambda point: point["turn"])
    return out


def load_rates(path: Path, delta: str) -> dict[tuple[str, str], list[dict[str, float]]]:
    out: dict[tuple[str, str], list[dict[str, float]]] = {}
    for row in read_csv(path):
        if row["variant"] != "unweighted" or row["delta"] != delta:
            continue
        if row["horizon"] not in ("t1", "t2", "t3"):
            continue
        turn = int(row["horizon"][1])
        for validity in VALIDITY_ORDER:
            mean_field, low_field, high_field = RATE_FIELDS[validity]
            out.setdefault((row["model"], validity), []).append({
                "turn": turn,
                "mean": float(row[mean_field]),
                "ci_low": float(row[low_field]),
                "ci_high": float(row[high_field]),
            })
    for values in out.values():
        values.sort(key=lambda point: point["turn"])
    return out


def resolve_models(models: list[str] | None) -> tuple[str, ...]:
    if not models:
        return DEFAULT_MODEL_ORDER
    if len(models) == 1 and models[0] == "all":
        return FULL_MODEL_ORDER
    unknown = [model for model in models if model not in MODEL_LABELS]
    if unknown:
        raise SystemExit(f"unknown models: {', '.join(unknown)}")
    return tuple(models)


def rounded_limit(value: float) -> float:
    return max(20.0, 5.0 * math.ceil(value * 1.05 / 5.0))


def drift_limit(drift: dict[tuple[str, str], list[dict[str, float]]], models: tuple[str, ...]) -> float:
    highs = [
        point["ci_high"]
        for model in models
        for validity in VALIDITY_ORDER
        for point in drift[(model, validity)]
    ]
    return rounded_limit(max(highs))


def style_metric_axis(ax: plt.Axes, metric: str, show_xlabel: bool, show_ylabel: bool, drift_ylim: float) -> None:
    ax.grid(True, color="#d9d9d9", linewidth=0.35, alpha=0.7)
    ax.set_xticks([1, 2, 3])
    ax.tick_params(axis="both", labelsize=6, length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if show_xlabel:
        ax.set_xlabel("arguments seen", fontsize=7)
    if show_ylabel and metric == "drift":
        ax.set_ylabel(r"mean shift toward request ($z$)", fontsize=7)
    if show_ylabel and metric == "rate":
        ax.set_ylabel(r"threshold update rate", fontsize=7)
    if metric == "drift":
        ax.set_ylim(0.0, drift_ylim)
    if metric == "rate":
        ax.set_ylim(0.0, 1.05)


def plot_series(ax: plt.Axes, series: dict[tuple[str, str], list[dict[str, float]]], model: str) -> None:
    for validity in VALIDITY_ORDER:
        points = series[(model, validity)]
        turns = [point["turn"] for point in points]
        means = [point["mean"] for point in points]
        ax.plot(
            turns,
            means,
            color=COLORS[validity],
            linewidth=1.2,
            marker="o",
            markersize=3.2,
        )
        ax.fill_between(
            turns,
            [point["ci_low"] for point in points],
            [point["ci_high"] for point in points],
            color=COLORS[validity],
            alpha=0.18,
            linewidth=0,
        )


def make_figure(
    drift: dict[tuple[str, str], list[dict[str, float]]],
    rates: dict[tuple[str, str], list[dict[str, float]]],
    models: tuple[str, ...],
) -> plt.Figure:
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["text.usetex"] = True
    plt.rcParams["legend.title_fontsize"] = 7
    if len(models) > 2:
        return make_full_figure(drift, rates, models)
    fig, axes = plt.subplots(2, 2, figsize=(7.08, 4.45), dpi=300, sharex=True, sharey="row")
    for col_idx, model in enumerate(models):
        plot_series(axes[0, col_idx], drift, model)
        plot_series(axes[1, col_idx], rates, model)
        axes[0, col_idx].set_title(COMPACT_LABELS.get(model, MODEL_LABELS[model]), fontsize=7, fontweight="bold", pad=4)
        for row_idx in range(2):
            axes[row_idx, col_idx].text(
                -0.14 if col_idx == 0 else -0.08,
                1.06,
                rf"\textbf{{{'abcd'[row_idx * 2 + col_idx]}}}",
                transform=axes[row_idx, col_idx].transAxes,
                ha="left",
                va="bottom",
                fontsize=7,
                clip_on=False,
            )
            style_metric_axis(
                axes[row_idx, col_idx],
                "drift" if row_idx == 0 else "rate",
                row_idx == 1,
                col_idx == 0,
                20.0,
            )
    handles = [
        Line2D([0], [0], color=COLORS["valid"], linewidth=1.2, marker="o", markersize=3.2, label="valid arguments"),
        Line2D([0], [0], color=COLORS["invalid"], linewidth=1.2, marker="o", markersize=3.2, label="invalid arguments"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=6, frameon=False)
    fig.tight_layout(rect=(0, 0.065, 1, 1), w_pad=1.0, h_pad=1.1)
    return fig


def make_full_figure(
    drift: dict[tuple[str, str], list[dict[str, float]]],
    rates: dict[tuple[str, str], list[dict[str, float]]],
    models: tuple[str, ...],
) -> plt.Figure:
    n_models = len(models)
    drift_ylim = drift_limit(drift, models)
    fig, axes = plt.subplots(n_models, 2, figsize=(7.08, 1.08 * n_models + 0.85), dpi=300, sharex=True)
    letters = "abcdefghijklmn"
    for row_idx, model in enumerate(models):
        plot_series(axes[row_idx, 0], drift, model)
        plot_series(axes[row_idx, 1], rates, model)
        axes[row_idx, 0].text(
            -0.24,
            0.5,
            rf"\textbf{{{MODEL_LABELS[model]}}}",
            transform=axes[row_idx, 0].transAxes,
            ha="right",
            va="center",
            fontsize=6,
            clip_on=False,
        )
        for col_idx, metric in enumerate(("drift", "rate")):
            axes[row_idx, col_idx].text(
                -0.12,
                1.05,
                rf"\textbf{{{letters[row_idx * 2 + col_idx]}}}",
                transform=axes[row_idx, col_idx].transAxes,
                ha="left",
                va="bottom",
                fontsize=7,
                clip_on=False,
            )
            style_metric_axis(
                axes[row_idx, col_idx],
                metric,
                row_idx == n_models - 1,
                row_idx == n_models // 2,
                drift_ylim,
            )
    axes[0, 0].set_title("cumulative drift", fontsize=7, fontweight="bold", pad=4)
    axes[0, 1].set_title(r"$\Delta \geq 5$ update rate", fontsize=7, fontweight="bold", pad=4)
    handles = [
        Line2D([0], [0], color=COLORS["valid"], linewidth=1.2, marker="o", markersize=3.2, label="valid arguments"),
        Line2D([0], [0], color=COLORS["invalid"], linewidth=1.2, marker="o", markersize=3.2, label="invalid arguments"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=6, frameon=False)
    fig.tight_layout(rect=(0.12, 0.055, 1, 0.985), w_pad=1.0, h_pad=0.62)
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
        fig.savefig(output_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
        fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curves", type=Path, default=DEFAULT_CURVES)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--delta", default="5")
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models", nargs="+")
    args = parser.parse_args()
    print(f"curves={args.curves}")
    print(f"summary={args.summary}")
    print(f"delta={args.delta}")
    print(f"output-path={args.output_path}")
    drift = load_drift(args.curves)
    rates = load_rates(args.summary, args.delta)
    models = resolve_models(args.models)
    print(f"models={','.join(models)}")
    fig = make_figure(drift, rates, models)
    save_figure(fig, args.output_path)
    plt.close("all")


if __name__ == "__main__":
    main()
