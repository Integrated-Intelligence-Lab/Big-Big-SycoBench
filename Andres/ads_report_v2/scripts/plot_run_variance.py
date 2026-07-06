import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / "Andres" / "ads_report_v2" / "outputs"
DEFAULT_VARIANCE = OUTPUT_DIR / "ads2_run_variance.csv"
DEFAULT_SUMMARY = OUTPUT_DIR / "ads2_summary.csv"
DEFAULT_OUTPUT = OUTPUT_DIR / "ads2_run_variance"

MODEL_ORDER = ("gpt55", "o4mini")
MODEL_LABELS = {"gpt55": "gpt-5.5", "o4mini": "o4-mini"}
MODEL_COLORS = {"gpt55": "#0173b2", "o4mini": "#de8f05"}
R_GRID = (2, 5, 10)
FULL_RUNS = 20


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_variance(path: Path) -> tuple[dict[tuple[str, ...], float], dict[str, list[tuple[str, float]]]]:
    values: dict[tuple[str, ...], float] = {}
    flips: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for row in read_csv(path):
        if row["metric"] == "direction_flip_risk":
            flips[row["model"]].append((row["artefact"], float(row["value"])))
        else:
            values[(row["model"], row["metric"], row["variant"], row["validity"], row["r_runs"])] = float(row["value"])
    return values, flips


def load_headline(path: Path) -> dict[str, dict[str, float]]:
    out = {}
    for row in read_csv(path):
        if row["horizon"] == "t1" and row["variant"] == "bt_weighted":
            out[row["model"]] = {
                "ads": float(row["ads"]),
                "ci_low": float(row["ads_ci_low"]),
                "ci_high": float(row["ads_ci_high"]),
            }
    return out


def panel_sources(ax: plt.Axes, values: dict[tuple[str, ...], float]) -> None:
    categories = (
        ("between artefacts", lambda m: math.sqrt(values[(m, "var_between_artefact_observed", "unweighted", "", "")])),
        ("between arguments\n(invalid pool)", lambda m: 100.0 * values[(m, "between_argument_sd", "", "invalid", "")]),
        ("between arguments\n(valid pool)", lambda m: 100.0 * values[(m, "between_argument_sd", "", "valid", "")]),
        (f"across runs ($R={FULL_RUNS}$)", lambda m: math.sqrt(values[(m, "var_within_run_noise", "unweighted", "", "")])),
    )
    positions = range(len(categories))
    for offset, model in zip((0.19, -0.19), MODEL_ORDER):
        heights = [getter(model) for _, getter in categories]
        ys = [pos + offset for pos in positions]
        ax.barh(ys, heights, height=0.34, color=MODEL_COLORS[model], alpha=0.85)
        for y, value in zip(ys, heights):
            ax.text(value + 0.7, y, f"{value:.1f}", va="center", ha="left", fontsize=5, color="#555555")
    ax.set_yticks(list(positions))
    ax.set_yticklabels([label for label, _ in categories], fontsize=6)
    ax.invert_yaxis()
    ax.set_xlim(0, 42)
    ax.set_xlabel("SD (points of update probability)", fontsize=7)
    ax.tick_params(axis="x", labelsize=6, length=0)
    ax.tick_params(axis="y", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_subsampling(
    ax: plt.Axes,
    values: dict[tuple[str, ...], float],
    headline: dict[str, dict[str, float]],
) -> None:
    for model in MODEL_ORDER:
        color = MODEL_COLORS[model]
        full = headline[model]
        ax.axhspan(full["ci_low"], full["ci_high"], color=color, alpha=0.10, linewidth=0)
        ax.axhline(full["ads"], color=color, linewidth=0.5, alpha=0.6)
        dodge = -0.35 if model == "gpt55" else 0.35
        for r in R_GRID:
            sd = values[(model, "subsample_point_sd", "bt_weighted", "", str(r))]
            mean = full["ads"] + values[(model, "subsample_point_bias", "bt_weighted", "", str(r))]
            ax.errorbar(
                r + dodge,
                mean,
                yerr=sd,
                fmt="o",
                color=color,
                markersize=3.0,
                capsize=1.5,
                linewidth=0.8,
                capthick=0.8,
            )
        ax.plot([FULL_RUNS + dodge], [full["ads"]], marker="o", color=color, markersize=3.0)
    ax.set_xticks([*R_GRID, FULL_RUNS])
    ax.set_xlim(0, 22.5)
    ax.set_ylim(12, 88)
    ax.set_xlabel("runs per arm ($R$)", fontsize=7)
    ax.set_ylabel(r"BT-weighted ADS", fontsize=7)
    ax.tick_params(axis="both", labelsize=6, length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_direction(ax: plt.Axes, flips: dict[str, list[tuple[str, float]]]) -> None:
    entries = [
        (artefact, risk, model)
        for model in MODEL_ORDER
        for artefact, risk in flips.get(model, [])
    ]
    entries.sort(key=lambda item: item[1])
    for y, (artefact, risk, model) in enumerate(entries):
        ax.hlines(y, 0, risk, color=MODEL_COLORS[model], linewidth=0.9)
        ax.plot([risk], [y], marker="o", color=MODEL_COLORS[model], markersize=3.2)
    ax.set_yticks(range(len(entries)))
    ax.set_yticklabels([entry[0] for entry in entries], fontsize=6)
    ax.set_xlim(0, 0.40)
    ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4])
    ax.set_xticklabels(["0", "10", "20", "30", "40"], fontsize=6)
    ax.set_xlabel(r"direction-flip probability (\%), $R=5$", fontsize=7)
    ax.tick_params(axis="both", length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.97,
        0.05,
        "all 39 other artefact--model\npairs: zero flips",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5,
        color="#777777",
        linespacing=1.3,
    )


def make_figure(
    values: dict[tuple[str, ...], float],
    flips: dict[str, list[tuple[str, float]]],
    headline: dict[str, dict[str, float]],
) -> plt.Figure:
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["text.usetex"] = True
    fig, axes = plt.subplots(1, 3, figsize=(7.08, 2.55), dpi=300, gridspec_kw={"width_ratios": [1.2, 1.0, 1.0]})
    panel_sources(axes[0], values)
    panel_subsampling(axes[1], values, headline)
    panel_direction(axes[2], flips)
    for idx, ax in enumerate(axes):
        ax.text(
            -0.02 if idx else -0.30,
            1.05,
            rf"\textbf{{{'abc'[idx]}}}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
        )
    handles = [
        Line2D([0], [0], marker="o", color=MODEL_COLORS[model], linewidth=0.9, markersize=3.2, label=MODEL_LABELS[model])
        for model in MODEL_ORDER
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=6, frameon=False)
    fig.tight_layout(rect=(0, 0.075, 1, 1), w_pad=1.6)
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
    parser.add_argument("--run-variance", type=Path, default=DEFAULT_VARIANCE)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(f"run-variance={args.run_variance}")
    print(f"summary={args.summary}")
    print(f"output-path={args.output_path}")
    values, flips = load_variance(args.run_variance)
    headline = load_headline(args.summary)
    fig = make_figure(values, flips, headline)
    save_figure(fig, args.output_path)
    plt.close("all")


if __name__ == "__main__":
    main()
