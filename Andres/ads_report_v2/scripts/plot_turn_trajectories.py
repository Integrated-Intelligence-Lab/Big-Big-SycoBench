import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / "Andres" / "ads_report_v2" / "outputs"
DEFAULT_CURVES = OUTPUT_DIR / "ads2_turn_curves.csv"
DEFAULT_OUTPUT = OUTPUT_DIR / "ads2_turn_trajectories"

MODEL_ORDER = ("gpt55", "o4mini")
MODEL_LABELS = {"gpt55": "gpt-5.5", "o4mini": "o4-mini"}
COLORS = {"valid": "#029e73", "invalid": "#d55e00"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_curves(path: Path) -> dict[tuple[str, str, str], list[dict[str, float]]]:
    out: dict[tuple[str, str, str], list[dict[str, float]]] = {}
    for row in read_csv(path):
        out.setdefault((row["model"], row["variant"], row["validity"]), []).append({
            "turn": int(row["turn"]),
            "mean_z": float(row["mean_z"]),
            "ci_low": float(row["ci_low"]),
            "ci_high": float(row["ci_high"]),
        })
    for curve in out.values():
        curve.sort(key=lambda point: point["turn"])
    return out


def style_axis(ax: plt.Axes, col_idx: int) -> None:
    ax.grid(True, color="#d9d9d9", linewidth=0.35, alpha=0.7)
    ax.axhline(0, color="#666666", linewidth=0.45)
    ax.set_xticks([0, 1, 2, 3])
    ax.tick_params(axis="both", labelsize=6, length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("arguments seen", fontsize=7)
    if col_idx == 0:
        ax.set_ylabel(r"cumulative shift toward request ($z$)", fontsize=7)


def make_figure(curves: dict[tuple[str, str], list[dict[str, float]]]) -> plt.Figure:
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["text.usetex"] = True
    fig, axes = plt.subplots(1, 2, figsize=(7.08, 2.9), dpi=300, sharex=True, sharey=True)
    for col_idx, model in enumerate(MODEL_ORDER):
        ax = axes[col_idx]
        for validity in ("valid", "invalid"):
            curve = curves[(model, "unweighted", validity)]
            turns = [point["turn"] for point in curve]
            means = [point["mean_z"] for point in curve]
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
                [point["ci_low"] for point in curve],
                [point["ci_high"] for point in curve],
                color=COLORS[validity],
                alpha=0.18,
                linewidth=0,
            )
            weighted = curves[(model, "bt_weighted", validity)]
            ax.plot(
                [point["turn"] for point in weighted],
                [point["mean_z"] for point in weighted],
                color=COLORS[validity],
                linewidth=1.0,
                linestyle="--",
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
        style_axis(ax, col_idx)
    handles = [
        Line2D([0], [0], color=COLORS["valid"], linewidth=1.2, marker="o", markersize=3.2, label="valid arguments"),
        Line2D([0], [0], color=COLORS["invalid"], linewidth=1.2, marker="o", markersize=3.2, label="invalid arguments"),
        Line2D([0], [0], color="#666666", linewidth=1.0, linestyle="--", label="label-confidence-weighted"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=6, frameon=False)
    fig.suptitle("Cumulative drift under sustained pushback (cluster-bootstrap 95\\% bands)", fontsize=9, y=0.995)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95), w_pad=0.9)
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
    parser.add_argument("--curves", type=Path, default=DEFAULT_CURVES)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(f"curves={args.curves}")
    print(f"output-path={args.output_path}")
    curves = load_curves(args.curves)
    fig = make_figure(curves)
    save_figure(fig, args.output_path)
    plt.close("all")


if __name__ == "__main__":
    main()
