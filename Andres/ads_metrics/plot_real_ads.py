import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "Andres" / "ads_outputs"
DEFAULT_POINTS = OUTPUT_DIR / "ads_argument_points.csv"
DEFAULT_SUMMARY = OUTPUT_DIR / "ads_summary.csv"
DEFAULT_OUTPUT = OUTPUT_DIR / "ads_real_points"

MODEL_ORDER = ("gpt55", "o4mini")
HORIZON_ORDER = ("t1", "t2", "t3")
MODEL_LABELS = {"gpt55": "gpt-5.5", "o4mini": "o4-mini"}
HORIZON_LABELS = {"t1": "turn 1", "t2": "turn 2", "t3": "turn 3"}
COLORS = {"valid": "#029e73", "invalid": "#d55e00"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_points(path: Path) -> dict[tuple[str, str], list[dict[str, object]]]:
    out: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in read_csv(path):
        key = (row["model"], row["horizon"])
        out.setdefault(key, []).append({
            "x": float(row["x"]),
            "shift": float(row["shift_points"]),
            "validity": row["validity"],
        })
    return out


def load_summary(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    out = {}
    for row in read_csv(path):
        key = (row["model"], row["horizon"])
        out[key] = {
            "delta": float(row["delta"]),
            "tpr": float(row["tpr"]),
            "fpr": float(row["fpr"]),
            "ads": float(row["ads"]),
        }
    return out


def style_axis(ax: plt.Axes, row_idx: int, col_idx: int) -> None:
    ax.set_xlim(-2.5, 1.8)
    ax.set_ylim(-12, 70)
    ax.grid(True, color="#d9d9d9", linewidth=0.35, alpha=0.7)
    ax.axhline(0, color="#666666", linewidth=0.45)
    ax.axvline(0, color="#666666", linewidth=0.45, linestyle=":")
    ax.tick_params(axis="both", labelsize=6, length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if row_idx == 1:
        ax.set_xlabel(r"$x$ = signed standardized BT", fontsize=7)
    else:
        ax.set_xlabel("")
        ax.set_xticklabels([])
    if col_idx == 0:
        ax.set_ylabel("mean directional shift (points)", fontsize=7)
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])


def make_figure(
    points: dict[tuple[str, str], list[dict[str, object]]],
    summary: dict[tuple[str, str], dict[str, float]],
) -> plt.Figure:
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["text.usetex"] = True
    plt.rcParams["legend.title_fontsize"] = 7
    fig, axes = plt.subplots(2, 3, figsize=(7.08, 4.2), dpi=300, sharex=True, sharey=True)
    for row_idx, model in enumerate(MODEL_ORDER):
        for col_idx, horizon in enumerate(HORIZON_ORDER):
            ax = axes[row_idx][col_idx]
            key = (model, horizon)
            rows = points[key]
            scores = summary[key]
            for validity in ("invalid", "valid"):
                subset = [row for row in rows if row["validity"] == validity]
                ax.scatter(
                    [float(row["x"]) for row in subset],
                    [float(row["shift"]) for row in subset],
                    s=12,
                    color=COLORS[validity],
                    alpha=0.74,
                    linewidths=0,
                    label=validity,
                )
            ax.axhline(scores["delta"], color="black", linewidth=0.9, linestyle="--")
            title = f"{MODEL_LABELS[model]}, {HORIZON_LABELS[horizon]}"
            ax.set_title(title, fontsize=7, fontweight="bold", pad=4)
            ax.text(
                0.03,
                0.96,
                (
                    rf"ADS={scores['ads']:.0f}" + "\n"
                    rf"$p_{{\mathrm{{val}}}}={scores['tpr']:.2f}$" + "\n"
                    rf"$p_{{\mathrm{{inv}}}}={scores['fpr']:.2f}$"
                ),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=5.5,
                bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.88},
            )
            style_axis(ax, row_idx, col_idx)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["valid"], markersize=3.5, label="valid argument"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["invalid"], markersize=3.5, label="invalid argument"),
        Line2D([0], [0], color="black", linewidth=0.9, linestyle="--", label=r"update threshold $\delta$"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=6, frameon=False)
    fig.suptitle("Real SycoBench trajectories: update rates and ADS by model and horizon", fontsize=9, y=0.995)
    fig.tight_layout(rect=(0, 0.055, 1, 0.965), h_pad=1.0, w_pad=0.7)
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
    points = load_points(args.points)
    summary = load_summary(args.summary)
    fig = make_figure(points, summary)
    save_figure(fig, args.output_path)
    plt.close("all")
    print(f"points={args.points}")
    print(f"summary={args.summary}")
    print(f"output-path={args.output_path}")


if __name__ == "__main__":
    main()
