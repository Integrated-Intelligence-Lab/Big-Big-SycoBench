import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[3]
INPUT_DIR = ROOT / "Andres" / "ads_inputs" / "illustration"
DEFAULT_INPUT = INPUT_DIR / "shape_gallery_synthetic_points.csv"
DEFAULT_OUTPUT = INPUT_DIR / "shape_gallery_data_only"

PALETTE = [
    "#0173b2",
    "#de8f05",
    "#029e73",
    "#d55e00",
    "#56b4e9",
    "#8c564b",
    "#cc78bc",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#4c4c7f",
    "#e5ae38",
]

SHAPE_ORDER = [
    "calibrated_ideal",
    "sycophant_floor",
    "sycophant_early_takeoff",
    "skeptic_late_takeoff",
    "stubborn",
    "pushover_flat_high",
    "super_sensitive",
    "linear",
    "true_sycophant_valid_invalid",
    "contrarian_decreasing",
    "bump_mid_quality",
    "calibrated_noisy",
]

DISPLAY_LABELS = {
    "calibrated_ideal": "calibrated (ideal)",
    "sycophant_floor": r"sycophant: floor $>$ 0",
    "sycophant_early_takeoff": "sycophant: early take-off",
    "skeptic_late_takeoff": "skeptic: late take-off",
    "stubborn": r"stubborn ($z \sim 0$)",
    "pushover_flat_high": "pushover (flat-high)",
    "super_sensitive": "super-sensitive",
    "linear": "linear",
    "true_sycophant_valid_invalid": "true sycophant (valid=invalid)",
    "contrarian_decreasing": "anti-discerning (decreasing)",
    "bump_mid_quality": "bump (mid-quality)",
    "calibrated_noisy": "calibrated but noisy",
}


def load_points(path: Path) -> dict[str, list[dict[str, float | str]]]:
    points: dict[str, list[dict[str, float | str]]] = {shape_id: [] for shape_id in SHAPE_ORDER}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            points[row["shape_id"]].append({
                "shape_label": row["shape_label"],
                "bt_rating": float(row["bt_rating"]),
                "synthetic_shift": float(row["synthetic_shift"]),
                "validity": row["validity"],
            })
    return points


def style_axis(ax: plt.Axes, row_idx: int, col_idx: int) -> None:
    ax.set_xlim(-2.6, 1.6)
    ax.set_ylim(-5.2, 7.9)
    ax.grid(True, color="#d9d9d9", linewidth=0.35, alpha=0.7)
    ax.axhline(0, color="#666666", linewidth=0.45)
    ax.axvline(0, color="#666666", linewidth=0.45, linestyle=":")
    ax.tick_params(axis="both", labelsize=6, length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if row_idx == 2:
        ax.set_xlabel("q = argument BT", fontsize=7)
    else:
        ax.set_xlabel("")
        ax.set_xticklabels([])
    if col_idx == 0:
        ax.set_ylabel("z = synthetic shift", fontsize=7)
    else:
        ax.set_ylabel("")
        ax.set_yticklabels([])


def make_figure(points: dict[str, list[dict[str, float | str]]]) -> plt.Figure:
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["text.usetex"] = True
    plt.rcParams["legend.title_fontsize"] = 7
    fig, axes = plt.subplots(3, 4, figsize=(7.08, 5.4), dpi=300, sharex=True, sharey=True)
    for panel_idx, shape_id in enumerate(SHAPE_ORDER):
        row_idx, col_idx = divmod(panel_idx, 4)
        ax = axes[row_idx][col_idx]
        rows = points[shape_id]
        x_values = [float(row["bt_rating"]) for row in rows]
        y_values = [float(row["synthetic_shift"]) for row in rows]
        ax.scatter(
            x_values,
            y_values,
            s=7,
            color=PALETTE[panel_idx],
            alpha=0.72,
            linewidths=0,
        )
        title = DISPLAY_LABELS[shape_id]
        ax.set_title(title, fontsize=7, fontweight="bold", pad=4)
        style_axis(ax, row_idx, col_idx)
    fig.suptitle(
        "Shape gallery: real BT values, synthetic shifts (data only)",
        fontsize=9,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975), h_pad=1.0, w_pad=0.7)
    return fig


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
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
        if fig._suptitle is not None:
            fig._suptitle.set_usetex(False)
        fig.savefig(output_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
        fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(f"input-file={args.input_file}")
    print(f"output-path={args.output_path}")
    points = load_points(args.input_file)
    fig = make_figure(points)
    save_figure(fig, args.output_path)
    plt.close("all")


if __name__ == "__main__":
    main()
