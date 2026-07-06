import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from Andres.ads_metrics.compute_ads import g_value, j_scores, metrics_for_records, sample_sd, tau_from_records


INPUT_DIR = ROOT / "Andres" / "ads_inputs" / "illustration"
DEFAULT_INPUT = INPUT_DIR / "shape_gallery_synthetic_points.csv"
DEFAULT_OUTPUT = INPUT_DIR / "shape_gallery_ads"
DEFAULT_SCORES = INPUT_DIR / "shape_gallery_ads_scores.csv"

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


def read_points(path: Path, alpha: float) -> dict[str, list[dict[str, object]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    scale = sample_sd([float(row["bt_rating"]) for row in rows if row["shape_id"] == SHAPE_ORDER[0]])
    out: dict[str, list[dict[str, object]]] = {shape_id: [] for shape_id in SHAPE_ORDER}
    for row in rows:
        bt = float(row["bt_rating"])
        x_value = bt / scale if scale > 0 else bt
        if bt > 0:
            signed_x = max(x_value, 0.0)
        elif bt < 0:
            signed_x = -max(-x_value, 0.0)
        else:
            signed_x = 0.0
        weight = abs(signed_x) ** alpha if signed_x != 0 else 0.0
        out[row["shape_id"]].append({
            "artifact": row["artefact_id"],
            "tier": row["tier"],
            "direction": row["direction"],
            "validity": row["validity"],
            "idx": row["idx"],
            "bt_rating": bt,
            "x": signed_x,
            "z": float(row["synthetic_shift"]),
            "weight": weight,
            "n": 1,
        })
    return out


def compute_scores(
    points: dict[str, list[dict[str, object]]],
    delta: float,
    eta_star: float,
    s_star: float,
) -> dict[str, dict[str, float]]:
    scores = {}
    for shape_id, records in points.items():
        samples = [(str(record["artifact"]), str(record["validity"]), float(record["z"])) for record in records]
        j = j_scores(samples, delta)
        tau = tau_from_records(records)
        metrics = metrics_for_records(records, tau, eta_star, s_star)
        scores[shape_id] = {
            "delta": delta,
            "tpr": j["tpr"],
            "fpr": j["fpr"],
            "ads": j["ads"],
            **{key: value for key, value in metrics.items() if key != "acsl"},
            "acsl_strict": metrics["acsl"],
            "tau": tau,
        }
    return scores


def write_scores(path: Path, scores: dict[str, dict[str, float]]) -> None:
    fields = [
        "shape_id",
        "shape_label",
        "delta",
        "tpr",
        "fpr",
        "ads",
        "tau",
        "s_minus",
        "i_minus",
        "eta_plus",
        "u_plus",
        "c_plus",
        "m_plus",
        "invalid_resistance",
        "sycophancy_score",
        "acsl_strict",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for shape_id in SHAPE_ORDER:
            row = {"shape_id": shape_id, "shape_label": DISPLAY_LABELS[shape_id]}
            row.update({key: f"{scores[shape_id][key]:.10g}" for key in fields[2:]})
            writer.writerow(row)


def fitted_y(bt: float, scale: float, eta: float, tau: float) -> float:
    x_value = max(bt / scale if scale > 0 else bt, 0.0)
    return eta * g_value(x_value, tau)


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


def make_figure(
    points: dict[str, list[dict[str, object]]],
    scores: dict[str, dict[str, float]],
) -> plt.Figure:
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["text.usetex"] = True
    plt.rcParams["legend.title_fontsize"] = 7
    scale = sample_sd([float(row["bt_rating"]) for row in points[SHAPE_ORDER[0]]])
    fig, axes = plt.subplots(3, 4, figsize=(7.08, 5.7), dpi=300, sharex=True, sharey=True)
    x_grid = [-2.5 + i * (4.0 / 240.0) for i in range(241)]
    for panel_idx, shape_id in enumerate(SHAPE_ORDER):
        row_idx, col_idx = divmod(panel_idx, 4)
        ax = axes[row_idx][col_idx]
        records = points[shape_id]
        score = scores[shape_id]
        color = PALETTE[panel_idx]
        ax.scatter(
            [float(record["bt_rating"]) for record in records],
            [float(record["z"]) for record in records],
            s=7,
            color=color,
            alpha=0.68,
            linewidths=0,
        )
        ax.plot(
            x_grid,
            [fitted_y(x, scale, score["eta_plus"], score["tau"]) for x in x_grid],
            color="black",
            linewidth=1.25,
        )
        ax.plot(
            [-2.5, 0],
            [score["s_minus"], score["s_minus"]],
            color="#d55e00",
            linewidth=0.9,
            linestyle="--",
        )
        ax.axhline(score["delta"], color="black", linewidth=0.7, linestyle=":")
        ax.set_title(DISPLAY_LABELS[shape_id], fontsize=7, fontweight="bold", pad=4)
        ax.text(
            0.03,
            0.96,
            (
                rf"ADS={score['ads']:.0f}" + "\n"
                rf"$p_{{\mathrm{{val}}}}={score['tpr']:.2f}$" + "\n"
                rf"$p_{{\mathrm{{inv}}}}={score['fpr']:.2f}$"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=5.5,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.88},
        )
        style_axis(ax, row_idx, col_idx)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#7f7f7f", markersize=3.5, label="synthetic data"),
        Line2D([0], [0], color="black", linewidth=0.7, linestyle=":", label=r"update threshold $\delta$"),
        Line2D([0], [0], color="black", linewidth=1.25, label=r"valid uptake fit (diagnostic)"),
        Line2D([0], [0], color="#d55e00", linewidth=0.9, linestyle="--", label=r"invalid compliance $S_-$ (diagnostic)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6, frameon=False)
    fig.suptitle(
        "Shape gallery: update rates and ADS on synthetic shifts",
        fontsize=9,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.975), h_pad=1.0, w_pad=0.7)
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
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scores-path", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--delta", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--eta-star", type=float, default=0.5)
    parser.add_argument("--s-star", type=float, default=0.5)
    args = parser.parse_args()
    points = read_points(args.input_file, args.alpha)
    scores = compute_scores(points, args.delta, args.eta_star, args.s_star)
    write_scores(args.scores_path, scores)
    fig = make_figure(points, scores)
    save_figure(fig, args.output_path)
    plt.close("all")
    print(f"input-file={args.input_file}")
    print(f"scores-path={args.scores_path}")
    print(f"output-path={args.output_path}")


if __name__ == "__main__":
    main()
