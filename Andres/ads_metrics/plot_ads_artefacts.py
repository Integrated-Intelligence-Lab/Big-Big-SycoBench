import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "Andres" / "ads_outputs"
DEFAULT_ARTEFACT_RATES = OUTPUT_DIR / "ads_artefact_rates.csv"
DEFAULT_SUMMARY = OUTPUT_DIR / "ads_summary.csv"
DEFAULT_POINTS = OUTPUT_DIR / "ads_argument_points.csv"
DEFAULT_DOSE = OUTPUT_DIR / "ads_dose_response.csv"
DEFAULT_OUTPUT = OUTPUT_DIR / "ads_artefact_landscape"

MODEL_ORDER = ("gpt55", "o4mini")
MODEL_LABELS = {"gpt55": "gpt-5.5", "o4mini": "o4-mini"}
MODEL_COLORS = {"gpt55": "#0173b2", "o4mini": "#de8f05"}
VALID_COLOR = "#029e73"
INVALID_COLOR = "#d55e00"
GRAY = "#8a8a8a"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_artefact_rates(path: Path) -> dict[str, list[tuple[float, float]]]:
    out = {model: [] for model in MODEL_ORDER}
    for row in read_csv(path):
        if row["horizon"] == "t1" and row["model"] in out:
            out[row["model"]].append((float(row["fpr"]), float(row["tpr"])))
    return out

def load_summary(path: Path) -> dict[str, tuple[float, float]]:
    out = {}
    for row in read_csv(path):
        if row["horizon"] == "t1":
            out[row["model"]] = (float(row["fpr"]), float(row["tpr"]))
    return out


def load_points(path: Path) -> dict[str, dict[str, list[tuple[float, float]]]]:
    out = {model: {"valid": [], "invalid": []} for model in MODEL_ORDER}
    for row in read_csv(path):
        if row["horizon"] == "t1" and row["model"] in out and row["validity"] in ("valid", "invalid"):
            out[row["model"]][row["validity"]].append((float(row["x"]), float(row["update_rate"])))
    return out


def load_dose(path: Path) -> dict[tuple[str, str], float]:
    return {
        (row["model"], row["pool"]): float(row["spearman_rho"])
        for row in read_csv(path)
    }


def panel_artefacts(ax: plt.Axes, rates: dict[str, list[tuple[float, float]]], summary: dict[str, tuple[float, float]]) -> None:
    pad = 0.035
    ax.set_xlim(-pad, 1 + pad)
    ax.set_ylim(-pad, 1 + pad)
    ax.set_aspect("equal")
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(axis="both", labelsize=6, length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel(r"$p_{\mathrm{inv}}$ per artefact", fontsize=7)
    ax.set_ylabel(r"$p_{\mathrm{val}}$ per artefact", fontsize=7)
    ax.set_title("Artefact-level rates (t1)", fontsize=7.2, loc="left", pad=6)
    ax.fill_between([0, 1], [0, 1], [0, 0], color="#f0f0f0", zorder=0)
    ax.plot([0, 1], [0, 1], color="#555555", linewidth=0.8, linestyle="--", zorder=1)
    for level in (0.25, 0.5, 0.75):
        ax.plot([0, 1 - level], [level, 1], color=GRAY, linewidth=0.45, linestyle=":", zorder=1)
        ax.text(0.02, level + 0.018, f"ADS {level*100:.0f}", ha="left", va="bottom", fontsize=5, color=GRAY)
    for model in MODEL_ORDER:
        color = MODEL_COLORS[model]
        xs = [fpr for fpr, _ in rates[model]]
        ys = [tpr for _, tpr in rates[model]]
        ax.scatter(xs, ys, s=11, facecolors="none", edgecolors=color, linewidths=0.65, alpha=0.85, zorder=3)
        agg_x, agg_y = summary[model]
        ax.scatter([agg_x], [agg_y], s=30, color=color, edgecolors="white", linewidths=0.6, zorder=5)
    ax.text(0.62, 0.475, "below diagonal:\nanti-discerning", fontsize=5, color=GRAY, ha="left", va="center", linespacing=1.2)


def panel_dose(ax: plt.Axes, model: str, points: dict[str, list[tuple[float, float]]], dose: dict[tuple[str, str], float], show_ylabel: bool) -> None:
    ax.set_xlim(-2.55, 1.85)
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xticks([-2, -1, 0, 1])
    ax.tick_params(axis="both", labelsize=6, length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("signed standardized BT ($x$)", fontsize=7)
    if show_ylabel:
        ax.set_ylabel("per-argument update rate", fontsize=7)
    ax.set_title(MODEL_LABELS[model], fontsize=7.2, loc="left", pad=6, color=MODEL_COLORS[model])
    ax.axvline(0, color="#bbbbbb", linewidth=0.5, linestyle=":", zorder=1)
    for validity, color in (("invalid", INVALID_COLOR), ("valid", VALID_COLOR)):
        xs = [x for x, _ in points[validity]]
        ys = [rate for _, rate in points[validity]]
        ax.scatter(xs, ys, s=6.5, color=color, linewidths=0, alpha=0.7, zorder=3)
    rho_inv = dose[(model, "invalid")]
    rho_val = dose[(model, "valid")]
    ax.text(0.63, 1.02, rf"$\rho_{{\mathrm{{inv}}}} = {rho_inv:+.2f}$", transform=ax.transAxes, fontsize=5.5, color=INVALID_COLOR, ha="right", va="bottom")
    ax.text(1.0, 1.02, rf"$\rho_{{\mathrm{{val}}}} = {rho_val:+.2f}$", transform=ax.transAxes, fontsize=5.5, color=VALID_COLOR, ha="right", va="bottom")


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


def make_figure(
    rates: dict[str, list[tuple[float, float]]],
    summary: dict[str, tuple[float, float]],
    points: dict[str, dict[str, list[tuple[float, float]]]],
    dose: dict[tuple[str, str], float],
) -> plt.Figure:
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["text.usetex"] = True
    fig, axes = plt.subplots(1, 3, figsize=(7.08, 2.55), dpi=300)
    fig.subplots_adjust(left=0.065, right=0.99, top=0.86, bottom=0.24, wspace=0.36)
    panel_artefacts(axes[0], rates, summary)
    panel_dose(axes[1], "gpt55", points["gpt55"], dose, show_ylabel=True)
    panel_dose(axes[2], "o4mini", points["o4mini"], dose, show_ylabel=False)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor=MODEL_COLORS["gpt55"], markersize=3.5, label="gpt-5.5 artefact"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor=MODEL_COLORS["o4mini"], markersize=3.5, label="o4-mini artefact"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=VALID_COLOR, markersize=3.5, label="valid argument"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=INVALID_COLOR, markersize=3.5, label="invalid argument"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6, frameon=False, bbox_to_anchor=(0.5, -0.02))
    for ax, letter in zip(axes, ("a", "b", "c")):
        ax.text(-0.16, 1.05, letter, transform=ax.transAxes, fontsize=7, fontweight="bold", va="bottom", ha="left")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artefact-rates", type=Path, default=DEFAULT_ARTEFACT_RATES)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--argument-points", type=Path, default=DEFAULT_POINTS)
    parser.add_argument("--dose-response", type=Path, default=DEFAULT_DOSE)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rates = load_artefact_rates(args.artefact_rates)
    summary = load_summary(args.summary)
    points = load_points(args.argument_points)
    dose = load_dose(args.dose_response)
    fig = make_figure(rates, summary, points, dose)
    save_figure(fig, args.output_path)
    plt.close("all")
    print(f"artefact-rates={args.artefact_rates}")
    print(f"summary={args.summary}")
    print(f"argument-points={args.argument_points}")
    print(f"dose-response={args.dose_response}")
    print(f"output-path={args.output_path}")


if __name__ == "__main__":
    main()
