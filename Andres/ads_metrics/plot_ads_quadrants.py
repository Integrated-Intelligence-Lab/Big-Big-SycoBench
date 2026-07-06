import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "Andres" / "ads_outputs"
DEFAULT_SUMMARY = OUTPUT_DIR / "ads_summary.csv"
DEFAULT_BOOTSTRAP = OUTPUT_DIR / "ads_bootstrap_summary.csv"
DEFAULT_TOYS = ROOT / "Andres" / "ads_inputs" / "illustration" / "shape_gallery_ads_scores.csv"
DEFAULT_OUTPUT = OUTPUT_DIR / "ads_quadrants"

MODEL_ORDER = ("gpt55", "o4mini")
HORIZON_ORDER = ("t1", "t2", "t3")
MODEL_LABELS = {"gpt55": "gpt-5.5", "o4mini": "o4-mini"}
MODEL_COLORS = {"gpt55": "#0173b2", "o4mini": "#de8f05"}
VALID_COLOR = "#029e73"
INVALID_COLOR = "#d55e00"
GRAY = "#8a8a8a"

QUADRANTS = (
    ("Discerning", "updates on valid,\nunmoved by invalid", 0.22, True, False, True),
    ("Credulous (sycophantic)", "updates on everything:\ncannot tell good from bad", 0.74, True, True, True),
    ("Stubborn", "updates on nothing:\ninsensitive, not sycophantic", 0.22, False, False, False),
    ("Anti-discerning", "updates on invalid only:\ndiscernment inverted", 0.78, False, True, False),
)

MINI_OFFSETS = (-0.04, 0.03, -0.01, 0.04, 0.0, -0.03)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_summary(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    out = {}
    for row in read_csv(path):
        out[(row["model"], row["horizon"])] = {
            "tpr": float(row["tpr"]),
            "fpr": float(row["fpr"]),
            "ads": float(row["ads"]),
        }
    return out


def load_intervals(path: Path) -> dict[tuple[str, str, str], tuple[float, float]]:
    out = {}
    for row in read_csv(path):
        out[(row["model"], row["horizon"], row["metric"])] = (float(row["ci_low"]), float(row["ci_high"]))
    return out


def load_toys(path: Path) -> list[dict[str, object]]:
    return [
        {"shape_id": row["shape_id"], "fpr": float(row["fpr"]), "tpr": float(row["tpr"]), "ads": float(row["ads"])}
        for row in read_csv(path)
    ]


def style_plane(ax: plt.Axes, iso_labels: bool, pad: float = 0.0) -> None:
    ax.set_xlim(-pad, 1 + pad)
    ax.set_ylim(-pad, 1 + pad)
    ax.set_aspect("equal")
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(axis="both", labelsize=6, length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel(r"$p_{\mathrm{inv}}$ = P(update $\mid$ invalid argument)", fontsize=7)
    ax.set_ylabel(r"$p_{\mathrm{val}}$ = P(update $\mid$ valid argument)", fontsize=7)
    ax.plot([0, 1], [0, 1], color="#555555", linewidth=0.8, linestyle="--", zorder=1)
    if iso_labels:
        for level in (0.25, 0.5, 0.75):
            ax.plot([0, 1 - level], [level, 1], color=GRAY, linewidth=0.45, linestyle=":", zorder=1)
            ax.text(0.02, level + 0.018, f"ADS {level*100:.0f}", ha="left", va="bottom", fontsize=5, color=GRAY)


def draw_mini(ax: plt.Axes, rect: tuple[float, float, float, float], invalid_up: bool, valid_up: bool) -> None:
    inset = ax.inset_axes(rect)
    inset.set_xlim(0, 1)
    inset.set_ylim(0, 1)
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_facecolor("white")
    for spine in inset.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#bbbbbb")
    inset.axhline(0.18, color="#666666", linewidth=0.45)
    for i, offset in enumerate(MINI_OFFSETS):
        x_invalid = 0.09 + i * 0.065
        x_valid = 0.59 + i * 0.065
        y_invalid = 0.72 + offset if invalid_up else 0.18 + 0.3 * offset
        y_valid = 0.72 + offset if valid_up else 0.18 + 0.3 * offset
        inset.scatter([x_invalid], [y_invalid], s=2.5, color=INVALID_COLOR, linewidths=0, zorder=3)
        inset.scatter([x_valid], [y_valid], s=2.5, color=VALID_COLOR, linewidths=0, zorder=3)


def panel_taxonomy(ax: plt.Axes) -> None:
    style_plane(ax, iso_labels=False)
    ax.set_title("The update-rate plane", fontsize=7.2, loc="left", pad=8)
    ax.fill_between([0, 1], [0, 1], [0, 0], color="#f0f0f0", zorder=0)
    ax.imshow(
        [[max(y - x, 0.0) for x in [i / 60 for i in range(61)]] for y in [i / 60 for i in range(61)]],
        origin="lower",
        extent=(0, 1, 0, 1),
        cmap="Greens",
        vmin=0,
        vmax=2.2,
        alpha=0.55,
        aspect="auto",
        zorder=0,
        interpolation="bilinear",
    )
    ax.text(0.515, 0.46, "indiscriminate: ADS = 0", fontsize=5, color="#555555", rotation=45, ha="center", va="center", rotation_mode="anchor", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.4})
    ax.text(0.815, 0.575, "ADS clipped to 0", fontsize=5, color=GRAY, rotation=45, ha="center", va="center", rotation_mode="anchor")
    ax.annotate(
        "",
        xy=(0.44, 0.97),
        xytext=(0.44, 0.44),
        arrowprops={"arrowstyle": "<|-|>", "color": "black", "lw": 0.7, "shrinkA": 0, "shrinkB": 0},
    )
    ax.text(0.468, 0.70, r"ADS = $100\,(p_{\mathrm{val}}-p_{\mathrm{inv}})$", fontsize=5.5, color="black", rotation=90, ha="left", va="center")
    for name, desc, x_center, top, invalid_up, valid_up in QUADRANTS:
        if top:
            name_y, desc_y = 0.995, 0.952
            rect = (x_center - 0.105, 0.645, 0.21, 0.22)
        else:
            name_y, desc_y = 0.375, 0.332
            rect = (x_center - 0.105, 0.025, 0.21, 0.22)
        ax.text(x_center, name_y, name, fontsize=6, fontweight="bold", ha="center", va="top", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.4})
        ax.text(x_center, desc_y, desc, fontsize=5, color="#444444", ha="center", va="top", linespacing=1.25, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.4})
        draw_mini(ax, rect, invalid_up, valid_up)


def panel_landscape(
    ax: plt.Axes,
    summary: dict[tuple[str, str], dict[str, float]],
    intervals: dict[tuple[str, str, str], tuple[float, float]],
    toys: list[dict[str, object]],
) -> None:
    style_plane(ax, iso_labels=True, pad=0.035)
    ax.set_title("Toy shapes and evaluated models", fontsize=7.2, loc="left", pad=8)
    for toy in toys:
        ax.scatter([toy["fpr"]], [toy["tpr"]], s=9, facecolors="none", edgecolors=GRAY, linewidths=0.6, zorder=2)
    toy_labels = (
        ("calibrated ideal", 0.028, 0.864, "left"),
        ("skeptic", 0.028, 0.682, "left"),
        ("stubborn", 0.028, 0.028, "left"),
        ("anti-discerning", 0.945, 0.136, "right"),
        ("pushover, floor and\ntrue sycophant", 0.985, 0.94, "right"),
        ("early take-off; linear", 0.41, 0.951, "center"),
        ("bump", 0.475, 0.565, "left"),
    )
    for label, lx, ly, align in toy_labels:
        ax.text(lx, ly, label, fontsize=5, color=GRAY, ha=align, va="center", linespacing=1.1)
    for model in MODEL_ORDER:
        color = MODEL_COLORS[model]
        xs = [summary[(model, horizon)]["fpr"] for horizon in HORIZON_ORDER]
        ys = [summary[(model, horizon)]["tpr"] for horizon in HORIZON_ORDER]
        fpr_lo, fpr_hi = intervals[(model, "t1", "fpr")]
        tpr_lo, tpr_hi = intervals[(model, "t1", "tpr")]
        ax.plot([fpr_lo, fpr_hi], [ys[0], ys[0]], color=color, linewidth=0.55, alpha=0.4, zorder=3)
        ax.plot([xs[0], xs[0]], [tpr_lo, tpr_hi], color=color, linewidth=0.55, alpha=0.4, zorder=3)
        ax.plot(xs, ys, color=color, linewidth=0.7, zorder=4)
        ax.scatter(xs, ys, s=[12, 12, 20], color=color, linewidths=0, zorder=5)
        ax.annotate(
            "",
            xy=(xs[2], ys[2]),
            xytext=(xs[1], ys[1]),
            arrowprops={"arrowstyle": "-|>", "color": color, "lw": 0.7, "shrinkA": 2, "shrinkB": 3},
        )
    ax.text(0.295, 0.908, "t1", fontsize=5, color=MODEL_COLORS["gpt55"], ha="left", va="center")
    ax.text(0.592, 0.885, "t1", fontsize=5, color=MODEL_COLORS["o4mini"], ha="left", va="center")
    ax.text(0.305, 0.775, "gpt-5.5\nADS 64$\\rightarrow$72", fontsize=5.5, color=MODEL_COLORS["gpt55"], ha="left", va="top", linespacing=1.2)
    ax.text(0.615, 0.80, "o4-mini\nADS 33$\\rightarrow$46", fontsize=5.5, color=MODEL_COLORS["o4mini"], ha="left", va="top", linespacing=1.2)


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
    summary: dict[tuple[str, str], dict[str, float]],
    intervals: dict[tuple[str, str, str], tuple[float, float]],
    toys: list[dict[str, object]],
) -> plt.Figure:
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["text.usetex"] = True
    fig, axes = plt.subplots(1, 2, figsize=(7.08, 3.55), dpi=300)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.87, bottom=0.13, wspace=0.28)
    panel_taxonomy(axes[0])
    panel_landscape(axes[1], summary, intervals, toys)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=MODEL_COLORS["gpt55"], markersize=3.5, label="gpt-5.5 (turns 1$\\rightarrow$3)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=MODEL_COLORS["o4mini"], markersize=3.5, label="o4-mini (turns 1$\\rightarrow$3)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor=GRAY, markersize=3.5, label="toy shape"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=6, frameon=False, bbox_to_anchor=(0.5, -0.015))
    for ax, letter in zip(axes, ("a", "b")):
        ax.text(-0.14, 1.06, letter, transform=ax.transAxes, fontsize=7, fontweight="bold", va="bottom", ha="left")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--bootstrap", type=Path, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--toys", type=Path, default=DEFAULT_TOYS)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = load_summary(args.summary)
    intervals = load_intervals(args.bootstrap)
    toys = load_toys(args.toys)
    fig = make_figure(summary, intervals, toys)
    save_figure(fig, args.output_path)
    plt.close("all")
    print(f"summary={args.summary}")
    print(f"bootstrap={args.bootstrap}")
    print(f"toys={args.toys}")
    print(f"output-path={args.output_path}")


if __name__ == "__main__":
    main()
