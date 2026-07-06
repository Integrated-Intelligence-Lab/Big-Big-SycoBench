import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / "Andres" / "ads_report_v2" / "outputs"
DEFAULT_SUMMARY = OUTPUT_DIR / "ads2_summary.csv"
DEFAULT_OUTPUT = OUTPUT_DIR / "ads2_quadrants"

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
MODEL_COLORS = {
    "gpt55": "#0173b2",
    "gpt55_prid": "#56b4e9",
    "gpt52_prid": "#029e73",
    "gpt5_prid": "#de8f05",
    "o3_prid": "#d55e00",
    "gpt41_prid": "#cc78bc",
    "o4mini": "#949494",
}
MODEL_MARKERS = {
    "gpt55": "o",
    "gpt55_prid": "D",
    "gpt52_prid": "s",
    "gpt5_prid": "^",
    "o3_prid": "v",
    "gpt41_prid": "P",
    "o4mini": "X",
}
COMPACT_LABELS = {"gpt55": "gpt-5.5", "o4mini": "o4-mini"}
NAME_LABEL_SIDE = {"gpt55": -1.0, "o4mini": 1.0}
GRAY = "#999999"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_points(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    out = {}
    for row in read_csv(path):
        if row["horizon"] == "t1" and row["variant"] in ("bt_weighted", "unweighted"):
            out[(row["model"], row["variant"])] = {
                "tpr": float(row["tpr"]),
                "fpr": float(row["fpr"]),
                "tpr_lo": float(row["tpr_ci_low"]),
                "tpr_hi": float(row["tpr_ci_high"]),
                "fpr_lo": float(row["fpr_ci_low"]),
                "fpr_hi": float(row["fpr_ci_high"]),
                "ads": float(row["ads"]),
            }
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


def make_figure(points: dict[tuple[str, str], dict[str, float]], models: tuple[str, ...]) -> plt.Figure:
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["text.usetex"] = True
    plt.rcParams["legend.title_fontsize"] = 7
    full = len(models) > 2
    fig, ax = plt.subplots(figsize=(7.08, 4.15) if full else (3.9, 4.05), dpi=300)
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.tick_params(axis="both", labelsize=6, length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel(r"$p_{\mathrm{inv}}$ = P(update $|$ invalid argument)", fontsize=7)
    ax.set_ylabel(r"$p_{\mathrm{val}}$ = P(update $|$ valid argument)", fontsize=7)

    ax.fill_between([0, 1], [0, 1], 0, color="#f2f2f2", zorder=0)
    ax.plot([0, 1], [0, 1], linestyle="--", color=GRAY, linewidth=0.8, zorder=1)
    ax.text(
        0.32,
        0.295,
        "indiscriminate (ADS = 0)",
        rotation=45,
        rotation_mode="anchor",
        ha="center",
        va="top",
        fontsize=5.5,
        color="#777777",
    )
    ax.plot([0, 0.5], [0.5, 1.0], linestyle=":", color=GRAY, linewidth=0.6, zorder=1)
    ax.text(
        0.055,
        0.505,
        "ADS = 50",
        rotation=45,
        rotation_mode="anchor",
        ha="left",
        va="bottom",
        fontsize=5,
        color="#777777",
    )
    ax.plot([0, 1], [0.25, 1.25], linestyle=":", color=GRAY, linewidth=0.6, zorder=1)
    ax.text(
        0.055,
        0.265,
        "ADS = 25",
        rotation=45,
        rotation_mode="anchor",
        ha="left",
        va="bottom",
        fontsize=5,
        color="#777777",
    )
    # ax.annotate(
    #     "",
    #     xy=(0.205, 0.525),
    #     xytext=(0.045, 0.365),
    #     arrowprops={"arrowstyle": "<->", "color": "#777777", "linewidth": 0.7, "shrinkA": 0, "shrinkB": 0},
    # )
    ax.text(
        0.385,
        0.615,
        "stubborn $\\leftrightarrow$ sycophantic",
        rotation=45,
        rotation_mode="anchor",
        ha="center",
        va="top",
        fontsize=5,
        color="#777777",
        linespacing=1.4,
    )

    corners = (
        (0.03, 0.975, "Discerning", "left", "top"),
        (0.975, 0.975, "Sycophantic", "right", "top"),
        (0.03, 0.025, "Stubborn", "left", "bottom"),
        (0.975, 0.025, "Anti-discerning", "right", "bottom"),
    )
    for x, y, label, ha, va in corners:
        ax.text(x, y, rf"\textbf{{{label}}}", ha=ha, va=va, fontsize=7)

    for model in models:
        color = MODEL_COLORS[model]
        marker = MODEL_MARKERS[model]
        weighted = points[(model, "bt_weighted")]
        unweighted = points[(model, "unweighted")]
        ax.plot(
            [weighted["fpr"], weighted["fpr"]],
            [weighted["fpr"], weighted["tpr"]],
            linestyle=":",
            color=color,
            linewidth=0.7,
            alpha=0.7,
            zorder=2,
        )
        ax.plot(
            [weighted["fpr_lo"], weighted["fpr_hi"]],
            [weighted["tpr"], weighted["tpr"]],
            color=color,
            linewidth=0.7,
            alpha=0.5,
            zorder=3,
        )
        ax.plot(
            [weighted["fpr"], weighted["fpr"]],
            [weighted["tpr_lo"], weighted["tpr_hi"]],
            color=color,
            linewidth=0.7,
            alpha=0.5,
            zorder=3,
        )
        ax.scatter([weighted["fpr"]], [weighted["tpr"]], s=28, color=color, marker=marker, zorder=5)
        ax.scatter(
            [unweighted["fpr"]],
            [unweighted["tpr"]],
            s=24,
            facecolors="none",
            edgecolors=color,
            marker=marker,
            linewidths=0.9,
            zorder=5,
        )
        if full:
            continue
        side = NAME_LABEL_SIDE[model]
        ax.text(
            unweighted["fpr"] + 0.035 if side > 0 else weighted["fpr"] - 0.035,
            weighted["tpr"] - 0.012,
            COMPACT_LABELS.get(model, MODEL_LABELS[model]) + "\n" + rf"ADS$_w$ = {weighted['ads']:.0f}",
            ha="left" if side > 0 else "right",
            va="top",
            fontsize=6,
            color=color,
            linespacing=1.25,
        )

    variant_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#555555", markersize=4.2, label="BT-weighted (headline)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor="#555555", markersize=4.0, label="unweighted"),
    ]
    if full:
        model_handles = [
            Line2D(
                [0],
                [0],
                marker=MODEL_MARKERS[model],
                color="none",
                markerfacecolor=MODEL_COLORS[model],
                markeredgecolor=MODEL_COLORS[model],
                markersize=4.4,
                label=f"{MODEL_LABELS[model]} ({points[(model, 'bt_weighted')]['ads']:.1f})",
            )
            for model in models
        ]
        fig.tight_layout(rect=(0, 0.07, 0.66, 1))
        fig.legend(
            handles=model_handles,
            title=r"ADS$_w$",
            loc="center left",
            bbox_to_anchor=(0.68, 0.54),
            fontsize=6,
            frameon=False,
        )
        fig.legend(handles=variant_handles, loc="lower left", bbox_to_anchor=(0.08, 0.0), ncol=2, fontsize=6, frameon=False)
    else:
        fig.legend(handles=variant_handles, loc="lower center", ncol=2, fontsize=6, frameon=False)
        fig.tight_layout(rect=(0, 0.055, 1, 1))
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
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models", nargs="+")
    args = parser.parse_args()
    print(f"summary={args.summary}")
    print(f"output-path={args.output_path}")
    points = load_points(args.summary)
    models = resolve_models(args.models)
    print(f"models={','.join(models)}")
    fig = make_figure(points, models)
    save_figure(fig, args.output_path)
    plt.close("all")


if __name__ == "__main__":
    main()
