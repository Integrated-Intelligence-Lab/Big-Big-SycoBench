import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "Andres" / "ads_outputs" / "acsl_hinge_alternative_schematic"

COLORS = {
    "valid": "#029e73",
    "invalid": "#d55e00",
    "sigmoid": "#7f7f7f",
    "hinge": "#000000",
    "blue": "#0173b2",
    "light_blue": "#56b4e9",
}


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def free_sigmoid(x_value: float) -> float:
    return 0.34 + 1.55 * sigmoid(2.25 * (x_value - 0.25))


def hinge_response(x_value: float, amplitude: float, tau: float) -> float:
    return amplitude * math.tanh(max(x_value, 0.0) / tau)


def score_without_monotonicity(s_minus: float, s_star: float) -> float:
    return 100.0 * (1.0 - math.exp(-((s_minus / s_star) ** 2)))


def score_with_monotonicity(s_minus: float, s_star: float, m_plus: float) -> float:
    return 100.0 * (1.0 - m_plus * math.exp(-((s_minus / s_star) ** 2)))


def set_axis_style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=6, length=0)
    ax.grid(True, color="#d9d9d9", linewidth=0.35, alpha=0.65)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.16,
        1.08,
        rf"\textbf{{{label}}}",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        clip_on=False,
    )


def make_figure() -> plt.Figure:
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["text.usetex"] = True
    plt.rcParams["legend.title_fontsize"] = 7

    fig, axes = plt.subplot_mosaic(
        [["A", "B", "C"]],
        figsize=(7.08, 2.45),
        dpi=300,
    )

    x_grid = [-2.4 + idx * (4.7 / 260.0) for idx in range(261)]
    invalid_x = [-2.05, -1.55, -1.05, -0.62, -0.25]
    invalid_z = [0.34, 0.47, 0.28, -0.18, 0.53]
    valid_x = [0.18, 0.44, 0.78, 1.20, 1.82]
    valid_z = [0.18, 0.48, 0.86, 1.24, 1.53]
    s_minus = math.sqrt(sum(max(value, 0.0) ** 2 for value in invalid_z) / len(invalid_z))

    ax = axes["A"]
    ax.axvspan(-2.4, 0.0, color=COLORS["invalid"], alpha=0.08, linewidth=0)
    ax.scatter(invalid_x, invalid_z, s=18, color=COLORS["invalid"], alpha=0.82, linewidths=0)
    ax.scatter(valid_x, valid_z, s=18, color=COLORS["valid"], alpha=0.82, linewidths=0)
    ax.plot(x_grid, [free_sigmoid(x_value) for x_value in x_grid], color=COLORS["sigmoid"], linewidth=1.25)
    ax.plot([-2.4, 0.0], [0.34, 0.34], color=COLORS["sigmoid"], linewidth=0.8, linestyle=":")
    ax.axhline(0, color="#666666", linewidth=0.45)
    ax.axvline(0, color="#666666", linewidth=0.45, linestyle=":")
    ax.text(-2.27, 1.78, r"\textbf{Free sigmoid}", fontsize=7, ha="left", va="top")
    ax.text(-2.27, 0.52, r"left tail absorbs\newline invalid movement", fontsize=5.5, ha="left", va="bottom")
    ax.set_xlim(-2.4, 2.3)
    ax.set_ylim(-0.45, 2.05)
    ax.set_xlabel(r"$x$ = signed BT validity", fontsize=7)
    ax.set_ylabel(r"$z$ = directional shift", fontsize=7)
    set_axis_style(ax)
    add_panel_label(ax, "a")

    ax = axes["B"]
    ax.axvspan(-2.4, 0.0, color=COLORS["invalid"], alpha=0.08, linewidth=0)
    ax.scatter(invalid_x, invalid_z, s=18, color=COLORS["invalid"], alpha=0.82, linewidths=0)
    ax.scatter(valid_x, valid_z, s=18, color=COLORS["valid"], alpha=0.82, linewidths=0)
    ax.plot(x_grid, [hinge_response(x_value, 1.55, 0.78) for x_value in x_grid], color=COLORS["hinge"], linewidth=1.25)
    for x_value, z_value in zip(invalid_x, invalid_z, strict=True):
        if z_value > 0:
            ax.plot([x_value, x_value], [0.0, z_value], color=COLORS["invalid"], linewidth=0.9, alpha=0.85)
    ax.plot([-2.4, 0.0], [s_minus, s_minus], color=COLORS["invalid"], linewidth=0.9, linestyle="--")
    ax.axhline(0, color="#666666", linewidth=0.45)
    ax.axvline(0, color="#666666", linewidth=0.45, linestyle=":")
    ax.text(-2.27, 1.78, r"\textbf{Rectified hinge}", fontsize=7, ha="left", va="top")
    ax.text(-2.27, 0.70, rf"$S_-={s_minus:.2f}$", fontsize=6, ha="left", va="bottom", color=COLORS["invalid"])
    ax.text(-1.05, 0.06, r"one-sided $[z]_+$ penalty", fontsize=5.5, ha="left", va="bottom", color=COLORS["invalid"])
    ax.set_xlim(-2.4, 2.3)
    ax.set_ylim(-0.45, 2.05)
    ax.set_xlabel(r"$x$ = signed BT validity", fontsize=7)
    ax.set_yticklabels([])
    set_axis_style(ax)
    add_panel_label(ax, "b")

    ax = axes["C"]
    s_grid = [idx * (1.45 / 220.0) for idx in range(221)]
    ax.plot(
        s_grid,
        [score_without_monotonicity(value, 0.5) for value in s_grid],
        color=COLORS["blue"],
        linewidth=1.25,
        label=r"no $M_+$ gate",
    )
    ax.plot(
        s_grid,
        [score_with_monotonicity(value, 0.5, 0.5) for value in s_grid],
        color=COLORS["light_blue"],
        linewidth=1.25,
        label=r"$M_+=0.5$",
    )
    ax.plot(
        s_grid,
        [score_with_monotonicity(value, 0.5, 0.0) for value in s_grid],
        color=COLORS["invalid"],
        linewidth=1.25,
        label=r"$M_+=0$",
    )
    ax.set_xlim(0.0, 1.45)
    ax.set_ylim(-2.0, 104.0)
    ax.set_xlabel(r"invalid compliance $S_-$", fontsize=7)
    ax.set_ylabel("loss", fontsize=7)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels([r"0", r"25", r"50", r"75", r"100"])
    ax.text(0.07, 97.0, r"\textbf{Score response}", fontsize=7, ha="left", va="top")
    ax.legend(loc="lower right", frameon=False, fontsize=6, handlelength=1.8)
    set_axis_style(ax)
    add_panel_label(ax, "c")

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["invalid"], markersize=4, label="invalid argument"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["valid"], markersize=4, label="valid argument"),
        Line2D([0], [0], color=COLORS["sigmoid"], linewidth=1.25, label="free sigmoid"),
        Line2D([0], [0], color=COLORS["hinge"], linewidth=1.25, label="hinge response"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6, frameon=False)
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 1.0), w_pad=1.0)
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
            for artist in ax.texts:
                artist.set_usetex(False)
        fig.savefig(output_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
        fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(f"output-path={args.output_path}")
    fig = make_figure()
    save_figure(fig, args.output_path)
    plt.close("all")


if __name__ == "__main__":
    main()
