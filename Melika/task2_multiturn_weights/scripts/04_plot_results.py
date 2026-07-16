"""Plot Task 2 ADS-style results."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from common import FIGURES_DIR, RESULTS_DIR, ensure_dirs


SUMMARY = RESULTS_DIR / "multiturn_ads_by_method.csv"

METHOD_ORDER = ["unweighted", "lead", "mean", "median", "max", "min", "sum"]
METHOD_LABEL = {
    "unweighted": "unweighted",
    "lead": "lead",
    "mean": "mean",
    "median": "median",
    "max": "max",
    "min": "min",
    "sum": "sum",
}
METHOD_MARKER = {
    "unweighted": "o",
    "lead": "s",
    "mean": "^",
    "median": "X",
    "max": "D",
    "min": "v",
    "sum": "P",
}
METHOD_COLOR = {
    "unweighted": "#333333",
    "lead": "#1f77b4",
    "mean": "#ff7f0e",
    "median": "#8c564b",
    "max": "#2ca02c",
    "min": "#d62728",
    "sum": "#9467bd",
}
MODEL_COLOR = {
    "gpt-5.5": "#0173b2",
    "o4-mini": "#de8f05",
}
TURN_MARKER = {
    1: "^",
    2: "s",
    3: "o",
}


def focus_methods(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the core comparison: official hinged variants plus unweighted."""

    hinged = df[df["weight_family"].eq("ads_hinged")].copy()
    unweighted = df[
        df["weight_family"].eq("none") & df["aggregation"].eq("unweighted")
    ].copy()
    focus = pd.concat([unweighted, hinged], ignore_index=True)
    focus = focus[focus["aggregation"].isin(METHOD_ORDER)].copy()
    focus["aggregation"] = pd.Categorical(
        focus["aggregation"], categories=METHOD_ORDER, ordered=True
    )
    return focus.sort_values(["model", "aggregation", "turn"])


def plot_ads_lines(focus: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    for ax, model in zip(axes, sorted(focus["model"].unique())):
        sub = focus[focus["model"] == model]
        for method in METHOD_ORDER:
            method_df = sub[sub["aggregation"] == method].sort_values("turn")
            if method_df.empty:
                continue
            ax.plot(
                method_df["turn"],
                method_df["ads"],
                marker=METHOD_MARKER[method],
                color=METHOD_COLOR[method],
                linewidth=1.8,
                markersize=6,
                label=METHOD_LABEL[method],
            )

        ax.set_title(model)
        ax.set_xticks([1, 2, 3])
        ax.set_xlabel("turn")
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(bottom=0)

    axes[0].set_ylabel("ADS")
    axes[1].legend(title="aggregation", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("Task 2: ADS Across Turns by Weight Aggregation")
    fig.tight_layout()
    out = FIGURES_DIR / "ads_by_method.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print(f"Wrote {out}")


def plot_ads_plane(focus: pd.DataFrame) -> None:
    models = sorted(focus["model"].unique())
    fig, axes = plt.subplots(
        len(models),
        3,
        figsize=(12, 6.4),
        squeeze=False,
    )

    for row_i, model in enumerate(models):
        for col_i, turn in enumerate([1, 2, 3]):
            ax = axes[row_i][col_i]
            sub = focus[(focus["model"] == model) & (focus["turn"] == turn)]

            for method in METHOD_ORDER:
                point = sub[sub["aggregation"] == method]
                if point.empty:
                    continue
                ax.scatter(
                    point["p_inval"],
                    point["p_val"],
                    s=45,
                    marker=METHOD_MARKER[method],
                    color=METHOD_COLOR[method],
                    edgecolor="white",
                    linewidth=0.6,
                    label=METHOD_LABEL[method],
                    zorder=3,
                )

            xmin = max(0, sub["p_inval"].min() - 0.035)
            xmax = min(1, sub["p_inval"].max() + 0.035)
            ymin = max(0, sub["p_val"].min() - 0.025)
            ymax = min(1.02, sub["p_val"].max() + 0.025)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1)
            ax.grid(alpha=0.25)
            ax.set_title(f"{model}, turn {turn}", fontsize=10)
            ax.set_xlabel("p_inval")
            if col_i == 0:
                ax.set_ylabel("p_val")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="aggregation",
        loc="lower center",
        ncol=7,
        frameon=False,
    )
    fig.suptitle("ADS Plane, Zoomed by Model and Turn")
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    out = FIGURES_DIR / "pval_pinval_plane.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print(f"Wrote {out}")


def style_plane(ax: plt.Axes, zoom: bool) -> None:

    if not zoom:
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        for level in (0.25, 0.50, 0.75):
            ax.plot(
                [0, 1 - level],
                [level, 1],
                color="#8a8a8a",
                linewidth=0.55,
                linestyle=":",
                zorder=1,
            )
            ax.text(
                0.02,
                level + 0.025,
                f"ADS {level * 100:.0f}",
                ha="left",
                va="bottom",
                fontsize=7,
                color="#777777",
            )

    ax.plot([0, 1], [0, 1], color="#555555", linewidth=0.9, linestyle="--", zorder=1)
    ax.grid(True, color="#d9d9d9", linewidth=0.45, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=8, length=0)
    ax.set_xlabel("p_inval = P(update | invalid argument)", fontsize=9)
    ax.set_ylabel("p_val = P(update | valid argument)", fontsize=9)


def plot_style_all_turns(focus: pd.DataFrame) -> None:
    """Show turn-1 -> turn-2 -> turn-3 movement in ADS-plane style."""

    all_turns = focus[focus["turn"].isin([1, 2, 3])].copy()
    models = sorted(all_turns["model"].unique())
    fig, axes = plt.subplots(2, len(models), figsize=(12, 8), squeeze=False)

    for col_i, model in enumerate(models):
        sub = all_turns[all_turns["model"] == model]

        full_ax = axes[0][col_i]
        style_plane(full_ax, zoom=False)
        full_ax.set_title(f"{model}: full ADS plane", fontsize=11, fontweight="bold")

        zoom_ax = axes[1][col_i]
        style_plane(zoom_ax, zoom=True)
        zoom_ax.set_title(f"{model}: zoom on aggregation methods", fontsize=11, fontweight="bold")

        x_pad = 0.035
        y_pad = 0.025
        zoom_ax.set_xlim(max(0, sub["p_inval"].min() - x_pad), min(1, sub["p_inval"].max() + x_pad))
        zoom_ax.set_ylim(max(0, sub["p_val"].min() - y_pad), min(1.02, sub["p_val"].max() + y_pad))

        for ax in (full_ax, zoom_ax):
            for method in METHOD_ORDER:
                method_df = sub[sub["aggregation"] == method].sort_values("turn")
                if method_df.empty:
                    continue
                x = method_df["p_inval"].tolist()
                y = method_df["p_val"].tolist()
                ax.plot(
                    x,
                    y,
                    color=METHOD_COLOR[method],
                    linewidth=1.2,
                    alpha=0.9,
                    zorder=3,
                )
                ax.scatter(
                    x,
                    y,
                    s=52,
                    marker=METHOD_MARKER[method],
                    color=METHOD_COLOR[method],
                    edgecolor="white",
                    linewidth=0.7,
                    label=METHOD_LABEL[method],
                    zorder=4,
                )
                for start, end in zip(range(len(x) - 1), range(1, len(x))):
                    ax.annotate(
                        "",
                        xy=(x[end], y[end]),
                        xytext=(x[start], y[start]),
                        arrowprops={
                            "arrowstyle": "-|>",
                            "color": METHOD_COLOR[method],
                            "lw": 1.15,
                            "mutation_scale": 10,
                            "shrinkA": 7,
                            "shrinkB": 7,
                        },
                        zorder=5,
                    )

            for turn, label_offset in ((1, (4, 4)), (2, (4, -9)), (3, (4, 4))):
                point = sub[
                    (sub["turn"] == turn)
                    & (sub["aggregation"] == "unweighted")
                ]
                if not point.empty:
                    row = point.iloc[0]
                    ax.annotate(
                        f"t{turn}",
                        (row["p_inval"], row["p_val"]),
                        xytext=label_offset,
                        textcoords="offset points",
                        fontsize=8,
                        color="#222222",
                        zorder=6,
                    )

    handles = [
        plt.Line2D(
            [0],
            [0],
            color=METHOD_COLOR[method],
            marker=METHOD_MARKER[method],
            linewidth=1.4,
            markersize=6,
            label=METHOD_LABEL[method],
        )
        for method in METHOD_ORDER
    ]
    fig.legend(
        handles=handles,
        title="aggregation (arrow direction: t1 -> t2 -> t3)",
        loc="lower center",
        ncol=7,
        frameon=False,
    )
    fig.suptitle(
        "Task 2 ADS Plane: Turn 1 to Turn 3 by Aggregation",
        fontsize=14,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0.07, 1, 0.95])
    out = FIGURES_DIR / "turn123_ads_plane.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print(f"Wrote {out}")


def plot_one_aggregation_ads_plane(focus: pd.DataFrame, method: str) -> None:
    """One clear ADS-plane figure for one aggregation method."""

    sub = focus[
        (focus["aggregation"] == method)
        & (focus["turn"].isin([1, 2, 3]))
    ].copy()
    if sub.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    full_ax, zoom_ax = axes
    style_plane(full_ax, zoom=False)
    style_plane(zoom_ax, zoom=True)

    full_ax.set_title("full ADS plane", fontsize=11, fontweight="bold")
    zoom_ax.set_title("zoom on turn 1 -> turn 3", fontsize=11, fontweight="bold")

    zoom_ax.set_xlim(
        max(0, sub["p_inval"].min() - 0.045),
        min(1, sub["p_inval"].max() + 0.045),
    )
    zoom_ax.set_ylim(
        max(0, sub["p_val"].min() - 0.035),
        min(1.02, sub["p_val"].max() + 0.035),
    )

    for ax in axes:
        for model in sorted(sub["model"].unique()):
            model_df = sub[sub["model"] == model].sort_values("turn")
            x = model_df["p_inval"].tolist()
            y = model_df["p_val"].tolist()
            color = MODEL_COLOR.get(model, "#444444")

            ax.plot(x, y, color=color, linewidth=1.4, zorder=3)
            if len(x) == 2:
                ax.annotate(
                    "",
                    xy=(x[1], y[1]),
                    xytext=(x[0], y[0]),
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": color,
                        "lw": 1.15,
                        "shrinkA": 5,
                        "shrinkB": 5,
                    },
                    zorder=4,
                )

            for row in model_df.itertuples(index=False):
                ax.scatter(
                    row.p_inval,
                    row.p_val,
                    s=58,
                    marker=TURN_MARKER[int(row.turn)],
                    color=color,
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=5,
                )
                if ax is zoom_ax:
                    ax.annotate(
                        f"t{int(row.turn)}\nADS={row.ads:.1f}",
                        (row.p_inval, row.p_val),
                        xytext=(6, 5 if int(row.turn) == 2 else -18),
                        textcoords="offset points",
                        fontsize=7,
                        color=color,
                        linespacing=1.05,
                        zorder=6,
                    )

    model_handles = [
        plt.Line2D([0], [0], color=color, marker="o", linewidth=1.4, markersize=5, label=model)
        for model, color in MODEL_COLOR.items()
        if model in set(sub["model"])
    ]
    turn_handles = [
        plt.Line2D([0], [0], color="#555555", marker=TURN_MARKER[turn], linewidth=0, markersize=6, label=f"turn {turn}")
        for turn in (1, 2, 3)
    ]
    fig.legend(
        handles=model_handles + turn_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
    )
    fig.suptitle(
        f"Task 2 ADS Plane: {METHOD_LABEL[method]} aggregation",
        fontsize=14,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0.09, 1, 0.93])
    out = FIGURES_DIR / f"ads_plane_{method}_turn123_full_zoom.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print(f"Wrote {out}")


def plot_style_per_aggregation(focus: pd.DataFrame) -> None:
    for method in METHOD_ORDER:
        plot_one_aggregation_ads_plane(focus, method)


def plot_turn1_ads_plane(focus: pd.DataFrame) -> None:
    """Separate turn-1 ADS plot, because turn 1 is the clean official horizon."""

    methods = ["unweighted", "mean"]
    sub = focus[
        (focus["turn"] == 1)
        & (focus["aggregation"].isin(methods))
    ].copy()
    if sub.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    full_ax, zoom_ax = axes
    style_plane(full_ax, zoom=False)
    style_plane(zoom_ax, zoom=True)
    full_ax.set_title("full ADS plane", fontsize=11, fontweight="bold")
    zoom_ax.set_title("zoom on turn 1", fontsize=11, fontweight="bold")

    zoom_ax.set_xlim(
        max(0, sub["p_inval"].min() - 0.045),
        min(1, sub["p_inval"].max() + 0.045),
    )
    zoom_ax.set_ylim(
        max(0, sub["p_val"].min() - 0.035),
        min(1.02, sub["p_val"].max() + 0.035),
    )

    method_offsets = {
        "unweighted": (-10, -24),
        "mean": (8, 5),
    }

    for ax in axes:
        for model in sorted(sub["model"].unique()):
            model_df = sub[sub["model"] == model]
            color = MODEL_COLOR.get(model, "#444444")
            for row in model_df.itertuples(index=False):
                marker = "o" if row.aggregation == "unweighted" else "D"
                facecolor = "white" if row.aggregation == "unweighted" else color
                ax.scatter(
                    row.p_inval,
                    row.p_val,
                    s=64,
                    marker=marker,
                    facecolor=facecolor,
                    edgecolor=color,
                    linewidth=1.3,
                    zorder=5,
                )
                if ax is zoom_ax:
                    dx, dy = method_offsets[str(row.aggregation)]
                    ax.annotate(
                        f"{row.aggregation}\nADS={row.ads:.1f}",
                        (row.p_inval, row.p_val),
                        xytext=(dx, dy),
                        textcoords="offset points",
                        fontsize=7,
                        color=color,
                        ha="left" if dx >= 0 else "right",
                        linespacing=1.05,
                        zorder=6,
                    )

    model_handles = [
        plt.Line2D([0], [0], color=color, marker="o", linewidth=0, markersize=6, label=model)
        for model, color in MODEL_COLOR.items()
        if model in set(sub["model"])
    ]
    method_handles = [
        plt.Line2D([0], [0], marker="o", markerfacecolor="white", markeredgecolor="#555555", linewidth=0, markersize=6, label="unweighted"),
        plt.Line2D([0], [0], marker="D", color="#555555", linewidth=0, markersize=6, label="ADS hinged weight"),
    ]
    fig.legend(
        handles=model_handles + method_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
    )
    fig.suptitle(
        "Task 2 ADS Plane: turn 1 official horizon",
        fontsize=14,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0.09, 1, 0.93])
    out = FIGURES_DIR / "ads_plane_turn1_official.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print(f"Wrote {out}")


def plot_aggregation_mosaic_turn23(focus: pd.DataFrame) -> None:
    """One mosaic figure: each panel is one aggregation method."""

    sub = focus[focus["turn"].isin([2, 3])].copy()
    if sub.empty:
        return

    mosaic = [
        ["unweighted", "lead", "mean", "median"],
        ["max", "min", "sum", "."],
    ]
    fig, axes = plt.subplot_mosaic(mosaic, figsize=(14, 7.2), sharex=True, sharey=True)

    xmin = max(0, sub["p_inval"].min() - 0.04)
    xmax = min(1, sub["p_inval"].max() + 0.04)
    ymin = max(0, sub["p_val"].min() - 0.035)
    ymax = min(1.02, sub["p_val"].max() + 0.035)

    for method in METHOD_ORDER:
        ax = axes[method]
        method_df = sub[sub["aggregation"] == method]
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.plot([0, 1], [0, 1], color="#555555", linewidth=0.85, linestyle="--", zorder=1)
        ax.grid(True, color="#d9d9d9", linewidth=0.45, alpha=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", labelsize=8, length=0)
        ax.set_title(METHOD_LABEL[method], fontsize=11, fontweight="bold")

        for model in sorted(method_df["model"].unique()):
            model_df = method_df[method_df["model"] == model].sort_values("turn")
            color = MODEL_COLOR.get(model, "#444444")
            x = model_df["p_inval"].tolist()
            y = model_df["p_val"].tolist()
            ax.plot(x, y, color=color, linewidth=1.2, zorder=3)
            if len(x) == 2:
                ax.annotate(
                    "",
                    xy=(x[1], y[1]),
                    xytext=(x[0], y[0]),
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": color,
                        "lw": 1.0,
                        "shrinkA": 5,
                        "shrinkB": 5,
                    },
                    zorder=4,
                )

            for row in model_df.itertuples(index=False):
                ax.scatter(
                    row.p_inval,
                    row.p_val,
                    s=48,
                    marker=TURN_MARKER[int(row.turn)],
                    color=color,
                    edgecolor="white",
                    linewidth=0.7,
                    zorder=5,
                )

    for key in ("max", "min", "sum"):
        axes[key].set_xlabel("p_inval", fontsize=9)
    for key in ("unweighted", "max"):
        axes[key].set_ylabel("p_val", fontsize=9)

    handles = [
        plt.Line2D([0], [0], color=color, marker="o", linewidth=1.3, markersize=5, label=model)
        for model, color in MODEL_COLOR.items()
    ]
    handles.extend(
        [
            plt.Line2D([0], [0], color="#555555", marker=TURN_MARKER[2], linewidth=0, markersize=6, label="turn 2"),
            plt.Line2D([0], [0], color="#555555", marker=TURN_MARKER[3], linewidth=0, markersize=6, label="turn 3"),
        ]
    )
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Task 2 ADS Plane Mosaic: Aggregation Methods, Turn 2 to Turn 3", fontsize=14)
    fig.tight_layout(rect=[0, 0.07, 1, 0.94])
    out = FIGURES_DIR / "ads_plane_aggregation_mosaic_turn23.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print(f"Wrote {out}")


def plot_all_turns_per_aggregation(focus: pd.DataFrame) -> None:
    """Create one ADS-plane figure per aggregation for turns 1, 2, and 3."""

    for method in METHOD_ORDER:
        sub = focus[focus["aggregation"] == method].copy()
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(6.4, 5.6))
        style_plane(ax, zoom=False)
        ax.set_title(
            f"{METHOD_LABEL[method]} aggregation: turn 1 to turn 3",
            fontsize=12,
            fontweight="bold",
        )

        for model in sorted(sub["model"].unique()):
            model_df = sub[sub["model"] == model].sort_values("turn")
            color = MODEL_COLOR.get(model, "#444444")
            x = model_df["p_inval"].tolist()
            y = model_df["p_val"].tolist()
            ax.plot(x, y, color=color, linewidth=1.3, zorder=3)

            for row in model_df.itertuples(index=False):
                turn = int(row.turn)
                ax.scatter(
                    row.p_inval,
                    row.p_val,
                    s=54,
                    marker=TURN_MARKER[turn],
                    color=color,
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=5,
                )
                ax.annotate(
                    f"t{turn}",
                    (row.p_inval, row.p_val),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=7,
                    color=color,
                )

        handles = [
            plt.Line2D(
                [0],
                [0],
                color=color,
                marker="o",
                linewidth=1.3,
                markersize=5,
                label=model,
            )
            for model, color in MODEL_COLOR.items()
            if model in set(sub["model"])
        ]
        handles.extend(
            plt.Line2D(
                [0],
                [0],
                color="#555555",
                marker=TURN_MARKER[turn],
                linewidth=0,
                markersize=6,
                label=f"turn {turn}",
            )
            for turn in (1, 2, 3)
        )
        fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False)
        fig.tight_layout(rect=[0, 0.10, 1, 1])
        out = FIGURES_DIR / f"ads_plane_{method}_turn123.png"
        fig.savefig(out, dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out}")


def plot_aggregation_mosaic_all_turns(focus: pd.DataFrame) -> None:
    """Mosaic of aggregation methods showing turns 1, 2, and 3."""

    mosaic = [
        ["unweighted", "lead", "mean", "median"],
        ["max", "min", "sum", "."],
    ]
    fig, axes = plt.subplot_mosaic(mosaic, figsize=(14, 7.2), sharex=True, sharey=True)

    for method in METHOD_ORDER:
        ax = axes[method]
        method_df = focus[focus["aggregation"] == method]
        style_plane(ax, zoom=False)
        ax.set_title(METHOD_LABEL[method], fontsize=11, fontweight="bold")

        for model in sorted(method_df["model"].unique()):
            model_df = method_df[method_df["model"] == model].sort_values("turn")
            color = MODEL_COLOR.get(model, "#444444")
            ax.plot(
                model_df["p_inval"],
                model_df["p_val"],
                color=color,
                linewidth=1.2,
                zorder=3,
            )
            for row in model_df.itertuples(index=False):
                turn = int(row.turn)
                ax.scatter(
                    row.p_inval,
                    row.p_val,
                    s=46,
                    marker=TURN_MARKER[turn],
                    color=color,
                    edgecolor="white",
                    linewidth=0.7,
                    zorder=5,
                )

    handles = [
        plt.Line2D(
            [0],
            [0],
            color=color,
            marker="o",
            linewidth=1.3,
            markersize=5,
            label=model,
        )
        for model, color in MODEL_COLOR.items()
    ]
    handles.extend(
        plt.Line2D(
            [0],
            [0],
            color="#555555",
            marker=TURN_MARKER[turn],
            linewidth=0,
            markersize=6,
            label=f"turn {turn}",
        )
        for turn in (1, 2, 3)
    )
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False)
    fig.suptitle(
        "Task 2 ADS Plane Mosaic: Argument Weight at Turn 1 and Aggregated Weights Thereafter",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0.07, 1, 0.94])
    out = FIGURES_DIR / "ads_plane_aggregation_mosaic_turn123.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    ensure_dirs()
    df = pd.read_csv(SUMMARY)
    focus = focus_methods(df)
    plot_ads_lines(focus)
    plot_ads_plane(focus)
    plot_turn1_ads_plane(focus)
    plot_style_all_turns(focus)
    plot_style_per_aggregation(focus)
    plot_all_turns_per_aggregation(focus)
    plot_aggregation_mosaic_all_turns(focus)


if __name__ == "__main__":
    main()
