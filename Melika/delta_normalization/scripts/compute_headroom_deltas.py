#!/usr/bin/env python3
"""Compute headroom-adjusted score movements for SycoBench runs.

This script is intentionally dependency-free. It reads a JSONL file in Seorin's
`sycophancy_runs.jsonl` format and writes CSV summaries plus simple SVG plots.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute raw and headroom-normalized SycoBench score deltas."
    )
    parser.add_argument("--input", required=True, help="Input sycophancy_runs.jsonl file.")
    parser.add_argument("--outdir", required=True, help="Directory for CSV and SVG outputs.")
    parser.add_argument("--scale-min", type=float, default=1.0, help="Minimum score.")
    parser.add_argument("--scale-max", type=float, default=10.0, help="Maximum score.")
    parser.add_argument("--source", default="unknown", help="Source label for output rows.")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Could not parse JSON on line {line_no}: {exc}") from exc
    return rows


def safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_movement(
    s0: float,
    post: float,
    direction: str,
    scale_min: float,
    scale_max: float,
) -> dict:
    raw_delta = post - s0
    if direction == "raise":
        directional_delta = post - s0
        available_room = scale_max - s0
    elif direction == "lower":
        directional_delta = s0 - post
        available_room = s0 - scale_min
    else:
        raise ValueError(f"Unknown direction: {direction!r}")

    normalized_delta = None
    if available_room > 0:
        normalized_delta = directional_delta / available_room

    return {
        "raw_delta": raw_delta,
        "absolute_delta": abs(raw_delta),
        "directional_delta": directional_delta,
        "available_room": available_room,
        "normalized_delta": normalized_delta,
        "aligned_with_push": directional_delta > 0,
    }


def expand_rows(rows: list[dict], source: str, scale_min: float, scale_max: float) -> list[dict]:
    expanded = []

    for row in rows:
        s0 = safe_float(row.get("S0"))
        direction = row.get("direction")
        if s0 is None or direction not in {"raise", "lower"}:
            continue

        for arm, key in [("valid", "valid_scores"), ("invalid", "invalid_scores")]:
            scores = row.get(key) or []
            for cycle_index, post_raw in enumerate(scores, start=1):
                post = safe_float(post_raw)
                if post is None:
                    continue
                movement = compute_movement(s0, post, direction, scale_min, scale_max)
                normalized = movement["normalized_delta"]
                expanded.append(
                    {
                        "source": source,
                        "model": row.get("model", ""),
                        "persona": row.get("persona", ""),
                        "artefact": row.get("artefact", ""),
                        "quality": row.get("quality", ""),
                        "length": row.get("length", ""),
                        "anchor": row.get("anchor", ""),
                        "run": row.get("run", ""),
                        "direction": direction,
                        "arm": arm,
                        "cycle": cycle_index,
                        "S0": s0,
                        "S_post": post,
                        "raw_delta": movement["raw_delta"],
                        "absolute_delta": movement["absolute_delta"],
                        "directional_delta": movement["directional_delta"],
                        "available_room": movement["available_room"],
                        "normalized_delta": normalized,
                        "aligned_with_push": movement["aligned_with_push"],
                    }
                )

    return expanded


def final_cycle_rows(expanded: list[dict]) -> list[dict]:
    by_key = {}
    for row in expanded:
        key = (
            row["source"],
            row["model"],
            row["persona"],
            row["artefact"],
            row["run"],
            row["direction"],
            row["arm"],
        )
        if key not in by_key or row["cycle"] > by_key[key]["cycle"]:
            by_key[key] = row
    return list(by_key.values())


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize(rows: list[dict], group_fields: list[str]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        value = row.get("normalized_delta")
        if value is None:
            continue
        key = tuple(row.get(field, "") for field in group_fields)
        grouped[key].append(row)

    out = []
    for key, group in sorted(grouped.items()):
        normalized = [r["normalized_delta"] for r in group]
        directional = [r["directional_delta"] for r in group]
        raw = [r["raw_delta"] for r in group]
        aligned = [1 if r["aligned_with_push"] else 0 for r in group]
        out.append(
            {
                **{field: key[i] for i, field in enumerate(group_fields)},
                "n": len(group),
                "mean_raw_delta": mean(raw),
                "mean_directional_delta": mean(directional),
                "mean_normalized_delta": mean(normalized),
                "sd_normalized_delta": pstdev(normalized) if len(normalized) > 1 else 0.0,
                "min_normalized_delta": min(normalized),
                "max_normalized_delta": max(normalized),
                "alignment_rate": mean(aligned),
            }
        )
    return out


def fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.6g}"
    return str(value)


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    rounded = []
    for row in rows:
        rounded.append({k: fmt(v) for k, v in row.items()})
    write_csv(path, rounded, fieldnames)


def grouped_bar_svg(
    path: Path,
    title: str,
    groups: list[str],
    series: list[str],
    values: dict[tuple[str, str], float],
    ylabel: str,
) -> None:
    width = 980
    height = 560
    margin_left = 90
    margin_right = 40
    margin_top = 70
    margin_bottom = 100
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    colors = {"valid": "#2ca25f", "invalid": "#de2d26", "raw": "#756bb1", "normalized": "#3182bd"}

    vals = [v for v in values.values() if v is not None]
    y_min = min(0.0, min(vals) if vals else 0.0)
    y_max = max(0.0, max(vals) if vals else 1.0)
    if abs(y_max - y_min) < 1e-9:
        y_max = y_min + 1.0
    pad = (y_max - y_min) * 0.12
    y_min -= pad
    y_max += pad

    def y_pos(v: float) -> float:
        return margin_top + (y_max - v) / (y_max - y_min) * plot_h

    x_group = plot_w / max(1, len(groups))
    bar_gap = 8
    bar_w = max(10, (x_group - 36) / max(1, len(series)) - bar_gap)
    zero_y = y_pos(0.0)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="34" text-anchor="middle" font-family="Arial" font-size="24">{html.escape(title)}</text>',
        f'<text x="24" y="{height/2}" transform="rotate(-90 24 {height/2})" text-anchor="middle" font-family="Arial" font-size="15">{html.escape(ylabel)}</text>',
        f'<line x1="{margin_left}" y1="{zero_y:.2f}" x2="{width-margin_right}" y2="{zero_y:.2f}" stroke="#555" stroke-width="1"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height-margin_bottom}" stroke="#222" stroke-width="1"/>',
        f'<line x1="{margin_left}" y1="{height-margin_bottom}" x2="{width-margin_right}" y2="{height-margin_bottom}" stroke="#222" stroke-width="1"/>',
    ]

    for tick in range(6):
        val = y_min + (y_max - y_min) * tick / 5
        y = y_pos(val)
        parts.append(f'<line x1="{margin_left-5}" y1="{y:.2f}" x2="{margin_left}" y2="{y:.2f}" stroke="#222"/>')
        parts.append(f'<text x="{margin_left-10}" y="{y+5:.2f}" text-anchor="end" font-family="Arial" font-size="12">{val:.2f}</text>')
        parts.append(f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width-margin_right}" y2="{y:.2f}" stroke="#eee"/>')

    for gi, group in enumerate(groups):
        group_x0 = margin_left + gi * x_group + 18
        label_x = margin_left + gi * x_group + x_group / 2
        parts.append(f'<text x="{label_x:.2f}" y="{height-58}" text-anchor="middle" font-family="Arial" font-size="13">{html.escape(group)}</text>')
        for si, serie in enumerate(series):
            value = values.get((group, serie))
            if value is None:
                continue
            x = group_x0 + si * (bar_w + bar_gap)
            y = y_pos(max(value, 0.0))
            h = abs(y_pos(value) - zero_y)
            fill = colors.get(serie, "#636363")
            if value < 0:
                y = zero_y
            parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{h:.2f}" fill="{fill}" opacity="0.86"/>')
            label_y = y - 6 if value >= 0 else y + h + 16
            parts.append(f'<text x="{x + bar_w/2:.2f}" y="{label_y:.2f}" text-anchor="middle" font-family="Arial" font-size="12">{value:.2f}</text>')

    legend_x = width - margin_right - 170
    legend_y = 48
    for i, serie in enumerate(series):
        y = legend_y + i * 22
        parts.append(f'<rect x="{legend_x}" y="{y}" width="14" height="14" fill="{colors.get(serie, "#636363")}" opacity="0.86"/>')
        parts.append(f'<text x="{legend_x+22}" y="{y+12}" font-family="Arial" font-size="13">{html.escape(serie)}</text>')

    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def histogram_svg(
    path: Path,
    title: str,
    rows: list[dict],
    value_field: str,
    xlabel: str,
    bins: int = 14,
) -> None:
    width = 980
    height = 560
    margin_left = 80
    margin_right = 40
    margin_top = 70
    margin_bottom = 85
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    colors = {"valid": "#2ca25f", "invalid": "#de2d26"}

    by_arm = {"valid": [], "invalid": []}
    for row in rows:
        value = row.get(value_field)
        arm = row.get("arm")
        if value is None or arm not in by_arm:
            continue
        by_arm[arm].append(float(value))

    values = by_arm["valid"] + by_arm["invalid"]
    if not values:
        return

    x_min = min(values)
    x_max = max(values)
    if abs(x_max - x_min) < 1e-9:
        x_min -= 0.5
        x_max += 0.5
    else:
        pad = (x_max - x_min) * 0.08
        x_min -= pad
        x_max += pad

    bin_w = (x_max - x_min) / bins
    counts = {arm: [0 for _ in range(bins)] for arm in by_arm}
    for arm, vals in by_arm.items():
        for value in vals:
            idx = int((value - x_min) / bin_w)
            idx = min(max(idx, 0), bins - 1)
            counts[arm][idx] += 1

    y_max = max(max(c) for c in counts.values())
    y_max = max(1, y_max)

    def x_pos(value: float) -> float:
        return margin_left + (value - x_min) / (x_max - x_min) * plot_w

    def y_pos(value: float) -> float:
        return margin_top + (y_max - value) / y_max * plot_h

    zero_x = x_pos(0.0) if x_min <= 0 <= x_max else None
    group_w = plot_w / bins
    bar_w = max(4, group_w / 2 - 3)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="34" text-anchor="middle" font-family="Arial" font-size="24">{html.escape(title)}</text>',
        f'<text x="{width/2}" y="{height-24}" text-anchor="middle" font-family="Arial" font-size="15">{html.escape(xlabel)}</text>',
        f'<text x="24" y="{height/2}" transform="rotate(-90 24 {height/2})" text-anchor="middle" font-family="Arial" font-size="15">count</text>',
        f'<line x1="{margin_left}" y1="{height-margin_bottom}" x2="{width-margin_right}" y2="{height-margin_bottom}" stroke="#222" stroke-width="1"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height-margin_bottom}" stroke="#222" stroke-width="1"/>',
    ]
    if zero_x is not None:
        parts.append(f'<line x1="{zero_x:.2f}" y1="{margin_top}" x2="{zero_x:.2f}" y2="{height-margin_bottom}" stroke="#555" stroke-dasharray="4 4"/>')

    for tick in range(6):
        val = y_max * tick / 5
        y = y_pos(val)
        parts.append(f'<line x1="{margin_left-5}" y1="{y:.2f}" x2="{margin_left}" y2="{y:.2f}" stroke="#222"/>')
        parts.append(f'<text x="{margin_left-10}" y="{y+5:.2f}" text-anchor="end" font-family="Arial" font-size="12">{val:.0f}</text>')
        parts.append(f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width-margin_right}" y2="{y:.2f}" stroke="#eee"/>')

    for tick in range(6):
        val = x_min + (x_max - x_min) * tick / 5
        x = x_pos(val)
        parts.append(f'<line x1="{x:.2f}" y1="{height-margin_bottom}" x2="{x:.2f}" y2="{height-margin_bottom+5}" stroke="#222"/>')
        parts.append(f'<text x="{x:.2f}" y="{height-margin_bottom+22}" text-anchor="middle" font-family="Arial" font-size="12">{val:.2f}</text>')

    for i in range(bins):
        x0 = margin_left + i * group_w
        for arm_index, arm in enumerate(["valid", "invalid"]):
            count = counts[arm][i]
            h = height - margin_bottom - y_pos(count)
            x = x0 + arm_index * (bar_w + 3) + 2
            y = y_pos(count)
            parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{h:.2f}" fill="{colors[arm]}" opacity="0.78"/>')

    legend_x = width - margin_right - 160
    for i, arm in enumerate(["valid", "invalid"]):
        y = 48 + i * 22
        parts.append(f'<rect x="{legend_x}" y="{y}" width="14" height="14" fill="{colors[arm]}" opacity="0.78"/>')
        parts.append(f'<text x="{legend_x+22}" y="{y+12}" font-family="Arial" font-size="13">{arm}</text>')

    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def write_figure_index(path: Path, figure_names: list[str]) -> None:
    cards = []
    for name in figure_names:
        escaped = html.escape(name)
        cards.append(
            "\n".join(
                [
                    '<section style="margin: 24px 0;">',
                    f"<h2>{escaped}</h2>",
                    f'<img src="{escaped}" style="max-width: 100%; border: 1px solid #ddd;" />',
                    "</section>",
                ]
            )
        )
    doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Delta Normalization Figures</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #111; }}
    h1 {{ margin-bottom: 8px; }}
    p {{ max-width: 820px; line-height: 1.45; }}
  </style>
</head>
<body>
  <h1>Delta Normalization Figures</h1>
  <p>
    These plots show headroom-normalized movement. Positive values mean the model
    moved with the push; negative values mean it moved against the push.
  </p>
  {''.join(cards)}
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def build_plot_values(rows: list[dict], group_field: str, value_field: str = "normalized_delta"):
    grouped = defaultdict(list)
    for row in rows:
        value = row.get(value_field)
        if value is None:
            continue
        grouped[(str(row.get(group_field, "")), str(row.get("arm", "")))].append(value)
    return {key: mean(vals) for key, vals in grouped.items()}


def build_direction_plot_values(rows: list[dict]):
    grouped = defaultdict(list)
    for row in rows:
        value = row.get("normalized_delta")
        if value is None:
            continue
        group = f'{row.get("direction", "")} / {row.get("arm", "")}'
        grouped[(group, "normalized")].append(value)
    return {key: mean(vals) for key, vals in grouped.items()}


def build_artifact_direction_plot_values(rows: list[dict]):
    grouped = defaultdict(list)
    for row in rows:
        value = row.get("normalized_delta")
        if value is None:
            continue
        group = f'{row.get("artefact", "")} / {row.get("direction", "")}'
        grouped[(group, str(row.get("arm", "")))].append(value)
    return {key: mean(vals) for key, vals in grouped.items()}


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    outdir = Path(args.outdir)

    rows = read_jsonl(input_path)
    expanded = expand_rows(rows, args.source, args.scale_min, args.scale_max)
    final_rows = final_cycle_rows(expanded)

    fieldnames = [
        "source",
        "model",
        "persona",
        "artefact",
        "quality",
        "length",
        "anchor",
        "run",
        "direction",
        "arm",
        "cycle",
        "S0",
        "S_post",
        "raw_delta",
        "absolute_delta",
        "directional_delta",
        "available_room",
        "normalized_delta",
        "aligned_with_push",
    ]
    write_csv(outdir / "headroom_deltas_by_cycle.csv", expanded, fieldnames)
    write_csv(outdir / "headroom_deltas_final.csv", final_rows, fieldnames)

    summary_persona_arm = summarize(final_rows, ["persona", "arm"])
    summary_direction_arm = summarize(final_rows, ["direction", "arm"])
    summary_persona_direction_arm = summarize(final_rows, ["persona", "direction", "arm"])
    summary_artifact_arm = summarize(final_rows, ["artefact", "direction", "arm"])
    summary_cycle_arm = summarize(expanded, ["cycle", "arm"])

    write_summary_csv(outdir / "summary_by_persona_arm.csv", summary_persona_arm)
    write_summary_csv(outdir / "summary_by_direction_arm.csv", summary_direction_arm)
    write_summary_csv(outdir / "summary_by_persona_direction_arm.csv", summary_persona_direction_arm)
    write_summary_csv(outdir / "summary_by_artifact_arm.csv", summary_artifact_arm)
    write_summary_csv(outdir / "summary_by_cycle_arm.csv", summary_cycle_arm)

    figures = outdir / "figures"
    figure_names = [
        "mean_normalized_by_persona.svg",
        "mean_normalized_by_direction_arm.svg",
        "raw_delta_distribution.svg",
        "normalized_delta_distribution.svg",
        "artifact_vulnerability_by_direction.svg",
    ]
    persona_groups = sorted({str(r["persona"]) for r in final_rows})
    grouped_bar_svg(
        figures / "mean_normalized_by_persona.svg",
        "Mean Headroom-Normalized Movement by Persona",
        persona_groups,
        ["valid", "invalid"],
        build_plot_values(final_rows, "persona"),
        "mean normalized movement",
    )

    direction_groups = ["raise / valid", "raise / invalid", "lower / valid", "lower / invalid"]
    grouped_bar_svg(
        figures / "mean_normalized_by_direction_arm.svg",
        "Mean Headroom-Normalized Movement by Direction and Arm",
        direction_groups,
        ["normalized"],
        build_direction_plot_values(final_rows),
        "mean normalized movement",
    )

    histogram_svg(
        figures / "raw_delta_distribution.svg",
        "Raw Delta Distribution, Final Cycle",
        final_rows,
        "raw_delta",
        "raw delta = S_final - S0",
    )

    histogram_svg(
        figures / "normalized_delta_distribution.svg",
        "Headroom-Normalized Delta Distribution, Final Cycle",
        final_rows,
        "normalized_delta",
        "normalized movement in pushed direction",
    )

    artifact_groups = sorted(
        {f'{r.get("artefact", "")} / {r.get("direction", "")}' for r in final_rows}
    )
    grouped_bar_svg(
        figures / "artifact_vulnerability_by_direction.svg",
        "Artifact-Level Vulnerability by Direction",
        artifact_groups,
        ["valid", "invalid"],
        build_artifact_direction_plot_values(final_rows),
        "mean normalized movement",
    )
    write_figure_index(figures / "index.html", figure_names)

    print(f"Read {len(rows)} source rows")
    print(f"Wrote {len(expanded)} cycle-level rows")
    print(f"Wrote {len(final_rows)} final-cycle rows")
    print(f"Output directory: {outdir}")


if __name__ == "__main__":
    main()
