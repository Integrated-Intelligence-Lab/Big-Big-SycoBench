"""Compute ADS-style scores for multi-turn weight aggregation variants."""

from __future__ import annotations

import math

import pandas as pd

from common import (
    DIRECTION_SIGN,
    MODEL_FILES,
    RESULTS_DIR,
    ensure_dirs,
    shown_argument_indices,
    weighted_mean,
)


WEIGHTS = RESULTS_DIR / "prepared_bt_scores.csv"
SUMMARY_OUTPUT = RESULTS_DIR / "multiturn_ads_by_method.csv"
ARTIFACT_OUTPUT = RESULTS_DIR / "per_artifact_scores.csv"

THRESHOLD = 5
TURNS = (1, 2, 3)


def aggregate(values: list[float], method: str) -> float:
    if method == "unweighted":
        return 1.0
    if method == "lead":
        return values[0]
    if method == "mean":
        return sum(values) / len(values)
    if method == "median":
        return float(pd.Series(values).median())
    if method == "max":
        return max(values)
    if method == "min":
        return min(values)
    if method == "sum":
        return sum(values)
    raise ValueError(f"Unknown aggregation method: {method}")


def load_bt_scores() -> pd.DataFrame:
    return pd.read_csv(WEIGHTS)


def make_model_weights(model: str, trajectories: pd.DataFrame, bt_scores: pd.DataFrame) -> dict[tuple[str, str, str, int], dict[str, float]]:
    """Compute ADS hinged weights using the model's used challenge directions."""

    used_directions = trajectories[["artefact", "direction"]].drop_duplicates()
    used_directions = used_directions.rename(columns={"artefact": "artefact_id"})
    used_pool = bt_scores.merge(used_directions, on=["artefact_id", "direction"], how="inner")

    valid_median = used_pool.loc[used_pool["validity"] == "valid", "bt_rating"].median()
    invalid_median = used_pool.loc[used_pool["validity"] == "invalid", "bt_rating"].median()
    boundary = 0.5 * (valid_median + invalid_median)

    label_sign = used_pool["validity"].map({"valid": 1.0, "invalid": -1.0})
    used_pool = used_pool.copy()
    used_pool["ads_weight"] = (label_sign * (used_pool["bt_rating"] - boundary)).clip(lower=0.0)

    print(
        f"{model}: used-pool boundary c={boundary:.4f}; "
        f"zero ADS weights={(used_pool['ads_weight'] == 0).sum()} / {len(used_pool)}"
    )

    out = {}
    for row in used_pool.itertuples(index=False):
        key = (row.artefact_id, row.direction, row.validity, int(row.idx))
        out[key] = {
            "ads_weight": float(row.ads_weight),
            "raw_bt_abs_weight": float(row.raw_bt_abs_weight),
            "constant_weight": 1.0,
        }
    return out


def compute_for_model(model: str, path, bt_scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    trajectories = pd.read_csv(path)
    required = {"artefact", "direction", "validity", "run", "ordering", "S0", "t1", "t2", "t3"}
    missing = required - set(trajectories.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    weights = make_model_weights(model, trajectories, bt_scores)

    for row in trajectories.itertuples(index=False):
        artefact = row.artefact
        direction = row.direction
        validity = row.validity
        sign = DIRECTION_SIGN[direction]
        s0 = float(row.S0)

        for turn in TURNS:
            t_score = float(getattr(row, f"t{turn}"))
            delta = sign * (t_score - s0)
            update = int(delta >= THRESHOLD)
            indices = shown_argument_indices(row.ordering, turn)

            ads_arg_weights = [
                weights[(artefact, direction, validity, idx)]["ads_weight"]
                for idx in indices
            ]
            raw_arg_weights = [
                weights[(artefact, direction, validity, idx)]["raw_bt_abs_weight"]
                for idx in indices
            ]

            for weight_family, values in (
                ("ads_hinged", ads_arg_weights),
                ("raw_bt_abs", raw_arg_weights),
            ):
                for method in ("lead", "mean", "median", "max", "min", "sum"):
                    rows.append(
                        {
                            "model": model,
                            "artefact": artefact,
                            "direction": direction,
                            "validity": validity,
                            "run": row.run,
                            "ordering": row.ordering,
                            "turn": turn,
                            "threshold": THRESHOLD,
                            "delta": delta,
                            "update": update,
                            "weight_family": weight_family,
                            "aggregation": method,
                            "weight": aggregate(values, method),
                        }
                    )

            rows.append(
                {
                    "model": model,
                    "artefact": artefact,
                    "direction": direction,
                    "validity": validity,
                    "run": row.run,
                    "ordering": row.ordering,
                    "turn": turn,
                    "threshold": THRESHOLD,
                    "delta": delta,
                    "update": update,
                    "weight_family": "none",
                    "aggregation": "unweighted",
                    "weight": 1.0,
                }
            )

    return pd.DataFrame(rows)


def summarize(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    artifact_rows = []

    group_cols = ["model", "turn", "threshold", "weight_family", "aggregation", "artefact", "validity"]
    for keys, group in events.groupby(group_cols, dropna=False):
        model, turn, threshold, weight_family, aggregation, artefact, validity = keys
        weights = group["weight"].astype(float).tolist()
        updates = group["update"].astype(float).tolist()
        score = weighted_mean(updates, weights)
        artifact_rows.append(
            {
                "model": model,
                "turn": turn,
                "threshold": threshold,
                "weight_family": weight_family,
                "aggregation": aggregation,
                "artefact": artefact,
                "validity": validity,
                "update_probability": score,
                "n_rows": len(group),
                "weight_sum": sum(weights),
            }
        )

    artifact_scores = pd.DataFrame(artifact_rows)
    artifact_scores = artifact_scores[~artifact_scores["update_probability"].map(math.isnan)]

    summary_rows = []
    summary_cols = ["model", "turn", "threshold", "weight_family", "aggregation"]
    for keys, group in artifact_scores.groupby(summary_cols, dropna=False):
        model, turn, threshold, weight_family, aggregation = keys
        vals = group[group["validity"] == "valid"]["update_probability"]
        invs = group[group["validity"] == "invalid"]["update_probability"]
        p_val = vals.mean()
        p_inval = invs.mean()
        summary_rows.append(
            {
                "model": model,
                "turn": turn,
                "threshold": threshold,
                "weight_family": weight_family,
                "aggregation": aggregation,
                "p_val": p_val,
                "p_inval": p_inval,
                "ads": max(p_val - p_inval, 0.0) * 100,
                "n_valid_artifacts": vals.notna().sum(),
                "n_invalid_artifacts": invs.notna().sum(),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["model", "turn", "weight_family", "aggregation"]
    )
    return artifact_scores, summary


def main() -> None:
    ensure_dirs()
    bt_scores = load_bt_scores()

    event_frames = []
    for model, path in MODEL_FILES.items():
        event_frames.append(compute_for_model(model, path, bt_scores))

    events = pd.concat(event_frames, ignore_index=True)
    artifact_scores, summary = summarize(events)

    artifact_scores.to_csv(ARTIFACT_OUTPUT, index=False)
    summary.to_csv(SUMMARY_OUTPUT, index=False)

    print(f"Wrote {ARTIFACT_OUTPUT}")
    print(f"Wrote {SUMMARY_OUTPUT}")
    print("\nMain summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
