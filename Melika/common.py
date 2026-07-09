"""Shared helpers for the Task multi-turn weighting experiment."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ROOT = PROJECT_ROOT / "task2_multiturn_weights"
DATA_DIR = TASK_ROOT / "data"
RESULTS_DIR = TASK_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

INPUTS = {
    "bt_scores": {
        "path": DATA_DIR / "bt_scores.csv",
        "url": "https://raw.githubusercontent.com/Integrated-Intelligence-Lab/Big-Big-SycoBench/main/Marthe/bt_global/results/bt_scores.csv",
    },
    "gpt55_trajectories": {
        "path": DATA_DIR / "trajectories_challenge_22.csv",
        "url": "https://raw.githubusercontent.com/Integrated-Intelligence-Lab/Big-Big-SycoBench/main/Marthe/figure_1/results/trajectories_challenge_22.csv",
    },
    "o4mini_trajectories": {
        "path": DATA_DIR / "trajectories_challenge_22_o4mini.csv",
        "url": "https://raw.githubusercontent.com/Integrated-Intelligence-Lab/Big-Big-SycoBench/main/Marthe/figure_1/results/trajectories_challenge_22_o4mini.csv",
    },
}

MODEL_FILES = {
    "gpt-5.5": INPUTS["gpt55_trajectories"]["path"],
    "o4-mini": INPUTS["o4mini_trajectories"]["path"],
}

DIRECTION_SIGN = {
    "raise": 1,
    "lower": -1,
}


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def shown_argument_indices(ordering: str, turn: int) -> list[int]:
    """Return argument indices shown up to a given turn.

    Example: ordering "120" means turn 1 saw 1, turn 2 saw 1 and 2,
    turn 3 saw 1, 2, and 0.
    """

    if turn not in (1, 2, 3):
        raise ValueError(f"turn must be 1, 2, or 3, got {turn}")
    return [int(ch) for ch in str(ordering)[:turn]]


def weighted_mean(values, weights) -> float:
    total_weight = sum(weights)
    if total_weight == 0:
        return float("nan")
    return sum(v * w for v, w in zip(values, weights)) / total_weight

