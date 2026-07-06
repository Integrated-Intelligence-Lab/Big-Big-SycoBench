import argparse
import csv
import math
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
INPUT_DIR = ROOT / "Andres" / "ads_inputs"
TRAJECTORY_PATH = INPUT_DIR / "trajectories" / "trajectories_challenge_22_gpt55.csv"
BT_PATH = INPUT_DIR / "bt" / "bt_scores_global.csv"
OUT_DIR = INPUT_DIR / "illustration"


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def positive_sigmoid(x_value: float, high: float, slope: float, midpoint: float) -> float:
    if x_value <= 0:
        return 0.0
    base = sigmoid(slope * (0.0 - midpoint))
    top = sigmoid(slope * (x_value - midpoint))
    return high * max(0.0, (top - base) / (1.0 - base))


def logistic(x_value: float, floor: float, high: float, slope: float, midpoint: float) -> float:
    return floor + (high - floor) * sigmoid(slope * (x_value - midpoint))


def gaussian(x_value: float, height: float, center: float, width: float) -> float:
    return height * math.exp(-0.5 * ((x_value - center) / width) ** 2)


def load_challenge_arguments() -> list[dict[str, str]]:
    seen = set()
    rows = []
    with TRAJECTORY_PATH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            idx = int(row["ordering"][0])
            key = (row["artefact"], row["direction"], row["validity"], idx)
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                "artefact_id": row["artefact"],
                "tier": row["tier"],
                "direction": row["direction"],
                "validity": row["validity"],
                "idx": str(idx),
            })
    return sorted(rows, key=lambda r: (r["artefact_id"], r["direction"], r["validity"], int(r["idx"])))


def load_bt() -> dict[tuple[str, str, str, str], dict[str, str]]:
    out = {}
    with BT_PATH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            out[(row["artefact_id"], row["direction"], row["validity"], row["idx"])] = row
    return out


def shape_value(shape_id: str, x_value: float) -> float:
    if shape_id == "calibrated_ideal":
        return positive_sigmoid(x_value, high=4.0, slope=3.4, midpoint=0.45)
    if shape_id == "sycophant_floor":
        return 1.55 + positive_sigmoid(x_value, high=4.0, slope=2.2, midpoint=0.35)
    if shape_id == "sycophant_early_takeoff":
        return logistic(x_value, floor=0.0, high=4.0, slope=3.8, midpoint=-0.25)
    if shape_id == "skeptic_late_takeoff":
        return positive_sigmoid(x_value, high=3.0, slope=2.4, midpoint=0.95)
    if shape_id == "stubborn":
        return 0.0
    if shape_id == "pushover_flat_high":
        return 4.0
    if shape_id == "super_sensitive":
        return positive_sigmoid(x_value, high=7.5, slope=5.4, midpoint=0.55)
    if shape_id == "linear":
        return 1.8 * x_value + 1.8
    if shape_id == "true_sycophant_valid_invalid":
        return 2.1 + 1.7 * abs(math.sin(1.65 * x_value + 0.2))
    if shape_id == "contrarian_decreasing":
        return 4.0 - 4.0 * sigmoid(4.0 * (x_value + 0.45))
    if shape_id == "bump_mid_quality":
        return gaussian(x_value, height=4.0, center=-0.05, width=0.36)
    if shape_id == "calibrated_noisy":
        return positive_sigmoid(x_value, high=4.0, slope=3.2, midpoint=0.45)
    raise ValueError(shape_id)


def pinned_overlay(shape_id: str, x_value: float) -> float:
    params = {
        "calibrated_ideal": (4.0, 3.0, 0.45),
        "sycophant_floor": (5.5, 0.6, 0.15),
        "sycophant_early_takeoff": (6.2, 0.8, 0.05),
        "skeptic_late_takeoff": (2.3, 2.4, 0.95),
        "stubborn": (0.0, 1.0, 0.5),
        "pushover_flat_high": (4.4, 0.1, 0.0),
        "super_sensitive": (8.0, 3.0, 0.55),
        "linear": (4.8, 3.7, 0.55),
        "true_sycophant_valid_invalid": (2.4, 0.1, 0.0),
        "contrarian_decreasing": (1.6, 0.1, 0.0),
        "bump_mid_quality": (2.0, 0.2, 0.0),
        "calibrated_noisy": (4.2, 3.2, 0.45),
    }
    high, slope, midpoint = params[shape_id]
    return positive_sigmoid(x_value, high=high, slope=slope, midpoint=midpoint)


def noise_sd(shape_id: str) -> float:
    if shape_id == "calibrated_noisy":
        return 0.9
    if shape_id in {"linear", "true_sycophant_valid_invalid", "contrarian_decreasing"}:
        return 0.25
    if shape_id in {"stubborn", "pushover_flat_high"}:
        return 0.15
    return 0.18


def shapes() -> list[tuple[str, str]]:
    return [
        ("calibrated_ideal", "calibrated (ideal)"),
        ("sycophant_floor", "sycophant: floor>0"),
        ("sycophant_early_takeoff", "sycophant: early take-off"),
        ("skeptic_late_takeoff", "skeptic: late take-off"),
        ("stubborn", "stubborn (z~0)"),
        ("pushover_flat_high", "pushover (flat-high)"),
        ("super_sensitive", "super-sensitive"),
        ("linear", "linear"),
        ("true_sycophant_valid_invalid", "true sycophant (valid=invalid)"),
        ("contrarian_decreasing", "anti-discerning (decreasing)"),
        ("bump_mid_quality", "bump (mid-quality)"),
        ("calibrated_noisy", "calibrated but noisy"),
    ]


def write_points(seed: int) -> Path:
    rng = random.Random(seed)
    args = load_challenge_arguments()
    bt = load_bt()
    out_path = OUT_DIR / "shape_gallery_synthetic_points.csv"
    fieldnames = [
        "shape_id", "shape_label", "artefact_id", "tier", "direction", "validity",
        "idx", "item_id", "bt_rating", "synthetic_shift", "true_shift",
        "pinned_overlay_shift", "seed",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for shape_id, shape_label in shapes():
            for arg in args:
                key = (arg["artefact_id"], arg["direction"], arg["validity"], arg["idx"])
                bt_row = bt[key]
                x_value = float(bt_row["bt_rating"])
                true_y = shape_value(shape_id, x_value)
                y_value = true_y + rng.gauss(0.0, noise_sd(shape_id))
                writer.writerow({
                    "shape_id": shape_id,
                    "shape_label": shape_label,
                    "artefact_id": arg["artefact_id"],
                    "tier": arg["tier"],
                    "direction": arg["direction"],
                    "validity": arg["validity"],
                    "idx": arg["idx"],
                    "item_id": bt_row["item_id"],
                    "bt_rating": f"{x_value:.12g}",
                    "synthetic_shift": f"{y_value:.12g}",
                    "true_shift": f"{true_y:.12g}",
                    "pinned_overlay_shift": f"{pinned_overlay(shape_id, x_value):.12g}",
                    "seed": str(seed),
                })
    return out_path


def write_curves() -> Path:
    out_path = OUT_DIR / "shape_gallery_synthetic_curves.csv"
    grid = [-2.5 + i * (4.0 / 240) for i in range(241)]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "shape_id", "shape_label", "bt_rating", "true_shift", "pinned_overlay_shift",
        ])
        writer.writeheader()
        for shape_id, shape_label in shapes():
            for x_value in grid:
                writer.writerow({
                    "shape_id": shape_id,
                    "shape_label": shape_label,
                    "bt_rating": f"{x_value:.12g}",
                    "true_shift": f"{shape_value(shape_id, x_value):.12g}",
                    "pinned_overlay_shift": f"{pinned_overlay(shape_id, x_value):.12g}",
                })
    return out_path


def write_parameters(seed: int) -> Path:
    out_path = OUT_DIR / "shape_gallery_synthetic_parameters.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["shape_id", "shape_label", "noise_sd", "seed"])
        writer.writeheader()
        for shape_id, shape_label in shapes():
            writer.writerow({
                "shape_id": shape_id,
                "shape_label": shape_label,
                "noise_sd": f"{noise_sd(shape_id):.12g}",
                "seed": str(seed),
            })
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"seed={args.seed}")
    print(f"trajectory-path={TRAJECTORY_PATH}")
    print(f"bt-path={BT_PATH}")
    print(write_points(args.seed))
    print(write_curves())
    print(write_parameters(args.seed))


if __name__ == "__main__":
    main()
