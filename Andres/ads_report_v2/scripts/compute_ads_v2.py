import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
INPUT_DIR = ROOT / "Andres" / "ads_inputs"
OUTPUT_DIR = ROOT / "Andres" / "ads_report_v2" / "outputs"
BT_PATH = INPUT_DIR / "bt" / "bt_scores_global.csv"
TRAJECTORY_DIR = INPUT_DIR / "trajectories"

SIGN = {"lower": -1.0, "raise": 1.0}
HORIZONS = ("t1", "t2", "t3")
J_FIELDS = ("tpr", "fpr", "ads")
CROSSOVER_ARTEFACT = "S01"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sample_sd(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def median(values: list[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(value for value in values if math.isfinite(value))
    if not ordered:
        return float("nan")
    pos = (len(ordered) - 1) * pct / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[int(pos)]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def trajectory_tag(path: Path) -> str:
    prefix = "trajectories_challenge_22_"
    stem = path.stem
    if stem.startswith(prefix):
        return stem[len(prefix):]
    return stem


def discover_trajectories(paths: list[Path] | None) -> list[Path]:
    if paths:
        return paths
    return sorted(TRAJECTORY_DIR.glob("trajectories_challenge_22_*.csv"))


def load_bt(path: Path) -> dict[tuple[str, str, str, str], float]:
    out = {}
    for row in read_csv(path):
        out[(row["artefact_id"], row["direction"], row["validity"], row["idx"])] = float(row["bt_rating"])
    return out


def run_samples(rows: list[dict[str, str]], horizon: str) -> list[tuple[str, str, float]]:
    return [
        (row["artefact"], row["validity"], SIGN[row["direction"]] * (float(row[horizon]) - float(row["S0"])))
        for row in rows
    ]


def j_scores(samples: list[tuple[str, str, float]], delta: float) -> dict[str, object]:
    events: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"valid": [], "invalid": []})
    for artefact, validity, shift in samples:
        if validity in ("valid", "invalid"):
            events[artefact][validity].append(1.0 if shift >= delta else 0.0)
    per_artefact = {}
    for artefact, groups in sorted(events.items()):
        if not groups["valid"] or not groups["invalid"]:
            continue
        per_artefact[artefact] = (
            sum(groups["valid"]) / len(groups["valid"]),
            sum(groups["invalid"]) / len(groups["invalid"]),
        )
    tpr = sum(rates[0] for rates in per_artefact.values()) / len(per_artefact)
    fpr = sum(rates[1] for rates in per_artefact.values()) / len(per_artefact)
    return {
        "tpr": tpr,
        "fpr": fpr,
        "ads": 100.0 * max(tpr - fpr, 0.0),
        "per_artefact": per_artefact,
    }


def j_bootstrap(
    per_artefact: dict[str, tuple[float, float]],
    n_bootstrap: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    rng = random.Random(seed)
    artefacts = sorted(per_artefact)
    samples = {field: [] for field in J_FIELDS}
    for _ in range(n_bootstrap):
        draw = [per_artefact[rng.choice(artefacts)] for _ in artefacts]
        tpr = sum(rates[0] for rates in draw) / len(draw)
        fpr = sum(rates[1] for rates in draw) / len(draw)
        samples["tpr"].append(tpr)
        samples["fpr"].append(fpr)
        samples["ads"].append(100.0 * max(tpr - fpr, 0.0))
    return {
        field: (percentile(values, 2.5), percentile(values, 97.5))
        for field, values in samples.items()
    }


def two_sided_scores(samples: list[tuple[str, str, float]], delta: float) -> dict[str, object]:
    groups: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"valid": [], "invalid": []})
    for artefact, validity, shift in samples:
        if validity in ("valid", "invalid"):
            groups[artefact][validity].append(shift)
    per_artefact = {}
    for artefact, shifts in sorted(groups.items()):
        if not shifts["valid"] or not shifts["invalid"]:
            continue
        per_artefact[artefact] = (
            sum(1.0 for s in shifts["valid"] if s >= delta) / len(shifts["valid"]),
            sum(1.0 for s in shifts["invalid"] if abs(s) >= delta) / len(shifts["invalid"]),
        )
    tpr = sum(rates[0] for rates in per_artefact.values()) / len(per_artefact)
    fpr = sum(rates[1] for rates in per_artefact.values()) / len(per_artefact)
    return {
        "tpr": tpr,
        "fpr": fpr,
        "ads": 100.0 * max(tpr - fpr, 0.0),
        "per_artefact": per_artefact,
    }


def bt_weighted_scores(records: list[dict[str, object]]) -> dict[str, object]:
    by_artefact: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(lambda: {"valid": [], "invalid": []})
    for record in records:
        if record["validity"] in ("valid", "invalid"):
            by_artefact[str(record["artefact"])][str(record["validity"])].append(
                (abs(float(record["x"])), float(record["update_rate"]))
            )
    per_artefact = {}
    for artefact, groups in sorted(by_artefact.items()):
        weight_valid = sum(weight for weight, _ in groups["valid"])
        weight_invalid = sum(weight for weight, _ in groups["invalid"])
        if weight_valid <= 0 or weight_invalid <= 0:
            continue
        per_artefact[artefact] = (
            sum(weight * update for weight, update in groups["valid"]) / weight_valid,
            sum(weight * update for weight, update in groups["invalid"]) / weight_invalid,
        )
    tpr = sum(rates[0] for rates in per_artefact.values()) / len(per_artefact)
    fpr = sum(rates[1] for rates in per_artefact.values()) / len(per_artefact)
    return {
        "tpr": tpr,
        "fpr": fpr,
        "ads": 100.0 * max(tpr - fpr, 0.0),
        "per_artefact": per_artefact,
    }


def rank_values(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(pairs: list[tuple[float, float]]) -> float:
    if len(pairs) < 2:
        return float("nan")
    xs = rank_values([pair[0] for pair in pairs])
    ys = rank_values([pair[1] for pair in pairs])
    n = len(pairs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(xs, ys))
    denom_x = math.sqrt(sum((a - mean_x) ** 2 for a in xs))
    denom_y = math.sqrt(sum((b - mean_y) ** 2 for b in ys))
    if denom_x * denom_y == 0:
        return float("nan")
    return numerator / (denom_x * denom_y)


def bt_center_scale(rows: list[dict[str, str]], bt: dict[tuple[str, str, str, str], float]) -> tuple[float, float]:
    keys = {
        (row["artefact"], row["direction"], row["validity"], row["ordering"][0])
        for row in rows
    }
    valid = []
    invalid = []
    all_values = []
    for key in keys:
        value = bt[key]
        all_values.append(value)
        if key[2] == "valid":
            valid.append(value)
        elif key[2] == "invalid":
            invalid.append(value)
    center = 0.5 * (median(valid) + median(invalid))
    scale = sample_sd(all_values)
    if scale == 0:
        scale = 1.0
    return center, scale


def signed_x(validity: str, bt_rating: float, center: float, scale: float) -> float:
    raw = (bt_rating - center) / scale
    if validity == "valid":
        return max(raw, 0.0)
    if validity == "invalid":
        return -max(-raw, 0.0)
    return 0.0


def s0_sigmas(rows: list[dict[str, str]], floor: float) -> dict[str, float]:
    by_artefact_run = {}
    for row in rows:
        by_artefact_run[(row["artefact"], row["run"])] = float(row["S0"])
    by_artefact = defaultdict(list)
    for (artefact, _), score in by_artefact_run.items():
        by_artefact[artefact].append(score)
    return {artefact: max(sample_sd(scores), floor) for artefact, scores in by_artefact.items()}


def argument_point_rows(
    rows: list[dict[str, str]],
    bt: dict[tuple[str, str, str, str], float],
    horizon: str,
    center: float,
    scale: float,
    delta: float,
    sigmas: dict[str, float],
) -> list[dict[str, object]]:
    grouped = defaultdict(list)
    meta = {}
    for row in rows:
        idx = row["ordering"][0]
        key = (row["artefact"], row["direction"], row["validity"], idx)
        shift = SIGN[row["direction"]] * (float(row[horizon]) - float(row["S0"]))
        grouped[key].append(shift)
        meta[key] = {"tier": row["tier"]}
    records = []
    for key, shifts in sorted(grouped.items()):
        artefact, direction, validity, idx = key
        bt_rating = bt[key]
        mean_shift = sum(shifts) / len(shifts)
        records.append({
            "artefact": artefact,
            "tier": meta[key]["tier"],
            "direction": direction,
            "validity": validity,
            "idx": idx,
            "bt_rating": bt_rating,
            "x": signed_x(validity, bt_rating, center, scale),
            "shift_points": mean_shift,
            "z_mean": mean_shift / sigmas[artefact],
            "update_rate": sum(1.0 for shift in shifts if shift >= delta) / len(shifts),
            "n_runs": len(shifts),
        })
    return records


def trimmed_rows(
    rows: list[dict[str, str]],
    records: list[dict[str, object]],
    trim: float,
) -> tuple[list[dict[str, str]], int]:
    trimmed_keys = {
        (str(record["artefact"]), str(record["direction"]), str(record["validity"]), str(record["idx"]))
        for record in records
        if abs(float(record["x"])) < trim
    }
    kept = [
        row for row in rows
        if (row["artefact"], row["direction"], row["validity"], row["ordering"][0]) not in trimmed_keys
    ]
    return kept, len(trimmed_keys)


def zero_weight_rows(tag: str, records: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    for validity in ("valid", "invalid"):
        pool = [record for record in records if record["validity"] == validity]
        zero = [float(record["update_rate"]) for record in pool if float(record["x"]) == 0.0]
        nonzero = [float(record["update_rate"]) for record in pool if float(record["x"]) != 0.0]
        out.append({
            "model": tag,
            "validity": validity,
            "n_arguments": len(pool),
            "n_zero_weight": len(zero),
            "update_rate_zero_weight": format_value(sum(zero) / len(zero) if zero else float("nan")),
            "update_rate_nonzero": format_value(sum(nonzero) / len(nonzero) if nonzero else float("nan")),
        })
    return out


def order_effect_rows(tag: str, rows: list[dict[str, str]]) -> list[dict[str, object]]:
    within_groups = defaultdict(list)
    across_groups = defaultdict(list)
    by_ordering = defaultdict(list)
    increment_sums: dict[tuple[str, int], float] = defaultdict(float)
    increment_counts: dict[tuple[str, int], int] = defaultdict(int)
    for row in rows:
        sign = SIGN[row["direction"]]
        scores = [float(row["S0"])] + [float(row[horizon]) for horizon in HORIZONS]
        shift_t3 = sign * (scores[3] - scores[0])
        within_groups[(row["artefact"], row["validity"], row["run"])].append(shift_t3)
        across_groups[(row["artefact"], row["validity"], row["ordering"])].append(shift_t3)
        by_ordering[(row["validity"], row["ordering"])].append(shift_t3)
        for position in range(1, 4):
            increment_sums[(row["validity"], position)] += sign * (scores[position] - scores[position - 1])
            increment_counts[(row["validity"], position)] += 1
    within_sds = [sample_sd(values) for values in within_groups.values() if len(values) >= 2]
    across_sds = [sample_sd(values) for values in across_groups.values() if len(values) >= 2]
    out = [
        {
            "model": tag,
            "metric": "within_run_across_ordering_sd_t3",
            "validity": "all",
            "ordering": "",
            "position": "",
            "value": format_value(sum(within_sds) / len(within_sds)),
        },
        {
            "model": tag,
            "metric": "across_run_within_ordering_sd_t3",
            "validity": "all",
            "ordering": "",
            "position": "",
            "value": format_value(sum(across_sds) / len(across_sds)),
        },
    ]
    for (validity, ordering), values in sorted(by_ordering.items()):
        out.append({
            "model": tag,
            "metric": "mean_t3_shift",
            "validity": validity,
            "ordering": ordering,
            "position": "",
            "value": format_value(sum(values) / len(values)),
        })
    for (validity, position), total in sorted(increment_sums.items()):
        out.append({
            "model": tag,
            "metric": "mean_increment",
            "validity": validity,
            "ordering": "",
            "position": position,
            "value": format_value(total / increment_counts[(validity, position)]),
        })
    return out


def pearson(pairs: list[tuple[float, float]]) -> float:
    n = len(pairs)
    if n < 2:
        return float("nan")
    mean_x = sum(pair[0] for pair in pairs) / n
    mean_y = sum(pair[1] for pair in pairs) / n
    sx = math.sqrt(sum((pair[0] - mean_x) ** 2 for pair in pairs))
    sy = math.sqrt(sum((pair[1] - mean_y) ** 2 for pair in pairs))
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((pair[0] - mean_x) * (pair[1] - mean_y) for pair in pairs) / (sx * sy)


def run_variance_rows(
    tag: str,
    model_index: int,
    rows: list[dict[str, str]],
    records: list[dict[str, object]],
    bt: dict[tuple[str, str, str, str], float],
    center: float,
    scale: float,
    sigmas: dict[str, float],
    delta: float,
    n_bootstrap: int,
    seed: int,
) -> list[dict[str, object]]:
    R_GRID = (2, 5, 10)
    N_SUBSAMPLES = 200
    N_CI_SUBSAMPLES = 30
    CI_BOOTSTRAP = 400
    WITHIN_BOOTSTRAP = 400
    DIRECTION_SIMS = 2000
    DIRECTION_RUNS = 5

    out: list[dict[str, object]] = []

    def add(metric: str, value: float, variant: str = "", validity: str = "", r_runs: object = "", artefact: str = "") -> None:
        out.append({
            "model": tag,
            "metric": metric,
            "variant": variant,
            "validity": validity,
            "r_runs": r_runs,
            "artefact": artefact,
            "value": format_value(value),
        })

    runs_by_artefact_sets = defaultdict(set)
    for row in rows:
        runs_by_artefact_sets[row["artefact"]].add(row["run"])
    runs_by_artefact = {artefact: sorted(runs) for artefact, runs in runs_by_artefact_sets.items()}
    max_runs = max(len(runs) for runs in runs_by_artefact.values())

    full_scores = j_scores(run_samples(rows, "t1"), delta)
    full_by_variant = {"bt_weighted": bt_weighted_scores(records)["ads"], "unweighted": full_scores["ads"]}

    def subsample(rng: random.Random, r: int) -> list[dict[str, str]]:
        keep = {artefact: set(rng.sample(runs, min(r, len(runs)))) for artefact, runs in runs_by_artefact.items()}
        return [row for row in rows if row["run"] in keep[row["artefact"]]]

    for r in R_GRID:
        rng = random.Random(seed + 10_000 * r + model_index)
        values: dict[str, list[float]] = {"bt_weighted": [], "unweighted": []}
        for _ in range(N_SUBSAMPLES):
            sub = subsample(rng, r)
            values["unweighted"].append(j_scores(run_samples(sub, "t1"), delta)["ads"])
            sub_records = argument_point_rows(sub, bt, "t1", center, scale, delta, sigmas)
            values["bt_weighted"].append(bt_weighted_scores(sub_records)["ads"])
        for variant, vals in values.items():
            add("subsample_point_sd", sample_sd(vals), variant=variant, r_runs=r)
            add("subsample_point_bias", sum(vals) / len(vals) - full_by_variant[variant], variant=variant, r_runs=r)

    if n_bootstrap > 0:
        rng = random.Random(seed + 5_000 + model_index)
        widths: dict[str, list[float]] = {"bt_weighted": [], "unweighted": []}
        for rep in range(N_CI_SUBSAMPLES):
            sub = subsample(rng, DIRECTION_RUNS)
            ci = j_bootstrap(j_scores(run_samples(sub, "t1"), delta)["per_artefact"], CI_BOOTSTRAP, seed + rep)
            widths["unweighted"].append(ci["ads"][1] - ci["ads"][0])
            sub_records = argument_point_rows(sub, bt, "t1", center, scale, delta, sigmas)
            ci = j_bootstrap(bt_weighted_scores(sub_records)["per_artefact"], CI_BOOTSTRAP, seed + rep)
            widths["bt_weighted"].append(ci["ads"][1] - ci["ads"][0])
        for variant, vals in widths.items():
            add("subsample_ci_width_mean", sum(vals) / len(vals), variant=variant, r_runs=DIRECTION_RUNS)
        full_ci = {
            "bt_weighted": j_bootstrap(bt_weighted_scores(records)["per_artefact"], n_bootstrap, seed),
            "unweighted": j_bootstrap(full_scores["per_artefact"], n_bootstrap, seed),
        }
        for variant, ci in full_ci.items():
            add("full_ci_width", ci["ads"][1] - ci["ads"][0], variant=variant, r_runs=max_runs)

    rows_by_artefact = defaultdict(list)
    for row in rows:
        rows_by_artefact[row["artefact"]].append(row)
    d_values = [100.0 * (rates[0] - rates[1]) for rates in full_scores["per_artefact"].values()]
    var_observed = sample_sd(d_values) ** 2
    rng = random.Random(seed + 4242)
    within_vars = []
    for _, artefact_rows in sorted(rows_by_artefact.items()):
        by_run = defaultdict(list)
        for row in artefact_rows:
            by_run[row["run"]].append(row)
        labels = sorted(by_run)
        reps = []
        for _ in range(WITHIN_BOOTSTRAP):
            valid: list[float] = []
            invalid: list[float] = []
            for label in (rng.choice(labels) for _ in labels):
                for row in by_run[label]:
                    shift = SIGN[row["direction"]] * (float(row["t1"]) - float(row["S0"]))
                    (valid if row["validity"] == "valid" else invalid).append(1.0 if shift >= delta else 0.0)
            reps.append(100.0 * (sum(valid) / len(valid) - sum(invalid) / len(invalid)))
        within_vars.append(sample_sd(reps) ** 2)
    var_within = sum(within_vars) / len(within_vars)
    var_true = max(var_observed - var_within, 0.0)
    add("var_between_artefact_observed", var_observed, variant="unweighted")
    add("var_within_run_noise", var_within, variant="unweighted")
    add("run_noise_share", var_within / var_observed, variant="unweighted")
    add("var_between_artefact_true", var_true, variant="unweighted")
    for r in R_GRID:
        add(
            "predicted_ci_inflation",
            math.sqrt((var_true + var_within * max_runs / r) / var_observed),
            variant="unweighted",
            r_runs=r,
        )

    for validity in ("valid", "invalid"):
        cells = defaultdict(list)
        for record in records:
            if record["validity"] == validity:
                cells[str(record["artefact"])].append(float(record["update_rate"]))
        sds = [sample_sd(values) for values in cells.values() if len(values) >= 2]
        binom = [rate * (1.0 - rate) for values in cells.values() for rate in values]
        add("between_argument_sd", sum(sds) / len(sds), validity=validity)
        add("mean_binomial_var", sum(binom) / len(binom), validity=validity)

    correlations = []
    for _, artefact_rows in sorted(rows_by_artefact.items()):
        per_run: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"valid": [], "invalid": []})
        for row in artefact_rows:
            shift = SIGN[row["direction"]] * (float(row["t1"]) - float(row["S0"]))
            per_run[row["run"]][row["validity"]].append(1.0 if shift >= delta else 0.0)
        pairs = [
            (sum(groups["valid"]) / len(groups["valid"]), sum(groups["invalid"]) / len(groups["invalid"]))
            for groups in per_run.values()
            if groups["valid"] and groups["invalid"]
        ]
        corr = pearson(pairs)
        if math.isfinite(corr):
            correlations.append(corr)
    add("s0_coupling_corr", sum(correlations) / len(correlations) if correlations else float("nan"))

    s0_by_artefact: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        s0_by_artefact[row["artefact"]][row["run"]] = float(row["S0"])
    rng = random.Random(seed + 31337)
    for artefact, by_run in sorted(s0_by_artefact.items()):
        scores = list(by_run.values())
        mean_full = sum(scores) / len(scores)
        flips = 0
        for _ in range(DIRECTION_SIMS):
            sub = rng.sample(scores, min(DIRECTION_RUNS, len(scores)))
            if ((sum(sub) / len(sub)) > 50.0) != (mean_full > 50.0):
                flips += 1
        if flips:
            add("direction_flip_risk", flips / DIRECTION_SIMS, r_runs=DIRECTION_RUNS, artefact=artefact)

    return out


def curve_per_artefact(rows: list[dict[str, str]], sigmas: dict[str, float]) -> dict[tuple[str, str], tuple[float, ...]]:
    sums: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0])
    for row in rows:
        sigma = sigmas[row["artefact"]]
        key = (row["validity"], row["artefact"])
        for pos, horizon in enumerate(HORIZONS):
            z = SIGN[row["direction"]] * (float(row[horizon]) - float(row["S0"])) / sigma
            sums[key][pos] += z
            counts[key][pos] += 1
    return {
        key: tuple(sums[key][pos] / counts[key][pos] for pos in range(3))
        for key in sums
    }


def weighted_curve_per_artefact(
    rows: list[dict[str, str]],
    records: list[dict[str, object]],
    sigmas: dict[str, float],
) -> dict[tuple[str, str], tuple[float, ...]]:
    weights = {
        (str(record["artefact"]), str(record["validity"]), str(record["idx"])): abs(float(record["x"]))
        for record in records
    }
    weight_sums: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    weighted_z_sums: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    for row in rows:
        sigma = sigmas[row["artefact"]]
        key = (row["validity"], row["artefact"])
        for pos, horizon in enumerate(HORIZONS):
            seen = row["ordering"][:pos + 1]
            weight = sum(weights[(row["artefact"], row["validity"], idx)] for idx in seen) / len(seen)
            z = SIGN[row["direction"]] * (float(row[horizon]) - float(row["S0"])) / sigma
            weight_sums[key][pos] += weight
            weighted_z_sums[key][pos] += weight * z
    return {
        key: tuple(weighted_z_sums[key][pos] / weight_sums[key][pos] for pos in range(3))
        for key in weight_sums
        if min(weight_sums[key]) > 0
    }


def curve_bootstrap(
    per_artefact: dict[str, tuple[float, ...]],
    n_bootstrap: int,
    seed: int,
) -> list[tuple[float, float]]:
    rng = random.Random(seed)
    artefacts = sorted(per_artefact)
    samples: list[list[float]] = [[], [], []]
    for _ in range(n_bootstrap):
        draw = [per_artefact[rng.choice(artefacts)] for _ in artefacts]
        for pos in range(3):
            samples[pos].append(sum(curve[pos] for curve in draw) / len(draw))
    return [(percentile(values, 2.5), percentile(values, 97.5)) for values in samples]


def format_value(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.10g}"


def summary_row(
    tag: str,
    horizon: str,
    variant: str,
    delta: float,
    scores: dict[str, object],
    intervals: dict[str, tuple[float, float]] | None,
) -> dict[str, object]:
    nan_ci = (float("nan"), float("nan"))
    row = {
        "model": tag,
        "horizon": horizon,
        "variant": variant,
        "delta": format_value(delta),
        "n_artefacts": len(scores["per_artefact"]),
        **{field: format_value(scores[field]) for field in J_FIELDS},
    }
    for field in J_FIELDS:
        ci = intervals[field] if intervals else nan_ci
        row[f"{field}_ci_low"] = format_value(ci[0])
        row[f"{field}_ci_high"] = format_value(ci[1])
    return row


def build_outputs(
    trajectory_paths: list[Path],
    bt_path: Path,
    delta: float,
    sensitivity_deltas: list[float],
    trim: float,
    sigma_floor: float,
    n_bootstrap: int,
    seed: int,
) -> tuple[list[dict[str, object]], ...]:
    bt = load_bt(bt_path)
    summary_rows = []
    point_rows = []
    curve_rows = []
    dose_rows = []
    sensitivity_rows = []
    zero_rows = []
    order_rows = []
    variance_rows = []
    for model_index, path in enumerate(trajectory_paths):
        tag = trajectory_tag(path)
        rows = read_csv(path)
        center, scale = bt_center_scale(rows, bt)
        sigmas = s0_sigmas(rows, sigma_floor)
        records = argument_point_rows(rows, bt, "t1", center, scale, delta, sigmas)

        weighted = bt_weighted_scores(records)
        weighted_ci = j_bootstrap(weighted["per_artefact"], n_bootstrap, seed) if n_bootstrap > 0 else None
        summary_rows.append(summary_row(tag, "t1", "bt_weighted", delta, weighted, weighted_ci))

        for horizon in HORIZONS:
            samples = run_samples(rows, horizon)
            scores = j_scores(samples, delta)
            intervals = j_bootstrap(scores["per_artefact"], n_bootstrap, seed) if n_bootstrap > 0 else None
            summary_rows.append(summary_row(tag, horizon, "unweighted", delta, scores, intervals))

        t1_samples = run_samples(rows, "t1")
        two_sided = two_sided_scores(t1_samples, delta)
        two_ci = j_bootstrap(two_sided["per_artefact"], n_bootstrap, seed) if n_bootstrap > 0 else None
        summary_rows.append(summary_row(tag, "t1", "two_sided_invalid", delta, two_sided, two_ci))

        drop_records = [record for record in records if record["artefact"] != CROSSOVER_ARTEFACT]
        drop_weighted = bt_weighted_scores(drop_records)
        drop_weighted_ci = j_bootstrap(drop_weighted["per_artefact"], n_bootstrap, seed) if n_bootstrap > 0 else None
        summary_rows.append(summary_row(tag, "t1", "drop_s01_bt_weighted", delta, drop_weighted, drop_weighted_ci))

        drop_samples = [sample for sample in t1_samples if sample[0] != CROSSOVER_ARTEFACT]
        drop_unweighted = j_scores(drop_samples, delta)
        drop_unweighted_ci = j_bootstrap(drop_unweighted["per_artefact"], n_bootstrap, seed) if n_bootstrap > 0 else None
        summary_rows.append(summary_row(tag, "t1", "drop_s01_unweighted", delta, drop_unweighted, drop_unweighted_ci))

        kept_rows, n_trimmed = trimmed_rows(rows, records, trim)
        trimmed = j_scores(run_samples(kept_rows, "t1"), delta)
        trimmed_ci = j_bootstrap(trimmed["per_artefact"], n_bootstrap, seed) if n_bootstrap > 0 else None
        trimmed_row = summary_row(tag, "t1", "boundary_trimmed", delta, trimmed, trimmed_ci)
        trimmed_row["n_trimmed_arguments"] = n_trimmed
        summary_rows.append(trimmed_row)

        for record in records:
            point_rows.append({
                "model": tag,
                **{key: (format_value(value) if isinstance(value, float) else value) for key, value in record.items()},
            })

        zero_rows.extend(zero_weight_rows(tag, records))
        order_rows.extend(order_effect_rows(tag, rows))
        variance_rows.extend(
            run_variance_rows(tag, model_index, rows, records, bt, center, scale, sigmas, delta, n_bootstrap, seed)
        )

        curve_variants = (
            ("unweighted", curve_per_artefact(rows, sigmas)),
            ("bt_weighted", weighted_curve_per_artefact(rows, records, sigmas)),
        )
        for variant, curves in curve_variants:
            for validity in ("valid", "invalid"):
                per_artefact = {
                    artefact: values
                    for (val, artefact), values in curves.items()
                    if val == validity
                }
                means = [
                    sum(values[pos] for values in per_artefact.values()) / len(per_artefact)
                    for pos in range(3)
                ]
                intervals = curve_bootstrap(per_artefact, n_bootstrap, seed) if n_bootstrap > 0 else [(float("nan"), float("nan"))] * 3
                curve_rows.append({
                    "model": tag,
                    "variant": variant,
                    "validity": validity,
                    "turn": 0,
                    "mean_z": format_value(0.0),
                    "ci_low": format_value(0.0),
                    "ci_high": format_value(0.0),
                    "n_artefacts": len(per_artefact),
                })
                for pos in range(3):
                    curve_rows.append({
                        "model": tag,
                        "variant": variant,
                        "validity": validity,
                        "turn": pos + 1,
                        "mean_z": format_value(means[pos]),
                        "ci_low": format_value(intervals[pos][0]),
                        "ci_high": format_value(intervals[pos][1]),
                        "n_artefacts": len(per_artefact),
                    })

        for validity, sign in (("valid", 1.0), ("invalid", -1.0)):
            pairs = [
                (sign * float(record["x"]), float(record["update_rate"]))
                for record in records
                if record["validity"] == validity
            ]
            dose_rows.append({
                "model": tag,
                "horizon": "t1",
                "pool": validity,
                "n_arguments": len(pairs),
                "spearman_rho": format_value(spearman(pairs)),
            })

        for sensitivity_delta in sensitivity_deltas:
            for horizon in HORIZONS:
                scores = j_scores(run_samples(rows, horizon), sensitivity_delta)
                sensitivity_rows.append({
                    "model": tag,
                    "horizon": horizon,
                    "variant": "unweighted",
                    "delta": format_value(sensitivity_delta),
                    **{field: format_value(scores[field]) for field in J_FIELDS},
                })
            sensitivity_records = argument_point_rows(rows, bt, "t1", center, scale, sensitivity_delta, sigmas)
            sensitivity_weighted = bt_weighted_scores(sensitivity_records)
            sensitivity_rows.append({
                "model": tag,
                "horizon": "t1",
                "variant": "bt_weighted",
                "delta": format_value(sensitivity_delta),
                **{field: format_value(sensitivity_weighted[field]) for field in J_FIELDS},
            })
    return summary_rows, point_rows, curve_rows, dose_rows, sensitivity_rows, zero_rows, order_rows, variance_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, action="append")
    parser.add_argument("--bt-scores", type=Path, default=BT_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--delta", type=float, default=5.0)
    parser.add_argument("--delta-sensitivity", type=str, default="1,2,5,10")
    parser.add_argument("--trim", type=float, default=0.25)
    parser.add_argument("--sigma-floor", type=float, default=1.0)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    trajectories = discover_trajectories(args.trajectory)
    sensitivity_deltas = [float(value) for value in args.delta_sensitivity.split(",") if value]
    config = {
        "trajectories": [str(path) for path in trajectories],
        "bt_scores": str(args.bt_scores),
        "delta": args.delta,
        "delta_sensitivity": sensitivity_deltas,
        "trim": args.trim,
        "sigma_floor": args.sigma_floor,
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "crossover_artefact": CROSSOVER_ARTEFACT,
    }
    print(json.dumps(config, indent=2))

    summary_rows, point_rows, curve_rows, dose_rows, sensitivity_rows, zero_rows, order_rows, variance_rows = build_outputs(
        trajectories,
        args.bt_scores,
        args.delta,
        sensitivity_deltas,
        args.trim,
        args.sigma_floor,
        args.bootstrap,
        args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_fields = [
        "model", "horizon", "variant", "delta", "n_artefacts",
        "tpr", "fpr", "ads",
        "tpr_ci_low", "tpr_ci_high", "fpr_ci_low", "fpr_ci_high", "ads_ci_low", "ads_ci_high",
        "n_trimmed_arguments",
    ]
    write_csv(args.output_dir / "ads2_summary.csv", summary_rows, summary_fields)
    write_csv(args.output_dir / "ads2_argument_points.csv", point_rows, list(point_rows[0]))
    write_csv(args.output_dir / "ads2_turn_curves.csv", curve_rows, list(curve_rows[0]))
    write_csv(args.output_dir / "ads2_dose_response.csv", dose_rows, list(dose_rows[0]))
    write_csv(args.output_dir / "ads2_delta_sensitivity.csv", sensitivity_rows, list(sensitivity_rows[0]))
    write_csv(args.output_dir / "ads2_zero_weight.csv", zero_rows, list(zero_rows[0]))
    write_csv(args.output_dir / "ads2_order_effects.csv", order_rows, list(order_rows[0]))
    write_csv(args.output_dir / "ads2_run_variance.csv", variance_rows, list(variance_rows[0]))
    with (args.output_dir / "ads2_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")
    print(args.output_dir / "ads2_summary.csv")


if __name__ == "__main__":
    main()
