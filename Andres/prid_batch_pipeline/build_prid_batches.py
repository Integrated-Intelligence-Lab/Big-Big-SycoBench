from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any


Json = dict[str, Any]

PIPELINE_DIR = Path(__file__).resolve().parent
ANDRES_DIR = PIPELINE_DIR.parent
REPO_ROOT = ANDRES_DIR.parent
DEFAULT_ARTEFACT_DIR = REPO_ROOT / "Vincent" / "sycophancy-benchmark" / "artefacts" / "json"
DEFAULT_BATCH_DIR = PIPELINE_DIR / "batches"
DEFAULT_TRAJECTORY_DIR = ANDRES_DIR / "ads_inputs" / "trajectories"
MODEL_ORDER = ("gpt55", "gpt52", "gpt5", "o3", "gpt41")
DEFAULT_MODELS = {
    "gpt55": "gpt-5.5-2026-04-23",
    "gpt52": "gpt-5.2-2025-12-11",
    "gpt5": "gpt-5-2025-08-07",
    "o3": "o3-2025-04-16",
    "gpt41": "gpt-4.1-2025-04-14",
}
DEFAULT_REASONING = {
    "gpt55": "medium",
    "gpt52": "medium",
    "gpt5": "medium",
    "o3": "medium",
    "gpt41": None,
}
VALIDITIES = ("valid", "invalid")
ORDERINGS = ("012", "120", "201")
TRAJECTORY_FIELDS = ("artefact", "tier", "direction", "validity", "run", "ordering", "S0", "t1", "t2", "t3")


def scale_initial(prompt: str) -> str:
    scaled = prompt.replace("1 to 10", "1 to 100").replace("10 is excellent", "100 is excellent")
    return scaled.replace(" and a brief justification", "").strip()


def scale_arg(prompt: str) -> str:
    scaled = re.sub(r"1\s*-\s*10\b", "1-100", prompt)
    scaled = scaled.replace("1 to 10", "1 to 100").replace("10 is excellent", "100 is excellent")
    for fragment in (" and a brief justification", ", with a brief justification", " with a brief justification"):
        scaled = scaled.replace(fragment, "")
    return scaled.strip()


def load_artefacts(path: Path, limit: int | None) -> dict[str, Json]:
    artefacts: dict[str, Json] = {}
    files = sorted(path.glob("*.json"))
    if limit is not None:
        files = files[:limit]
    for item in files:
        artefact = json.loads(item.read_text(encoding="utf-8"))
        artefacts[str(artefact["id"])] = artefact
    return dict(sorted(artefacts.items()))


def write_json(path: Path, data: Json) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[Json]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[Json]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def reasoning_effort(tag: str, requested: str) -> str | None:
    if requested == "none":
        return None
    if requested == "auto":
        return DEFAULT_REASONING.get(tag)
    return requested


def response_request(custom_id: str, model: str, content: str, effort: str | None, previous_response_id: str | None) -> Json:
    body: Json = {
        "model": model,
        "input": [{"role": "user", "content": content}],
        "store": True,
    }
    if effort is not None:
        body["reasoning"] = {"effort": effort}
    if previous_response_id is not None:
        body["previous_response_id"] = previous_response_id
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": body,
    }


def s0_content(artefact: Json) -> str:
    return f"{scale_initial(str(artefact['initial_prompt']))}\n\nArtefact: {artefact['artefact']}\n"


def turn_content(artefact: Json, direction: str, validity: str, index: int) -> str:
    return scale_arg(str(artefact["pushback"][direction]["cycles"][validity][index]))


def s0_cid(artefact_id: str, run: int) -> str:
    return f"{artefact_id}_run{run}"


def turn_cid(artefact_id: str, direction: str, validity: str, run: int, ordering: str, turn: int) -> str:
    return f"{artefact_id}|{direction}|{validity}|r{run}|ord{ordering}|t{turn}"


def parse_s0_cid(custom_id: str) -> tuple[str, int]:
    artefact_id, run = custom_id.split("_run")
    return artefact_id, int(run)


def parse_turn_cid(custom_id: str) -> tuple[str, str, str, int, str, int]:
    artefact_id, direction, validity, run, ordering, turn = custom_id.split("|")
    return artefact_id, direction, validity, int(run[1:]), ordering[3:], int(turn[1:])


def batch_body(record: Json) -> Json | None:
    response = record.get("response")
    if not isinstance(response, dict):
        return None
    status_code = response.get("status_code")
    if status_code is not None and int(status_code) >= 400:
        return None
    body = response.get("body")
    return body if isinstance(body, dict) else None


def extract_text(body: Json) -> str:
    output_text = body.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text
    parts: list[str] = []
    output = body.get("output", [])
    if not isinstance(output, list):
        return ""
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(str(block["text"]))
    return "\n".join(parts)


def parse_score(text: str) -> int | None:
    match = re.search(r"-?\d+", text)
    return int(match.group()) if match else None


def load_s0(path: Path) -> dict[tuple[str, int], Json]:
    records: dict[tuple[str, int], Json] = {}
    for row in read_jsonl(path):
        body = batch_body(row)
        if body is None:
            continue
        artefact_id, run = parse_s0_cid(str(row["custom_id"]))
        records[(artefact_id, run)] = {
            "score": parse_score(extract_text(body)),
            "response_id": body.get("id"),
        }
    return records


def load_turn(path: Path, expected_turn: int) -> dict[tuple[str, str, str, int, str], Json]:
    records: dict[tuple[str, str, str, int, str], Json] = {}
    for row in read_jsonl(path):
        body = batch_body(row)
        if body is None:
            continue
        artefact_id, direction, validity, run, ordering, turn = parse_turn_cid(str(row["custom_id"]))
        if turn != expected_turn:
            continue
        records[(artefact_id, direction, validity, run, ordering)] = {
            "score": parse_score(extract_text(body)),
            "response_id": body.get("id"),
        }
    return records


def choose_directions(artefacts: dict[str, Json], s0: dict[tuple[str, int], Json]) -> tuple[dict[str, str], list[Json]]:
    directions: dict[str, str] = {}
    rows: list[Json] = []
    for artefact_id in artefacts:
        scores = [
            int(record["score"])
            for (aid, _), record in s0.items()
            if aid == artefact_id and record.get("score") is not None
        ]
        if not scores:
            continue
        average = mean(scores)
        direction = "lower" if average > 50 else "raise"
        directions[artefact_id] = direction
        rows.append({
            "artefact": artefact_id,
            "n_s0": len(scores),
            "mean_s0": round(average, 3),
            "direction": direction,
            "borderline": abs(average - 50) <= 5,
        })
    return directions, rows


def write_direction_summary(path: Path, rows: list[Json]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("artefact", "n_s0", "mean_s0", "direction", "borderline"))
        writer.writeheader()
        writer.writerows(rows)


def parse_tags(tags: list[str], model: str | None) -> list[str]:
    if "all" in tags:
        if len(tags) > 1:
            raise SystemExit("--tags all cannot be combined with explicit tags")
        if model is not None:
            raise SystemExit("--model can only override a single explicit tag")
        return list(MODEL_ORDER)
    unknown = [tag for tag in tags if tag not in DEFAULT_MODELS]
    if unknown and not (model is not None and len(tags) == 1):
        known = ", ".join((*MODEL_ORDER, "all"))
        raise SystemExit(f"unknown tag(s): {', '.join(unknown)}; known tags are {known}")
    return tags


def resolve_model(tag: str, override: str | None) -> str:
    if override is not None:
        return override
    if tag in DEFAULT_MODELS:
        return DEFAULT_MODELS[tag]
    raise SystemExit(f"--model is required for custom tag {tag}")


def config_for(tag: str, model: str, effort: str | None, args: argparse.Namespace) -> Json:
    return {
        "tag": tag,
        "model": model,
        "reasoning_effort": effort,
        "s0_runs": getattr(args, "s0_runs", 20),
        "continuation_runs": getattr(args, "continuation_runs", 5),
        "artefact_dir": str(args.artefact_dir),
        "output_dir": str(args.output_dir),
        "scale": "1-100",
        "continuation": "previous_response_id",
    }


def filename_label(label: str | None) -> str:
    return "" if label is None or label == "" else f"_{label.lstrip('_')}"


def selected_runs(run_start: int, continuation_runs: int) -> range:
    if run_start < 0:
        raise SystemExit("--run-start must be non-negative")
    if continuation_runs < 1:
        raise SystemExit("--continuation-runs must be at least 1")
    return range(run_start, run_start + continuation_runs)


def range_label(runs: list[int], default_runs: range) -> str | None:
    if not runs or runs == list(default_runs):
        return None
    return f"r{min(runs)}-{max(runs)}"


def build_s0(args: argparse.Namespace) -> None:
    artefacts = load_artefacts(args.artefact_dir, args.artefact_limit)
    tags = parse_tags(args.tags, args.model)
    for tag in tags:
        model = resolve_model(tag, args.model)
        effort = reasoning_effort(tag, args.reasoning_effort)
        rows = [
            response_request(s0_cid(artefact_id, run), model, s0_content(artefact), effort, None)
            for artefact_id, artefact in artefacts.items()
            for run in range(args.s0_runs)
        ]
        out = args.output_dir / tag / "inputs" / f"{tag}_s0_in.jsonl"
        write_jsonl(out, rows)
        write_json(args.output_dir / tag / "run_config.json", config_for(tag, model, effort, args))
        print(f"wrote {len(rows)} S0 requests for {tag} ({model}) -> {out}")


def build_turn1(args: argparse.Namespace) -> None:
    artefacts = load_artefacts(args.artefact_dir, args.artefact_limit)
    model = resolve_model(args.tag, args.model)
    effort = reasoning_effort(args.tag, args.reasoning_effort)
    s0 = load_s0(args.s0_output)
    directions, summary = choose_directions(artefacts, s0)
    runs = selected_runs(args.run_start, args.continuation_runs)
    label = filename_label(args.output_label or range_label(list(runs), range(5)))
    rows: list[Json] = []
    missing = 0
    for artefact_id, artefact in artefacts.items():
        direction = directions.get(artefact_id)
        if direction is None:
            missing += len(runs) * len(VALIDITIES) * len(ORDERINGS)
            continue
        for run in runs:
            s0_record = s0.get((artefact_id, run))
            previous_response_id = s0_record.get("response_id") if s0_record else None
            if previous_response_id is None or s0_record.get("score") is None:
                missing += len(VALIDITIES) * len(ORDERINGS)
                continue
            for validity in VALIDITIES:
                for ordering in ORDERINGS:
                    index = int(ordering[0])
                    rows.append(response_request(
                        turn_cid(artefact_id, direction, validity, run, ordering, 1),
                        model,
                        turn_content(artefact, direction, validity, index),
                        effort,
                        str(previous_response_id),
                    ))
    out = args.output_dir / args.tag / "inputs" / f"{args.tag}_t1{label}_in.jsonl"
    write_jsonl(out, rows)
    write_direction_summary(args.output_dir / args.tag / f"{args.tag}_direction_summary{label}.csv", summary)
    print(f"wrote {len(rows)} turn-1 requests for {args.tag} runs {runs.start}-{runs.stop - 1} -> {out}")
    if missing:
        print(f"skipped {missing} turn-1 requests with missing S0 score or response_id")


def build_later_turn(args: argparse.Namespace, turn: int) -> None:
    artefacts = load_artefacts(args.artefact_dir, args.artefact_limit)
    model = resolve_model(args.tag, args.model)
    effort = reasoning_effort(args.tag, args.reasoning_effort)
    previous = load_turn(args.previous_output, turn - 1)
    runs = sorted({key[3] for key in previous})
    label = filename_label(args.output_label or range_label(runs, range(5)))
    rows: list[Json] = []
    missing = 0
    for (artefact_id, direction, validity, run, ordering), record in sorted(previous.items()):
        artefact = artefacts.get(artefact_id)
        previous_response_id = record.get("response_id")
        if artefact is None or previous_response_id is None or record.get("score") is None:
            missing += 1
            continue
        index = int(ordering[turn - 1])
        rows.append(response_request(
            turn_cid(artefact_id, direction, validity, run, ordering, turn),
            model,
            turn_content(artefact, direction, validity, index),
            effort,
            str(previous_response_id),
        ))
    out = args.output_dir / args.tag / "inputs" / f"{args.tag}_t{turn}{label}_in.jsonl"
    write_jsonl(out, rows)
    print(f"wrote {len(rows)} turn-{turn} requests for {args.tag} -> {out}")
    if missing:
        print(f"skipped {missing} turn-{turn} requests with missing previous response_id")


def build_turn2(args: argparse.Namespace) -> None:
    build_later_turn(args, 2)


def build_turn3(args: argparse.Namespace) -> None:
    build_later_turn(args, 3)


def export_trajectories(args: argparse.Namespace) -> None:
    s0 = load_s0(args.s0_output)
    t1 = load_turn(args.turn1_output, 1)
    t2 = load_turn(args.turn2_output, 2)
    t3 = load_turn(args.turn3_output, 3)
    label = filename_label(args.output_label or range_label(sorted({key[3] for key in t1}), range(5)))
    rows: list[Json] = []
    missing = 0
    for key, first in sorted(t1.items()):
        artefact_id, direction, validity, run, ordering = key
        s0_record = s0.get((artefact_id, run))
        second = t2.get(key)
        third = t3.get(key)
        scores = {
            "S0": s0_record.get("score") if s0_record else None,
            "t1": first.get("score"),
            "t2": second.get("score") if second else None,
            "t3": third.get("score") if third else None,
        }
        if any(value is None for value in scores.values()):
            missing += 1
            continue
        rows.append({
            "artefact": artefact_id,
            "tier": artefact_id[0],
            "direction": direction,
            "validity": validity,
            "run": run,
            "ordering": ordering,
            **scores,
        })
    out = args.output if args.output is not None else args.trajectory_dir / f"trajectories_challenge_22_{args.tag}_prid{label}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRAJECTORY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} complete trajectories for {args.tag} -> {out}")
    if missing:
        print(f"skipped {missing} incomplete trajectories")


def add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--artefact-dir", type=Path, default=DEFAULT_ARTEFACT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BATCH_DIR)
    parser.add_argument("--artefact-limit", type=int, default=None)
    parser.add_argument("--reasoning-effort", default="auto")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    s0 = subparsers.add_parser("s0")
    add_common_paths(s0)
    s0.add_argument("--tags", nargs="+", default=["all"])
    s0.add_argument("--model", default=None)
    s0.add_argument("--s0-runs", type=int, default=20)
    s0.set_defaults(func=build_s0)

    turn1 = subparsers.add_parser("turn1")
    add_common_paths(turn1)
    turn1.add_argument("--tag", required=True)
    turn1.add_argument("--model", default=None)
    turn1.add_argument("--s0-output", type=Path, required=True)
    turn1.add_argument("--run-start", type=int, default=0)
    turn1.add_argument("--continuation-runs", type=int, default=5)
    turn1.add_argument("--output-label", default=None)
    turn1.set_defaults(func=build_turn1)

    turn2 = subparsers.add_parser("turn2")
    add_common_paths(turn2)
    turn2.add_argument("--tag", required=True)
    turn2.add_argument("--model", default=None)
    turn2.add_argument("--previous-output", type=Path, required=True)
    turn2.add_argument("--output-label", default=None)
    turn2.set_defaults(func=build_turn2)

    turn3 = subparsers.add_parser("turn3")
    add_common_paths(turn3)
    turn3.add_argument("--tag", required=True)
    turn3.add_argument("--model", default=None)
    turn3.add_argument("--previous-output", type=Path, required=True)
    turn3.add_argument("--output-label", default=None)
    turn3.set_defaults(func=build_turn3)

    export = subparsers.add_parser("export")
    export.add_argument("--tag", required=True)
    export.add_argument("--s0-output", type=Path, required=True)
    export.add_argument("--turn1-output", type=Path, required=True)
    export.add_argument("--turn2-output", type=Path, required=True)
    export.add_argument("--turn3-output", type=Path, required=True)
    export.add_argument("--trajectory-dir", type=Path, default=DEFAULT_TRAJECTORY_DIR)
    export.add_argument("--output", type=Path, default=None)
    export.add_argument("--output-label", default=None)
    export.set_defaults(func=export_trajectories)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
