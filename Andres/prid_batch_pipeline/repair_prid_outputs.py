from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import build_prid_batches as builder
import openai_batch_io as batch_io
import run_prid_pipeline as runner


Json = dict[str, Any]


def read_jsonl(path: Path) -> list[Json]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[Json]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def load_stage(output_dir: Path, tag: str, label: str) -> tuple[dict[tuple[str, int], Json], dict[tuple[str, str, str, int, str], Json], dict[tuple[str, str, str, int, str], Json], dict[tuple[str, str, str, int, str], Json]]:
    s0 = builder.load_s0(runner.output_path(output_dir, tag, "s0", label))
    t1 = builder.load_turn(runner.output_path(output_dir, tag, "t1", label), 1)
    t2 = builder.load_turn(runner.output_path(output_dir, tag, "t2", label), 2)
    t3 = builder.load_turn(runner.output_path(output_dir, tag, "t3", label), 3)
    return s0, t1, t2, t3


def usable(record: Json | None) -> bool:
    return record is not None and record.get("score") is not None and record.get("response_id") is not None


def score_only_content(content: str, strict: bool) -> str:
    if not strict:
        return content
    return content + "\n\nFor this repair, provide only the revised score as a single integer on the same scale. Do not include words, punctuation, explanation, or a question."


def turn_request(tag: str, artefacts: dict[str, Json], key: tuple[str, str, str, int, str], turn: int, previous_response_id: str, effort: str | None, strict: bool) -> Json:
    artefact_id, direction, validity, run, ordering = key
    index = int(ordering[turn - 1])
    return builder.response_request(
        builder.turn_cid(artefact_id, direction, validity, run, ordering, turn),
        builder.resolve_model(tag, None),
        score_only_content(builder.turn_content(artefacts[artefact_id], direction, validity, index), strict),
        effort,
        previous_response_id,
    )


def repair_turn1_requests(tag: str, artefacts: dict[str, Json], s0: dict[tuple[str, int], Json], t1: dict[tuple[str, str, str, int, str], Json], args: argparse.Namespace) -> list[Json]:
    directions, _ = builder.choose_directions(artefacts, s0)
    runs = builder.selected_runs(args.run_start, args.continuation_runs)
    effort = builder.reasoning_effort(tag, args.reasoning_effort)
    requests: list[Json] = []
    for artefact_id in artefacts:
        direction = directions.get(artefact_id)
        if direction is None:
            continue
        for run in runs:
            previous = s0.get((artefact_id, run))
            if not usable(previous):
                continue
            for validity in builder.VALIDITIES:
                for ordering in builder.ORDERINGS:
                    key = (artefact_id, direction, validity, run, ordering)
                    if usable(t1.get(key)):
                        continue
                    requests.append(turn_request(tag, artefacts, key, 1, str(previous["response_id"]), effort, args.strict_score_repair))
    return requests


def repair_later_requests(tag: str, artefacts: dict[str, Json], previous: dict[tuple[str, str, str, int, str], Json], current: dict[tuple[str, str, str, int, str], Json], turn: int, args: argparse.Namespace) -> list[Json]:
    effort = builder.reasoning_effort(tag, args.reasoning_effort)
    requests: list[Json] = []
    for key, record in sorted(previous.items()):
        if not usable(record) or usable(current.get(key)):
            continue
        artefact_id = key[0]
        if artefact_id not in artefacts:
            continue
        requests.append(turn_request(tag, artefacts, key, turn, str(record["response_id"]), effort, args.strict_score_repair))
    return requests


def repair_requests(tag: str, turn: int, label: str, args: argparse.Namespace) -> list[Json]:
    artefacts = builder.load_artefacts(args.artefact_dir, args.artefact_limit)
    s0, t1, t2, t3 = load_stage(args.output_dir, tag, label)
    if turn == 1:
        return repair_turn1_requests(tag, artefacts, s0, t1, args)
    if turn == 2:
        return repair_later_requests(tag, artefacts, t1, t2, 2, args)
    return repair_later_requests(tag, artefacts, t2, t3, 3, args)


def repair_dir(output_dir: Path, tag: str) -> Path:
    return output_dir / tag / "repairs"


def repair_path(output_dir: Path, tag: str, turn: int, label: str, stem: str) -> Path:
    return repair_dir(output_dir, tag) / f"{tag}_repair_t{turn}{runner.suffix(label)}_{stem}"


def submit_repair(client: Any, tag: str, turn: int, label: str, requests: list[Json], args: argparse.Namespace) -> Path:
    in_path = repair_path(args.output_dir, tag, turn, label, "in.jsonl")
    out_path = repair_path(args.output_dir, tag, turn, label, "out.jsonl")
    err_path = repair_path(args.output_dir, tag, turn, label, "errors.jsonl")
    batch_path = repair_path(args.output_dir, tag, turn, label, "batch.json")
    done_path = repair_path(args.output_dir, tag, turn, label, "done.json")
    write_jsonl(in_path, requests)
    if out_path.exists() and not args.force:
        print(f"using existing repair output -> {out_path}")
        return out_path
    if batch_path.exists() and not args.force:
        batch_id = runner.batch_id_from(batch_path)
        print(f"resuming {tag} t{turn} repair: batch_id={batch_id}")
    else:
        batch_id = runner.submit_batch(client, in_path, batch_path)
    batches = runner.wait_for_batches(client, {f"{tag}:repair_t{turn}": (batch_id, done_path)}, args.poll_seconds)
    batch = batches[f"{tag}:repair_t{turn}"]
    runner.download_batch(client, batch, out_path, err_path)
    if runner.terminal_status(batch) != "completed" or runner.failed_count(batch) > 0:
        raise SystemExit(f"repair batch failed for {tag} t{turn}")
    return out_path


def merge_repair(canonical: Path, repair: Path, tag: str, turn: int) -> None:
    original = read_jsonl(canonical)
    repaired = read_jsonl(repair)
    if not repaired:
        raise SystemExit(f"empty repair output: {repair}")
    repairs = {row["custom_id"]: row for row in repaired}
    seen: set[str] = set()
    merged: list[Json] = []
    for row in original:
        custom_id = row["custom_id"]
        if custom_id in repairs:
            merged.append(repairs[custom_id])
            seen.add(custom_id)
        else:
            merged.append(row)
    for custom_id, row in repairs.items():
        if custom_id not in seen:
            merged.append(row)
    backup = canonical.parents[1] / "repairs" / f"{tag}_t{turn}_before_repair.jsonl"
    if not backup.exists():
        write_jsonl(backup, original)
    write_jsonl(canonical, merged)
    print(f"merged {len(repaired)} repair rows into {canonical}")


def incomplete_count(tag: str, label: str, args: argparse.Namespace) -> int:
    s0, t1, t2, t3 = load_stage(args.output_dir, tag, label)
    missing = 0
    for key, first in t1.items():
        artefact_id, _, _, run, _ = key
        s0_record = s0.get((artefact_id, run))
        if not usable(s0_record) or not usable(first) or not usable(t2.get(key)) or not usable(t3.get(key)):
            missing += 1
    return missing


def export_tag(tag: str, label: str, args: argparse.Namespace) -> None:
    export_args = argparse.Namespace(
        tag=tag,
        s0_output=runner.output_path(args.output_dir, tag, "s0", label),
        turn1_output=runner.output_path(args.output_dir, tag, "t1", label),
        turn2_output=runner.output_path(args.output_dir, tag, "t2", label),
        turn3_output=runner.output_path(args.output_dir, tag, "t3", label),
        trajectory_dir=args.trajectory_dir,
        output=None,
        output_label=args.output_label,
    )
    builder.export_trajectories(export_args)


def repair_tag(client: Any, tag: str, label: str, args: argparse.Namespace) -> None:
    print(f"\n=== repair {tag} ===")
    for turn in (1, 2, 3):
        requests = repair_requests(tag, turn, label, args)
        print(f"{tag} t{turn}: {len(requests)} repair requests")
        if not requests or args.dry_run:
            continue
        out = submit_repair(client, tag, turn, label, requests, args)
        canonical = runner.output_path(args.output_dir, tag, f"t{turn}", label)
        merge_repair(canonical, out, tag, turn)
    if args.dry_run:
        return
    export_tag(tag, label, args)
    missing = incomplete_count(tag, label, args)
    print(f"{tag}: incomplete trajectories after repair = {missing}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", nargs="+", default=["all"])
    parser.add_argument("--artefact-dir", type=Path, default=builder.DEFAULT_ARTEFACT_DIR)
    parser.add_argument("--output-dir", type=Path, default=builder.DEFAULT_BATCH_DIR)
    parser.add_argument("--trajectory-dir", type=Path, default=builder.DEFAULT_TRAJECTORY_DIR)
    parser.add_argument("--artefact-limit", type=int, default=None)
    parser.add_argument("--run-start", type=int, default=0)
    parser.add_argument("--continuation-runs", type=int, default=5)
    parser.add_argument("--reasoning-effort", default="auto")
    parser.add_argument("--output-label", default=None)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--env", type=Path, default=batch_io.DEFAULT_ENV)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--strict-score-repair", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tags = runner.parse_tags(args.tags)
    label = runner.continuation_label(args)
    client = None if args.dry_run else batch_io.client_from_env(args.env)
    for tag in tags:
        repair_tag(client, tag, label, args)


if __name__ == "__main__":
    main()
