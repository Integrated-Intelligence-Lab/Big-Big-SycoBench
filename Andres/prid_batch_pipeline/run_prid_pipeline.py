from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import build_prid_batches as builder
import openai_batch_io as batch_io


Json = dict[str, Any]
STAGES = ("s0", "t1", "t2", "t3")


def suffix(label: str) -> str:
    return "" if label == "" else f"_{label}"


def continuation_label(args: argparse.Namespace) -> str:
    if args.output_label is not None:
        return args.output_label.lstrip("_")
    if args.run_start == 0 and args.continuation_runs == 5:
        return ""
    return f"r{args.run_start}-{args.run_start + args.continuation_runs - 1}"


def parse_tags(tags: list[str]) -> list[str]:
    return builder.parse_tags(tags, None)


def input_path(output_dir: Path, tag: str, stage: str, label: str) -> Path:
    if stage == "s0":
        return output_dir / tag / "inputs" / f"{tag}_s0_in.jsonl"
    return output_dir / tag / "inputs" / f"{tag}_{stage}{suffix(label)}_in.jsonl"


def output_path(output_dir: Path, tag: str, stage: str, label: str) -> Path:
    if stage == "s0":
        return output_dir / tag / "outputs" / f"{tag}_s0_out.jsonl"
    return output_dir / tag / "outputs" / f"{tag}_{stage}{suffix(label)}_out.jsonl"


def error_path(output_dir: Path, tag: str, stage: str, label: str) -> Path:
    if stage == "s0":
        return output_dir / tag / "outputs" / f"{tag}_s0_errors.jsonl"
    return output_dir / tag / "outputs" / f"{tag}_{stage}{suffix(label)}_errors.jsonl"


def batch_path(output_dir: Path, tag: str, stage: str, label: str) -> Path:
    if stage == "s0":
        return output_dir / tag / f"{tag}_s0_batch.json"
    return output_dir / tag / f"{tag}_{stage}{suffix(label)}_batch.json"


def done_path(output_dir: Path, tag: str, stage: str, label: str) -> Path:
    if stage == "s0":
        return output_dir / tag / f"{tag}_s0_done.json"
    return output_dir / tag / f"{tag}_{stage}{suffix(label)}_done.json"


def batch_id_from(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    batch = data.get("batch", {})
    batch_id = batch.get("id")
    if not isinstance(batch_id, str):
        raise SystemExit(f"missing batch id in {path}")
    return batch_id


def failed_count(batch: Any) -> int:
    counts = getattr(batch, "request_counts", None)
    failed = getattr(counts, "failed", 0) if counts is not None else 0
    return int(failed or 0)


def terminal_status(batch: Any) -> str:
    return str(getattr(batch, "status", "unknown"))


def build_stage(tag: str, stage: str, label: str, args: argparse.Namespace) -> Path:
    common: Json = {
        "artefact_dir": args.artefact_dir,
        "output_dir": args.output_dir,
        "artefact_limit": args.artefact_limit,
        "reasoning_effort": args.reasoning_effort,
    }
    if stage == "s0":
        builder.build_s0(argparse.Namespace(
            **common,
            tags=[tag],
            model=None,
            s0_runs=args.s0_runs,
        ))
    elif stage == "t1":
        builder.build_turn1(argparse.Namespace(
            **common,
            tag=tag,
            model=None,
            s0_output=output_path(args.output_dir, tag, "s0", label),
            run_start=args.run_start,
            continuation_runs=args.continuation_runs,
            output_label=args.output_label,
        ))
    elif stage == "t2":
        builder.build_turn2(argparse.Namespace(
            **common,
            tag=tag,
            model=None,
            previous_output=output_path(args.output_dir, tag, "t1", label),
            output_label=args.output_label,
        ))
    elif stage == "t3":
        builder.build_turn3(argparse.Namespace(
            **common,
            tag=tag,
            model=None,
            previous_output=output_path(args.output_dir, tag, "t2", label),
            output_label=args.output_label,
        ))
    return input_path(args.output_dir, tag, stage, label)


def submit_batch(client: Any, in_path: Path, metadata_path: Path) -> str:
    batch_io.require_input(in_path)
    with in_path.open("rb") as handle:
        uploaded = client.files.create(file=handle, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window="24h",
    )
    batch_io.write_json(metadata_path, {"input_file_id": uploaded.id, "batch": batch_io.jsonable(batch)})
    print(f"submitted {in_path.name}: batch_id={batch.id}")
    return str(batch.id)


def wait_for_batches(client: Any, pending: dict[str, tuple[str, Path]], poll_seconds: int) -> dict[str, Any]:
    batches: dict[str, Any] = {}
    last: dict[str, str] = {}
    while pending:
        for key, (batch_id, done) in list(pending.items()):
            batch = client.batches.retrieve(batch_id)
            status = terminal_status(batch)
            counts = getattr(batch, "request_counts", None)
            completed = getattr(counts, "completed", "?") if counts is not None else "?"
            failed = getattr(counts, "failed", "?") if counts is not None else "?"
            total = getattr(counts, "total", "?") if counts is not None else "?"
            line = f"{key}: {status} {completed}/{total} completed, {failed} failed"
            if last.get(key) != line:
                print(line)
                last[key] = line
            if status in batch_io.TERMINAL_STATUSES:
                batch_io.write_json(done, {"batch": batch_io.jsonable(batch)})
                batches[key] = batch
                del pending[key]
        if pending:
            time.sleep(poll_seconds)
    return batches


def download_batch(client: Any, batch: Any, out_path: Path, err_path: Path) -> None:
    output_file_id = getattr(batch, "output_file_id", None)
    error_file_id = getattr(batch, "error_file_id", None)
    if output_file_id is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(batch_io.content_text(client.files.content(output_file_id)), encoding="utf-8")
        print(f"wrote output -> {out_path}")
    if error_file_id is not None:
        err_path.parent.mkdir(parents=True, exist_ok=True)
        err_path.write_text(batch_io.content_text(client.files.content(error_file_id)), encoding="utf-8")
        print(f"wrote errors -> {err_path}")


def run_stage(client: Any, tags: list[str], stage: str, label: str, args: argparse.Namespace) -> None:
    pending: dict[str, tuple[str, Path]] = {}
    skipped: list[str] = []
    for tag in tags:
        out = output_path(args.output_dir, tag, stage, label)
        if out.exists() and not args.force:
            skipped.append(tag)
            continue
        in_path = build_stage(tag, stage, label, args)
        metadata = batch_path(args.output_dir, tag, stage, label)
        done = done_path(args.output_dir, tag, stage, label)
        if metadata.exists() and not args.force:
            batch_id = batch_id_from(metadata)
            print(f"resuming {tag} {stage}: batch_id={batch_id}")
        else:
            batch_id = submit_batch(client, in_path, metadata)
        pending[f"{tag}:{stage}"] = (batch_id, done)
    if skipped:
        print(f"skipped existing {stage} outputs: {', '.join(skipped)}")
    batches = wait_for_batches(client, pending, args.poll_seconds)
    failed: list[str] = []
    for key, batch in batches.items():
        tag, _ = key.split(":", 1)
        out = output_path(args.output_dir, tag, stage, label)
        err = error_path(args.output_dir, tag, stage, label)
        download_batch(client, batch, out, err)
        if terminal_status(batch) != "completed" or failed_count(batch) > 0:
            failed.append(key)
    if failed:
        raise SystemExit(f"stopping because batch stage failed: {', '.join(failed)}")


def export_tag(tag: str, label: str, args: argparse.Namespace) -> None:
    output = None
    trajectory_dir = args.trajectory_dir
    if trajectory_dir is None and args.artefact_limit is not None:
        output = args.output_dir / tag / f"trajectories_{tag}{suffix(label)}.csv"
        trajectory_dir = builder.DEFAULT_TRAJECTORY_DIR
    elif trajectory_dir is None:
        trajectory_dir = builder.DEFAULT_TRAJECTORY_DIR
    builder.export_trajectories(argparse.Namespace(
        tag=tag,
        s0_output=output_path(args.output_dir, tag, "s0", label),
        turn1_output=output_path(args.output_dir, tag, "t1", label),
        turn2_output=output_path(args.output_dir, tag, "t2", label),
        turn3_output=output_path(args.output_dir, tag, "t3", label),
        trajectory_dir=trajectory_dir,
        output=output,
        output_label=args.output_label,
    ))


def run(args: argparse.Namespace) -> None:
    tags = parse_tags(args.tags)
    label = continuation_label(args)
    client = batch_io.client_from_env(args.env)
    print(f"tags={','.join(tags)}")
    print(f"output_dir={args.output_dir}")
    print(f"s0_runs={args.s0_runs} continuation_runs={args.continuation_runs} run_start={args.run_start}")
    for stage in STAGES:
        print(f"\n=== {stage} ===")
        run_stage(client, tags, stage, label, args)
    print("\n=== export ===")
    for tag in tags:
        export_tag(tag, label, args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", nargs="+", default=["all"])
    parser.add_argument("--artefact-dir", type=Path, default=builder.DEFAULT_ARTEFACT_DIR)
    parser.add_argument("--output-dir", type=Path, default=builder.DEFAULT_BATCH_DIR)
    parser.add_argument("--trajectory-dir", type=Path, default=None)
    parser.add_argument("--artefact-limit", type=int, default=None)
    parser.add_argument("--s0-runs", type=int, default=20)
    parser.add_argument("--run-start", type=int, default=0)
    parser.add_argument("--continuation-runs", type=int, default=5)
    parser.add_argument("--reasoning-effort", default="auto")
    parser.add_argument("--output-label", default=None)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--env", type=Path, default=batch_io.DEFAULT_ENV)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
