from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


Json = dict[str, Any]
PIPELINE_DIR = Path(__file__).resolve().parent
DEFAULT_ENV = PIPELINE_DIR.parent / ".env"
TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if not item or item.startswith("#") or "=" not in item:
            continue
        key, value = item.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def client_from_env(path: Path) -> Any:
    load_env(path)
    from openai import OpenAI

    return OpenAI()


def jsonable(value: Any) -> Json:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return json.loads(value.model_dump_json())


def write_json(path: Path, data: Json) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def input_candidates(path: Path) -> list[Path]:
    if not path.parent.exists():
        return []
    suffix = "_in.jsonl"
    if not path.name.endswith(suffix):
        return []
    prefix = path.name.removesuffix(suffix)
    return sorted(item for item in path.parent.glob(f"{prefix}_*{suffix}") if item.is_file())


def require_input(path: Path) -> None:
    if path.exists():
        return
    candidates = input_candidates(path)
    if candidates:
        lines = "\n".join(f"  {item}" for item in candidates[:5])
        raise SystemExit(f"input file not found: {path}\ndid you mean:\n{lines}")
    raise SystemExit(f"input file not found: {path}")


def content_text(content: Any) -> str:
    text = getattr(content, "text", None)
    if isinstance(text, str):
        return text
    read = getattr(content, "read", None)
    if callable(read):
        data = read()
        if isinstance(data, bytes):
            return data.decode("utf-8")
        return str(data)
    return str(content)


def print_batch(batch: Any) -> None:
    counts = getattr(batch, "request_counts", None)
    completed = getattr(counts, "completed", "?") if counts is not None else "?"
    failed = getattr(counts, "failed", "?") if counts is not None else "?"
    total = getattr(counts, "total", "?") if counts is not None else "?"
    print(f"batch_id={batch.id}")
    print(f"status={batch.status}")
    print(f"requests={completed}/{total} completed, {failed} failed")
    print(f"output_file_id={getattr(batch, 'output_file_id', None)}")
    print(f"error_file_id={getattr(batch, 'error_file_id', None)}")


def submit(args: argparse.Namespace) -> None:
    require_input(args.input)
    client = client_from_env(args.env)
    with args.input.open("rb") as handle:
        uploaded = client.files.create(file=handle, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint=args.endpoint,
        completion_window=args.completion_window,
    )
    print(f"input_file_id={uploaded.id}")
    print_batch(batch)
    if args.metadata_output is not None:
        write_json(args.metadata_output, {"input_file_id": uploaded.id, "batch": jsonable(batch)})


def status(args: argparse.Namespace) -> None:
    client = client_from_env(args.env)
    batch = client.batches.retrieve(args.batch_id)
    print_batch(batch)
    if args.metadata_output is not None:
        write_json(args.metadata_output, {"batch": jsonable(batch)})


def wait(args: argparse.Namespace) -> None:
    client = client_from_env(args.env)
    while True:
        batch = client.batches.retrieve(args.batch_id)
        print_batch(batch)
        if batch.status in TERMINAL_STATUSES:
            if args.metadata_output is not None:
                write_json(args.metadata_output, {"batch": jsonable(batch)})
            return
        time.sleep(args.poll_seconds)


def download(args: argparse.Namespace) -> None:
    client = client_from_env(args.env)
    batch = client.batches.retrieve(args.batch_id)
    if getattr(batch, "output_file_id", None) is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content_text(client.files.content(batch.output_file_id)), encoding="utf-8")
        print(f"wrote output -> {args.output}")
    else:
        print("no output_file_id available")
    if getattr(batch, "error_file_id", None) is not None and args.error_output is not None:
        args.error_output.parent.mkdir(parents=True, exist_ok=True)
        args.error_output.write_text(content_text(client.files.content(batch.error_file_id)), encoding="utf-8")
        print(f"wrote errors -> {args.error_output}")


def add_env(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--env", type=Path, default=DEFAULT_ENV)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit_parser = subparsers.add_parser("submit")
    add_env(submit_parser)
    submit_parser.add_argument("--input", type=Path, required=True)
    submit_parser.add_argument("--endpoint", default="/v1/responses")
    submit_parser.add_argument("--completion-window", default="24h")
    submit_parser.add_argument("--metadata-output", type=Path, default=None)
    submit_parser.set_defaults(func=submit)

    status_parser = subparsers.add_parser("status")
    add_env(status_parser)
    status_parser.add_argument("--batch-id", required=True)
    status_parser.add_argument("--metadata-output", type=Path, default=None)
    status_parser.set_defaults(func=status)

    wait_parser = subparsers.add_parser("wait")
    add_env(wait_parser)
    wait_parser.add_argument("--batch-id", required=True)
    wait_parser.add_argument("--poll-seconds", type=int, default=60)
    wait_parser.add_argument("--metadata-output", type=Path, default=None)
    wait_parser.set_defaults(func=wait)

    download_parser = subparsers.add_parser("download")
    add_env(download_parser)
    download_parser.add_argument("--batch-id", required=True)
    download_parser.add_argument("--output", type=Path, required=True)
    download_parser.add_argument("--error-output", type=Path, default=None)
    download_parser.set_defaults(func=download)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
