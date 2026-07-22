"""Submit / track / fetch Batch API jobs, with a hard budget guard.

Usage:
  python3 submit.py estimate <stage>      cost projection for out/batch_<stage>.jsonl
  python3 submit.py submit   <stage>      preflight + budget check + upload + create batch
  python3 submit.py status                status of all tracked batches
  python3 submit.py fetch    <stage>      download results -> out/results_<stage>.jsonl
"""

import json
import sys
from datetime import datetime, timezone

import oa
from config import (BUDGET_USD, EXP_ADJUDICATE, EXP_ARTEFACT, EXP_PUSHBACK,
                    EXP_QC, EXP_REPAIR, LEDGER, MODEL, PRICE_INPUT,
                    PRICE_OUTPUT, ROOT)

BATCHES = ROOT / "out" / "batches.json"


def _load(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def _expected_output(stage: str, custom_id: str) -> int:
    if stage == "artefacts":
        # tier is not in the custom_id; use the spec
        sid = custom_id.split("-", 1)[1]
        for line in open(ROOT / "specs" / "specs.jsonl"):
            s = json.loads(line)
            if s["id"] == sid:
                return EXP_ARTEFACT[s["length"]]
        return max(EXP_ARTEFACT.values())
    return {"pushbacks": EXP_PUSHBACK, "qc": EXP_QC,
            "repair": EXP_REPAIR, "adjudicate": EXP_ADJUDICATE}[stage]


def estimate(stage: str) -> float:
    path = ROOT / "out" / f"batch_{stage}.jsonl"
    if not path.exists():
        sys.exit(f"{path} missing - run build_batches.py {stage} first")
    tok_in = tok_out = n = 0
    for line in open(path):
        req = json.loads(line)
        chars = sum(len(m["content"]) for m in req["body"]["messages"])
        tok_in += chars // 4 + 50
        tok_out += _expected_output(stage, req["custom_id"])
        n += 1
    cost = tok_in / 1e6 * PRICE_INPUT + tok_out / 1e6 * PRICE_OUTPUT
    print(f"{stage}: {n} requests, ~{tok_in/1e6:.2f}M in, ~{tok_out/1e6:.2f}M out "
          f"(expected, not caps) -> ~${cost:.2f} at batch pricing")
    return cost


def spent() -> float:
    return sum(e["cost_usd"] for e in _load(LEDGER, []))


def submit(stage: str) -> None:
    cost = estimate(stage)
    already = spent()
    if already + cost > BUDGET_USD:
        sys.exit(f"BUDGET GUARD: ${already:.2f} spent + ${cost:.2f} projected "
                 f"> ${BUDGET_USD:.2f} cap. Not submitting.")
    oa.preflight_model(MODEL)
    path = ROOT / "out" / f"batch_{stage}.jsonl"
    up = oa.upload_file(str(path))
    batch = oa.create_batch(up["id"])
    tracked = _load(BATCHES, {})
    tracked[stage] = {"batch_id": batch["id"], "input_file_id": up["id"],
                      "submitted": datetime.now(timezone.utc).isoformat(),
                      "projected_usd": round(cost, 2)}
    BATCHES.write_text(json.dumps(tracked, indent=2))
    print(f"submitted {stage}: batch {batch['id']} (status {batch['status']})\n"
          f"budget: ${already:.2f} spent, ${cost:.2f} projected for this batch, "
          f"${BUDGET_USD:.2f} cap")


def status() -> None:
    tracked = _load(BATCHES, {})
    if not tracked:
        sys.exit("no batches tracked yet")
    for stage, info in tracked.items():
        b = oa.batch_status(info["batch_id"])
        counts = b.get("request_counts", {})
        print(f"{stage}: {b['status']}  "
              f"({counts.get('completed', 0)}/{counts.get('total', 0)} done, "
              f"{counts.get('failed', 0)} failed)  batch={b['id']}")


def fetch(stage: str) -> None:
    tracked = _load(BATCHES, {})
    if stage not in tracked:
        sys.exit(f"no tracked batch for stage '{stage}'")
    b = oa.batch_status(tracked[stage]["batch_id"])
    if b["status"] != "completed":
        sys.exit(f"batch is '{b['status']}', not completed")
    out = ROOT / "out" / f"results_{stage}.jsonl"
    out.write_bytes(oa.download_file(b["output_file_id"]))
    n_lines = sum(1 for _ in open(out))
    if b.get("error_file_id"):
        err = ROOT / "out" / f"errors_{stage}.jsonl"
        err.write_bytes(oa.download_file(b["error_file_id"]))
        print(f"WARNING: error file written -> {err}")

    tok_in = tok_out = 0
    for line in open(out):
        u = json.loads(line).get("response", {}).get("body", {}).get("usage", {})
        tok_in += u.get("prompt_tokens", 0)
        tok_out += u.get("completion_tokens", 0)
    cost = tok_in / 1e6 * PRICE_INPUT + tok_out / 1e6 * PRICE_OUTPUT
    ledger = _load(LEDGER, [])
    ledger.append({"stage": stage, "batch_id": b["id"],
                   "tokens_in": tok_in, "tokens_out": tok_out,
                   "cost_usd": round(cost, 2),
                   "fetched": datetime.now(timezone.utc).isoformat()})
    LEDGER.write_text(json.dumps(ledger, indent=2))
    print(f"fetched {n_lines} results -> {out}")
    print(f"actual usage: {tok_in/1e6:.2f}M in, {tok_out/1e6:.2f}M out = ${cost:.2f}")
    print(f"total spent so far: ${spent():.2f} of ${BUDGET_USD:.2f}")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "estimate" and len(sys.argv) > 2:
        estimate(sys.argv[2])
    elif cmd == "submit" and len(sys.argv) > 2:
        submit(sys.argv[2])
    elif cmd == "status":
        status()
    elif cmd == "fetch" and len(sys.argv) > 2:
        fetch(sys.argv[2])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
