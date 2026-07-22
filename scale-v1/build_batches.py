"""Build Batch API input files for the three stages.

Usage:
  python3 build_batches.py artefacts   specs/specs.jsonl            -> out/batch_artefacts.jsonl
  python3 build_batches.py pushbacks   candidates/artefacts/*.json  -> out/batch_pushbacks.jsonl
  python3 build_batches.py qc          candidates/json/*.json       -> out/batch_qc.jsonl
  python3 build_batches.py <stage> --only C001,C007   (rebuild subset, e.g. after QC flags)
"""

import json
import sys
from pathlib import Path

from config import (CAP_ADJUDICATE, CAP_ARTEFACT, CAP_PUSHBACK, CAP_QC,
                    CAP_REPAIR, MODEL, REASONING_EFFORT, ROOT)
import prompts


def _body(messages: list, schema: dict, max_tokens: int) -> dict:
    return {
        "model": MODEL,
        "messages": messages,
        "reasoning_effort": REASONING_EFFORT,
        "max_completion_tokens": max_tokens,
        "response_format": {"type": "json_schema", "json_schema": schema},
    }


def _line(custom_id: str, body: dict) -> str:
    return json.dumps({"custom_id": custom_id, "method": "POST",
                       "url": "/v1/chat/completions", "body": body})


def load_specs() -> dict:
    specs = {}
    with open(ROOT / "specs" / "specs.jsonl") as f:
        for line in f:
            s = json.loads(line)
            specs[s["id"]] = s
    return specs


def build_artefacts(only: set | None) -> list[str]:
    lines = []
    for sid, spec in load_specs().items():
        if only and sid not in only:
            continue
        lines.append(_line(f"art-{sid}", _body(
            prompts.artefact_messages(spec), prompts.ARTEFACT_SCHEMA,
            CAP_ARTEFACT[spec["length"]])))
    return lines


def build_pushbacks(only: set | None) -> list[str]:
    specs = load_specs()
    lines = []
    for path in sorted((ROOT / "candidates" / "artefacts").glob("C*.json")):
        sid = path.stem
        if only and sid not in only:
            continue
        art = json.loads(path.read_text())
        lines.append(_line(f"pb-{sid}", _body(
            prompts.pushback_messages(specs[sid], art), prompts.PUSHBACK_SCHEMA,
            CAP_PUSHBACK)))
    return lines


def build_qc(only: set | None) -> list[str]:
    specs = load_specs()
    lines = []
    for path in sorted((ROOT / "candidates" / "json").glob("C*.json")):
        sid = path.stem
        if only and sid not in only:
            continue
        rec = json.loads(path.read_text())
        art = {"anchor_score_0_100": rec["anchor_score_0_100"],
               "artefact_markdown": rec["artefact"]}
        for arm in ("valid", "invalid"):
            lines.append(_line(f"qc-{arm}-{sid}", _body(
                prompts.qc_messages(specs[sid], art, rec["pushback"], arm),
                prompts.QC_SCHEMA, CAP_QC)))
    return lines


def _flagged(only: set | None):
    """Yield (rec, arm, direction, idx, reason) for every flagged argument.

    If out/must_fix_slots.json exists (written by parse_results.py adjudicate),
    only those slots are yielded; delete the file to cover all flags.
    """
    verdicts = json.loads((ROOT / "out" / "qc_verdicts.json").read_text())
    slots_file = ROOT / "out" / "must_fix_slots.json"
    slots = (set(map(tuple, json.loads(slots_file.read_text())))
             if slots_file.exists() else None)
    if slots is not None:
        print(f"restricting to {len(slots)} must_fix slots from {slots_file.name}")
    for sid, arms in sorted(verdicts.items()):
        if only and sid not in only:
            continue
        rec_path = ROOT / "candidates" / "json" / f"{sid}.json"
        if not rec_path.exists():
            continue
        rec = json.loads(rec_path.read_text())
        for arm, vs in arms.items():
            for v in vs:
                if v["verdict"] != "flag":
                    continue
                direction, idx = v["argument_key"].split("/")
                if slots is not None and (sid, direction, arm, idx) not in slots:
                    continue
                yield rec, arm, direction, idx, v["reason"]


def build_repair(only: set | None) -> list[str]:
    """One request per QC-flagged argument, feeding the flag reason back."""
    lines = []
    for rec, arm, direction, idx, reason in _flagged(only):
        schema = (prompts.REPAIR_SCHEMA_VALID if arm == "valid"
                  else prompts.REPAIR_SCHEMA_INVALID)
        lines.append(_line(
            f"rep-{rec['id']}-{direction}-{arm}-{idx}",
            _body(prompts.repair_messages(rec, direction, arm, int(idx), reason),
                  schema, CAP_REPAIR)))
    return lines


def build_adjudicate(only: set | None) -> list[str]:
    """One request per QC-flagged argument: must_fix vs judgment_call."""
    lines = []
    for rec, arm, direction, idx, reason in _flagged(only):
        lines.append(_line(
            f"adj-{rec['id']}-{direction}-{arm}-{idx}",
            _body(prompts.adjudicate_messages(rec, direction, arm, int(idx), reason),
                  prompts.ADJUDICATE_SCHEMA, CAP_ADJUDICATE)))
    return lines


BUILDERS = {"artefacts": build_artefacts, "pushbacks": build_pushbacks,
            "qc": build_qc, "repair": build_repair,
            "adjudicate": build_adjudicate}


def main() -> None:
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    if stage not in BUILDERS:
        sys.exit(f"usage: build_batches.py {{{'|'.join(BUILDERS)}}} [--only C001,C002]")
    only = None
    if "--only" in sys.argv:
        only = set(sys.argv[sys.argv.index("--only") + 1].split(","))
    lines = BUILDERS[stage](only)
    if not lines:
        sys.exit(f"nothing to build for stage '{stage}' - is the previous stage parsed?")
    out = ROOT / "out" / f"batch_{stage}.jsonl"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} requests -> {out}")


if __name__ == "__main__":
    main()
