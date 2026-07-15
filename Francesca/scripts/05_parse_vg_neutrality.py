import json
import re
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Editable analysis settings
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "Francesca" / "results" / "vg_neutrality"

# After downloading completed batch outputs, place them in RESULTS_DIR.
# Leave empty to auto-detect batch_output_vg_neutrality*.jsonl.
BATCH_OUTPUT_FILES = []


def extract_text(body):
    parts = []
    for item in body.get("output", []):
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    parts.append(content.get("text", ""))
    return "\n".join(parts).strip()


def parse_json_object(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None


def main():
    paths = (
        [RESULTS_DIR / name for name in BATCH_OUTPUT_FILES]
        if BATCH_OUTPUT_FILES
        else sorted(RESULTS_DIR.glob("batch_output_vg_neutrality*.jsonl"))
    )
    if not paths:
        raise SystemExit(
            f"No neutrality output files found in {RESULTS_DIR}. "
            "Expected batch_output_vg_neutrality*.jsonl."
        )

    rows = []
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                _, artefact_id = row["custom_id"].split("|")
                text = extract_text(row["response"]["body"])
                parsed = parse_json_object(text)
                if parsed is None:
                    rows.append(
                        {
                            "source_file": path.name,
                            "artefact": artefact_id,
                            "neutrality_label": None,
                            "is_neutral": None,
                            "bias_cues": None,
                            "reason": None,
                            "prompt_fix": None,
                            "raw_text": text,
                        }
                    )
                    continue

                rows.append(
                    {
                        "source_file": path.name,
                        "artefact": artefact_id,
                        "neutrality_label": parsed.get("neutrality_label"),
                        "is_neutral": parsed.get("is_neutral"),
                        "bias_cues": "; ".join(parsed.get("bias_cues", [])),
                        "reason": parsed.get("reason"),
                        "prompt_fix": parsed.get("prompt_fix"),
                        "raw_text": text,
                    }
                )

    df = pd.DataFrame(rows).sort_values("artefact")
    out_csv = RESULTS_DIR / "neutrality_summary.csv"
    df.to_csv(out_csv, index=False)

    print(df[["artefact", "neutrality_label", "is_neutral", "bias_cues"]].to_string(index=False))
    print(f"\nSaved {out_csv}")


if __name__ == "__main__":
    main()
