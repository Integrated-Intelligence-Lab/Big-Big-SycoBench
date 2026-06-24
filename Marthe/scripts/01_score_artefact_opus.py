"""Initial-scoring batch for Claude Opus 4.8, built the same way as the gpt-5.5
default S0 batch (01_score_artefact.py): default neutral prompt, number-only
1-100 score, N=20 runs, on the same three scale-100 artefacts.

Difference from the gpt-5.5 version is only the provider surface:
- Anthropic Messages Batches API shape instead of the OpenAI /v1/responses shape.
- `initial_prompt` goes in `system` (the gpt-5.5 run put it in `instructions`),
  the artefact goes in the user turn.
- Opus 4.8 has no `temperature` (it 400s); run-to-run variance across the 20
  runs is inherent sampling. Reasoning is adaptive thinking; depth is set with
  `output_config.effort`, here "medium" to parallel gpt-5.5's reasoning=medium.

This writes a JSONL of {custom_id, params} lines — one Anthropic batch request
per line — mirroring the "build the input file, submit separately" workflow.

To submit (needs ANTHROPIC_API_KEY in the environment; pip install anthropic):

    import json, anthropic
    from anthropic.types.messages.batch_create_params import Request
    client = anthropic.Anthropic()
    reqs = [Request(custom_id=r["custom_id"], params=r["params"])
            for r in map(json.loads, open(OUT_PATH, encoding="utf-8"))]
    batch = client.messages.batches.create(requests=reqs)
    print(batch.id, batch.processing_status)

custom_id: {artefact}_run{n}  (matches the gpt-5.5 default S0 convention, so the
existing cycle scripts' load_s0 split-on "_run" can read this output too).
"""
import json
import os

ARTEFACT_FILES = ["L01_scale100", "M02_scale100", "S02_scale100"]
ARTEFACT_DIR = "Marthe/artefacts"
OUT_DIR = "Marthe/results/opus48"
OUT_PATH = os.path.join(OUT_DIR, "initial_default_batch_in.jsonl")

N_RUNS = 20
MODEL = "claude-opus-4-8"
EFFORT = "medium"     # parallels gpt-5.5 reasoning=medium; the comparability knob
MAX_TOKENS = 4096     # room for adaptive thinking + the number-only answer


def read_artefact(name):
    with open(os.path.join(ARTEFACT_DIR, name + ".json"), encoding="utf-8") as f:
        return json.load(f)


# Structured output: force a single integer. The first uploaded run showed Opus
# 4.8 ignoring "Reply with the number" 76% of the time (appended justifications,
# one in Russian) and drifting scale (M02 scored 5-10, S02 bimodal 5-80). A
# json_schema with one integer field removes the free-text channel so the answer
# is always a clean parseable score; the 1-100 framing stays in the prompt +
# field description (json_schema can't express numeric min/max). Adaptive thinking
# still runs in thinking blocks, so reasoning isn't suppressed -- only the
# visible answer is constrained.
SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {
            "type": "integer",
            "description": "Overall quality, an integer from 1 (very poor) to 100 (excellent).",
        }
    },
    "required": ["score"],
    "additionalProperties": False,
}


def make_request(art, run):
    return {
        "custom_id": f"{art['id']}_run{run}",
        "params": {
            "model": MODEL,
            "max_tokens": MAX_TOKENS,
            "thinking": {"type": "adaptive"},
            "output_config": {
                "effort": EFFORT,
                "format": {"type": "json_schema", "schema": SCORE_SCHEMA},
            },
            "system": art["initial_prompt"].strip(),
            "messages": [
                {"role": "user", "content": f"Artefact: {art['artefact']}\n"}
            ],
        },
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    reqs = []
    for name in ARTEFACT_FILES:
        art = read_artefact(name)
        for run in range(N_RUNS):
            reqs.append(make_request(art, run))
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in reqs:
            json.dump(r, f, ensure_ascii=False)
            f.write("\n")
    print(f"Wrote {len(reqs)} requests ({len(ARTEFACT_FILES)} artefacts x {N_RUNS} runs) to {OUT_PATH}")


if __name__ == "__main__":
    main()
