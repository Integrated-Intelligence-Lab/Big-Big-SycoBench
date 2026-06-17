"""Multi-turn pushback CYCLES via stateless replay (prefill), reusing the
existing default S0 runs (original prompt, neutral) on all three artefacts.

The cycles present ONE escalating argument per turn (3 turns), each ending with
the rescore prompt, so we get a score after every turn: S0 -> S1 -> S2 -> S3.
Because the conversation is replayed in full each call, we run one batch per
turn and chain them (each stage prefills the model's real prior scores):

    python 05_build_cycles_prefill.py 1
    python 05_build_cycles_prefill.py 2 --prev <cycle1_output.jsonl>
    python 05_build_cycles_prefill.py 3 --prev <cycle1_output> <cycle2_output>

Turn 0 (S0) is read from the default initial-scoring output. previous_response_id
is NOT usable here: batch-created responses aren't chainable (all 120 prid
requests 400'd with previous_response_not_found), so replay is the only option.

custom_id: {artefact}|{validity}|r{run}|c{k}  (k = cycle turn 1..3)
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import re

ARTEFACT_FILES = ["L01_scale100", "M02_scale100", "S02_scale100"]
ARTEFACT_DIR = "Marthe/artefacts"
S0_OUTPUT = "Marthe/results/initial_scores/batch_6a2ab6ba613c8190b307db0984f42a29_output.jsonl"
OUT_DIR = "Marthe/results/pushback"
VALIDITIES = ["valid", "invalid"]
N_RUNS = 20
MODEL = "gpt-5.5-2026-04-23"
EFFORT = "medium"


def load_artefacts():
    arts = {}
    for f in ARTEFACT_FILES:
        d = json.load(open(os.path.join(ARTEFACT_DIR, f + ".json"), encoding="utf-8"))
        d["_direction"] = "lower" if d["quality"] == "good" else "raise"
        arts[d["id"]] = d
    return arts


def extract_text(body):
    parts = []
    for item in body.get("output", []):
        if item.get("type") == "message":
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    parts.append(c.get("text", ""))
    return "\n".join(parts)


def parse_score(text):
    m = re.search(r"-?\d+", text)
    return int(m.group()) if m else None


def load_s0(path):
    """(aid, run) -> S0 score, from the default initial-scoring output."""
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        aid, run = r["custom_id"].split("_run")
        out[(aid, int(run))] = parse_score(extract_text(r["response"]["body"]))
    return out


def load_cycle_scores(paths):
    """{(aid, validity, run, k): score} merged across cycle output file(s)."""
    out = {}
    for path in paths:
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            aid, val, run, c = r["custom_id"].split("|")
            out[(aid, val, int(run[1:]), int(c[1:]))] = parse_score(
                extract_text(r["response"]["body"])
            )
    return out


def initial_user_turn(art):
    return {
        "role": "user",
        "content": f"{art['initial_prompt'].strip()}\n\nArtefact: {art['artefact']}\n",
    }


def cycle_turn(art, validity, idx):
    """idx is 0-based; cycle strings already embed the rescore prompt."""
    direction = art["_direction"]
    return {"role": "user", "content": art["pushback"][direction]["cycles"][validity][idx].strip()}


def assistant_score(score):
    return {"role": "assistant", "content": str(score)}


def make_request(custom_id, messages):
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": MODEL,
            "reasoning": {"effort": EFFORT},
            "input": [m for m in messages if m is not None],
            "store": True,
        },
    }


def write_batch(stage, reqs):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"batch_in_cycle{stage}_prefill.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in reqs:
            json.dump(r, f, ensure_ascii=False)
            f.write("\n")
    print(f"Wrote {len(reqs)} requests to {path}")


def build_cycle(arts, s0, prior, k):
    """Replay history through cycle k-1, then ask cycle index k-1 (the new turn)."""
    reqs = []
    missing = 0
    for aid, art in arts.items():
        for val in VALIDITIES:
            for run in range(N_RUNS):
                s0_score = s0.get((aid, run))
                prev_scores = [prior.get((aid, val, run, j)) for j in range(1, k)]
                if s0_score is None or any(s is None for s in prev_scores):
                    missing += 1
                    continue
                msgs = [initial_user_turn(art), assistant_score(s0_score)]
                for j in range(1, k):                       # replay cycles 1..k-1
                    msgs.append(cycle_turn(art, val, j - 1))
                    msgs.append(assistant_score(prev_scores[j - 1]))
                msgs.append(cycle_turn(art, val, k - 1))     # the new pushback turn
                reqs.append(make_request(f"{aid}|{val}|r{run}|c{k}", msgs))
    if missing:
        print(f"WARNING: skipped {missing} runs missing S0/prior scores")
    return reqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", type=int, choices=[1, 2, 3])
    ap.add_argument("--prev", nargs="*", default=[], help="prior cycle output jsonl(s)")
    a = ap.parse_args()

    arts = load_artefacts()
    s0 = load_s0(S0_OUTPUT)
    if a.stage > 1 and not a.prev:
        ap.error(f"stage {a.stage} needs --prev <cycle output(s) for turns 1..{a.stage - 1}>")
    prior = load_cycle_scores(a.prev)

    write_batch(a.stage, build_cycle(arts, s0, prior, a.stage))


if __name__ == "__main__":
    main()
