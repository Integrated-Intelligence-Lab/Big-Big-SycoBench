"""Method-comparison test for the multi-turn pushback: one pushback turn fired
through TWO mechanisms, reusing the existing 20 default S0 runs (original
prompt, neutral mitigation) on all three artefacts.

  - prefill : stateless replay. Rebuild the whole conversation each call:
              [user(initial_prompt + artefact), assistant(S0 score),
               user(pushback + rescore)].
  - prid    : previous_response_id. Send only [user(pushback + rescore)] and
              point at the stored S0 response id; the server carries the prior
              artefact turn, the S0 answer, and the model's turn-1 reasoning.

Both read S0 (score + response id) from the default initial-scoring output.
One batch input file per mechanism so the two can be diffed run-for-run.

NOTE on a built-in asymmetry: the reused S0 runs put the rating instruction in
the `instructions` field, which does NOT carry across previous_response_id. So
prefill has the original "rate 1-100" instruction in context while prid does
not (its rescore prompt restates the scale). If that matters, regenerate S0
with the prompt in the user message instead.

Usage:
    python 04_build_methodtest.py
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re

ARTEFACT_FILES = ["L01_scale100", "M02_scale100", "S02_scale100"]
ARTEFACT_DIR = "Marthe/artefacts"
# Default initial-scoring output ("original prompt", neutral): 3 x 20 responses.
S0_OUTPUT = "Marthe/results/initial_scores/initial_default_output.jsonl"
OUT_DIR = "Marthe/results/pushback"
VALIDITIES = ["valid", "invalid"]
MODEL = "gpt-5.5-2026-04-23"
EFFORT = "medium"


def load_artefacts():
    arts = {}
    for f in ARTEFACT_FILES:
        d = json.load(open(os.path.join(ARTEFACT_DIR, f + ".json"), encoding="utf-8"))
        # Push good artefacts DOWN, bad artefacts UP: the off-floor/ceiling
        # direction where sycophancy has room to show.
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
    """custom_id 'L01_run3' -> (aid, run) -> {'score': int, 'resp_id': str}."""
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        aid, run = r["custom_id"].split("_run")
        body = r["response"]["body"]
        out[(aid, int(run))] = {
            "score": parse_score(extract_text(body)),
            "resp_id": body["id"],
        }
    return out


def initial_user_turn(art):
    return {
        "role": "user",
        "content": f"{art['initial_prompt'].strip()}\n\nArtefact: {art['artefact']}\n",
    }


def pushback_turn(art, validity):
    direction = art["_direction"]
    args = art["pushback"][direction]["core_arguments"][validity]
    body = " ".join(a.strip() for a in args)
    return {"role": "user", "content": f"{body} {art['rescore_prompt'].strip()}"}


def make_request(custom_id, messages, previous_response_id=None):
    body = {
        "model": MODEL,
        "reasoning": {"effort": EFFORT},
        "input": [m for m in messages if m is not None],
        "store": True,
    }
    if previous_response_id:
        body["previous_response_id"] = previous_response_id
    return {"custom_id": custom_id, "method": "POST", "url": "/v1/responses", "body": body}


def write_batch(name, reqs):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"batch_in_{name}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in reqs:
            json.dump(r, f, ensure_ascii=False)
            f.write("\n")
    print(f"Wrote {len(reqs)} requests to {path}")


def main():
    arts = load_artefacts()
    s0 = load_s0(S0_OUTPUT)

    prefill, prid = [], []
    missing = 0
    for aid, art in arts.items():
        for val in VALIDITIES:
            for run in range(20):
                rec = s0.get((aid, run))
                if rec is None or rec["score"] is None or rec["resp_id"] is None:
                    missing += 1
                    continue
                cid = f"{aid}|{val}|r{run}"
                prefill.append(make_request(
                    cid,
                    [
                        initial_user_turn(art),
                        {"role": "assistant", "content": str(rec["score"])},
                        pushback_turn(art, val),
                    ],
                ))
                prid.append(make_request(
                    cid,
                    [pushback_turn(art, val)],
                    previous_response_id=rec["resp_id"],
                ))
    if missing:
        print(f"WARNING: skipped {missing} runs missing S0 score/id")

    write_batch("methodtest_prefill", prefill)
    write_batch("methodtest_prid", prid)


if __name__ == "__main__":
    main()
