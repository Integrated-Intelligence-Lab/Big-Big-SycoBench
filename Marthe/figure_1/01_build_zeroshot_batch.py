"""Build 0-shot batch: argument presented inside the initial scoring prompt.

In every other shot condition the model first commits to S0 (no argument),
then sees the argument as a follow-up turn.  Here the argument is baked into
the FIRST (and only) user message alongside the artefact, so there is no prior
S0 to anchor on.  This measures initial susceptibility rather than revision
willingness.

20 independent runs per argument (matching the multi-turn S0 run count) give
a distribution for the figure.  Both directions are included for all three
artefacts.

  3 artefacts × 2 directions × 2 validities × 3 args × 20 runs = 720 requests

Message format (single turn):
  user: <initial_prompt_without_trailing_reply_ask>

         Artefact: <artefact>

         <cycle_text>   ← already ends with "...what score from 1 to 100?"

custom_id: {aid}|{direction}|{validity}|idx{idx}|r{run}|zeroshot

Output: results/batch_zeroshot_in.jsonl
"""
import json
import os

from common import (
    ARTEFACTS, EFFORT, MODEL, OPPOSITE_DIR, ORIGINAL_DIR, RESULTS_DIR,
    VALIDITIES, initial_prompt_for_zeroshot, load_all_artefacts,
)

N_RUNS = 20


def make_request(aid, direction, validity, idx, run, art):
    cycle_text = art["pushback"][direction]["cycles"][validity][idx].strip()
    init = initial_prompt_for_zeroshot(art)
    content = f"{init}\n\nArtefact: {art['artefact']}\n\n{cycle_text}"
    return {
        "custom_id": f"{aid}|{direction}|{validity}|idx{idx}|r{run}|zeroshot",
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": MODEL,
            "reasoning": {"effort": EFFORT},
            "input": [{"role": "user", "content": content}],
            "store": True,
        },
    }


def main():
    arts = load_all_artefacts()
    requests = []
    for aid in ARTEFACTS:
        art = arts[aid]
        for direction in [ORIGINAL_DIR[aid], OPPOSITE_DIR[aid]]:
            for validity in VALIDITIES:
                for idx in range(3):
                    for run in range(N_RUNS):
                        requests.append(make_request(aid, direction, validity, idx, run, art))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "batch_zeroshot_in.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in requests:
            json.dump(r, f, ensure_ascii=False)
            f.write("\n")
    print(f"Wrote {len(requests)} requests → {out}")
    print("3 artefacts × 2 directions × 2 validities × 3 args × 20 runs = 720 requests")


if __name__ == "__main__":
    main()
