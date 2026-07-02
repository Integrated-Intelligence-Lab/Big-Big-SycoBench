"""Build 1-shot batch for the OPPOSITE push direction.

The original-direction singleshot data (L01 lower, M02 raise, S02 lower)
already exists in results/singleshot/gpt55/.  This script builds the missing
mirror: L01 raise, M02 lower, S02 raise.

The format is identical to script 14 (Marthe/scripts/14_build_singleshot_args.py):
  [ user: initial_prompt + artefact,  assistant: <S0>,  user: cycle_text ]

S0 scores are reused from the live multi-turn runlog (same model, same artefacts,
20 runs each) so we get exactly 20 runs per argument, matching the original-
direction singleshot coverage.

custom_id: {aid}|{direction}|{validity}|idx{idx}|r{run}
  (same schema as original singleshot custom_ids, direction encodes which side)

Output: results/batch_oneshot_opp_in.jsonl
  3 artefacts × 1 opp direction × 2 validities × 3 args × 20 runs = 360 requests
"""
import json
import os

from common import (
    ARTEFACTS, EFFORT, MODEL, OPPOSITE_DIR, RESULTS_DIR,
    VALIDITIES, load_all_artefacts, load_s0_from_runlog,
)

N_RUNS = 20


def make_request(aid, direction, validity, idx, run, s0_score, art):
    init = art["initial_prompt"].strip()
    artefact_text = art["artefact"]
    cycle_text = art["pushback"][direction]["cycles"][validity][idx].strip()
    return {
        "custom_id": f"{aid}|{direction}|{validity}|idx{idx}|r{run}",
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": MODEL,
            "reasoning": {"effort": EFFORT},
            "input": [
                {"role": "user",    "content": f"{init}\n\nArtefact: {artefact_text}\n"},
                {"role": "assistant","content": str(s0_score)},
                {"role": "user",    "content": cycle_text},
            ],
            "store": True,
        },
    }


def main():
    arts = load_all_artefacts()
    s0_all = load_s0_from_runlog()

    requests = []
    missing = 0
    for aid in ARTEFACTS:
        art = arts[aid]
        direction = OPPOSITE_DIR[aid]
        s0_runs = s0_all.get(aid, {})
        for validity in VALIDITIES:
            for idx in range(3):
                for run in range(N_RUNS):
                    s0 = s0_runs.get(run)
                    if s0 is None:
                        missing += 1
                        continue
                    requests.append(
                        make_request(aid, direction, validity, idx, run, s0, art)
                    )

    if missing:
        print(f"WARNING: skipped {missing} (run, artefact) combos with missing S0")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "batch_oneshot_opp_in.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in requests:
            json.dump(r, f, ensure_ascii=False)
            f.write("\n")
    print(f"Wrote {len(requests)} requests → {out}")
    print("3 artefacts × 1 opp direction × 2 validities × 3 args × 20 runs = 360 requests")


if __name__ == "__main__":
    main()
