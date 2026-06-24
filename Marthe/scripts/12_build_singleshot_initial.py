"""Single-shot isolation experiment, STAGE 1: initial S0 scoring on ALL 22
artefacts (both push directions are scored from the same neutral S0, so one S0
per artefact serves both directions in stage 2).

Why a new S0 run: the existing gpt-5.5 S0 only covers L01/M02/S02. Single-shot
isolation needs an S0 baseline for every artefact so each argument's one-turn
effect can be measured as (rescore - S0).

Reads Vincent's 1-10 artefacts directly and scales the initial prompt to 1-100,
number-only (matching the scale-100 design: "and a brief justification" dropped).
Stage 2 (12_build_singleshot_args.py, after S0 results) replays
[initial, S0, ONE cycle argument] for each of the 6 args per pool and rescoring.

Model-parametrized so the same script builds gpt-5.5 and a weaker OpenAI control:
    python 12_build_singleshot_initial.py --model gpt-5.5-2026-04-23 --tag gpt55
    python 12_build_singleshot_initial.py --model <weaker-id>       --tag <tag>

OpenAI Responses Batch shape, reasoning effort medium (matches the main runs).
custom_id: {artefact}_run{n}.
"""
import argparse
import glob
import json
import os

ARTEFACT_DIR = "Vincent/sycophancy-benchmark/artefacts/json"
N_RUNS = 20
EFFORT = "medium"


def scale_to_100(prompt):
    """1-10 number-with-justification prompt -> 1-100 number-only."""
    s = prompt.replace("1 to 10", "1 to 100").replace("10 is excellent", "100 is excellent")
    s = s.replace(" and a brief justification", "")
    return s


def load_artefacts():
    arts = []
    for f in sorted(glob.glob(os.path.join(ARTEFACT_DIR, "*.json"))):
        arts.append(json.load(open(f, encoding="utf-8")))
    return arts


def make_request(art, run, model):
    return {
        "custom_id": f"{art['id']}_run{run}",
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": model,
            "reasoning": {"effort": EFFORT},
            "instructions": scale_to_100(art["initial_prompt"]).strip(),
            "input": f"Artefact: {art['artefact']}\n",
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="OpenAI model id")
    ap.add_argument("--tag", required=True, help="short folder tag, e.g. gpt55")
    a = ap.parse_args()

    out_dir = os.path.join("Marthe/results/singleshot", a.tag)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "initial_batch_in.jsonl")

    arts = load_artefacts()
    reqs = [make_request(art, run, a.model) for art in arts for run in range(N_RUNS)]
    with open(out_path, "w", encoding="utf-8") as f:
        for r in reqs:
            json.dump(r, f, ensure_ascii=False)
            f.write("\n")
    print(f"Wrote {len(reqs)} requests ({len(arts)} artefacts x {N_RUNS} runs, model={a.model}) to {out_path}")


if __name__ == "__main__":
    main()
