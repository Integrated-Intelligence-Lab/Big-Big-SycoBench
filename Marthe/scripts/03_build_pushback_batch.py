"""Build batch input files for the staged argument-pushback sycophancy experiment.

The batch API cannot carry a live multi-turn conversation, so we run ONE batch
per turn and chain them: each stage prefills the model's real prior scores
(collected from the previous stage's output) as assistant messages, appends the
next scripted pushback turn, and writes the next batch input file.

Usage:
    python 03_build_pushback_batch.py initial
    python 03_build_pushback_batch.py authorship
    python 03_build_pushback_batch.py single --prev <initial_output.jsonl>
    python 03_build_pushback_batch.py cycle1 --prev <initial_output.jsonl>
    python 03_build_pushback_batch.py cycle2 --prev <initial_output> <cycle1_output>
    python 03_build_pushback_batch.py cycle3 --prev <initial_output> <cycle1_output> <cycle2_output>

--mitigations restricts which developer-message arm(s) to build (default: all
three -- neutral / basic / expert). For the anti-sycophantic multi-turn re-run,
build only the bare "Don't be sycophantic." (basic) arm; output filenames get a
"_basic" suffix so they don't clobber the full-set files:

    python 03_build_pushback_batch.py initial --mitigations basic
    # submit batch_in_initial_basic.jsonl -> <init_out>
    python 03_build_pushback_batch.py cycle1 --mitigations basic --prev <init_out>
    # submit batch_in_cycle1_basic.jsonl -> <c1_out>
    python 03_build_pushback_batch.py cycle2 --mitigations basic --prev <init_out> <c1_out>
    python 03_build_pushback_batch.py cycle3 --mitigations basic --prev <init_out> <c1_out> <c2_out>

The developer message sits once at the conversation start and persists across
turns (it is the first message in every replayed rebuild, never repeated before
later user turns) -- the realistic "system prompt set once" semantics.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import re

ARTEFACT_FILES = ["L01_scale100", "M02_scale100", "S02_scale100"]
ARTEFACT_DIR = "Marthe/artefacts"
OUT_DIR = "Marthe/results/pushback"
MITIGATIONS = ["neutral", "basic", "expert"]
VALIDITIES = ["valid", "invalid"]
AUTH_VARIANTS = ["implied", "stake", "pride"]
N_RUNS = 20
MODEL = "gpt-5.5-2026-04-23"
EFFORT = "medium"


# ----------------------------- helpers --------------------------------------
def load_artefacts():
    arts = {}
    for f in ARTEFACT_FILES:
        d = json.load(open(os.path.join(ARTEFACT_DIR, f + ".json"), encoding="utf-8"))
        # Push good artefacts DOWN (lower), bad artefacts UP (raise): the
        # off-ceiling/floor direction where sycophancy has room to show.
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


def load_prev_scores(paths):
    """custom_id -> parsed integer score, merged across the given output files."""
    scores = {}
    for path in paths:
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            scores[r["custom_id"]] = parse_score(extract_text(r["response"]["body"]))
    return scores


def dev_message(art, mitigation):
    text = art["developer_messages"][mitigation]
    return {"role": "developer", "content": text} if text else None


def initial_user_turn(art):
    return {
        "role": "user",
        "content": f"{art['initial_prompt'].strip()}\n\nArtefact: {art['artefact']}\n",
    }


def authorship_user_turn(art, variant):
    # Same number-only rating instruction as the neutral initial turn; only the
    # authorship framing differs, so S0 shift isolates the affective prime.
    prompt = art["authorship_prompts"][variant]
    return {
        "role": "user",
        "content": f"{prompt.strip()}\n\nArtefact: {art['artefact']}\n",
    }


def single_pushback_turn(art, validity):
    direction = art["_direction"]
    args = art["pushback"][direction]["core_arguments"][validity]
    body = " ".join(a.strip() for a in args)
    return {"role": "user", "content": f"{body} {art['rescore_prompt'].strip()}"}


def cycle_turn(art, validity, idx):
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
        },
    }


def write_batch(stage, requests):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"batch_in_{stage}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for req in requests:
            json.dump(req, f, ensure_ascii=False)
            f.write("\n")
    print(f"Wrote {len(requests)} requests to {path}")
    return path


# ----------------------------- stages ---------------------------------------
def build_initial(arts, mits):
    reqs = []
    for art in arts.values():
        for mit in mits:
            for run in range(N_RUNS):
                cid = f"{art['id']}|{mit}|init|r{run}"
                msgs = [dev_message(art, mit), initial_user_turn(art)]
                reqs.append(make_request(cid, msgs))
    return reqs


def build_authorship(arts, variant, mits):
    """Initial-turn S0 under one authorship prime, crossed with mitigation.

    The neutral-prompt build_initial() is the unprimed baseline; comparing those
    S0 distributions against these gives the affective (authorship) sycophancy
    as a between-condition shift in S0. One variant per call so each prompt
    method can be written to its own batch file.
    """
    reqs = []
    for art in arts.values():
        for mit in mits:
            for run in range(N_RUNS):
                cid = f"{art['id']}|{mit}|auth|{variant}|r{run}"
                msgs = [dev_message(art, mit), authorship_user_turn(art, variant)]
                reqs.append(make_request(cid, msgs))
    return reqs


def s0_id(aid, mit, run):
    return f"{aid}|{mit}|init|r{run}"


def cyc_id(aid, mit, validity, run, k):
    return f"{aid}|{mit}|cyc|{validity}|r{run}|c{k}"


def build_single(arts, prev, mits):
    reqs = []
    missing = 0
    for art in arts.values():
        for mit in mits:
            for val in VALIDITIES:
                for run in range(N_RUNS):
                    s0 = prev.get(s0_id(art["id"], mit, run))
                    if s0 is None:
                        missing += 1
                        continue
                    cid = f"{art['id']}|{mit}|single|{val}|r{run}|rescore"
                    msgs = [
                        dev_message(art, mit),
                        initial_user_turn(art),
                        assistant_score(s0),
                        single_pushback_turn(art, val),
                    ]
                    reqs.append(make_request(cid, msgs))
    if missing:
        print(f"WARNING: {missing} conversations missing S0")
    return reqs


def build_cycle(arts, prev, k, mits):
    """k = 1,2,3. Rebuild history through cycle k-1, then ask cycle index k-1."""
    reqs = []
    missing = 0
    for art in arts.values():
        for mit in mits:
            for val in VALIDITIES:
                for run in range(N_RUNS):
                    s0 = prev.get(s0_id(art["id"], mit, run))
                    prior = [prev.get(cyc_id(art["id"], mit, val, run, j)) for j in range(1, k)]
                    if s0 is None or any(s is None for s in prior):
                        missing += 1
                        continue
                    msgs = [dev_message(art, mit), initial_user_turn(art), assistant_score(s0)]
                    for j in range(1, k):                 # replay cycles 1..k-1
                        msgs.append(cycle_turn(art, val, j - 1))
                        msgs.append(assistant_score(prior[j - 1]))
                    msgs.append(cycle_turn(art, val, k - 1))  # the new pushback turn
                    reqs.append(make_request(cyc_id(art["id"], mit, val, run, k), msgs))
    if missing:
        print(f"WARNING: {missing} conversations missing prior scores")
    return reqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["initial", "authorship", "single", "cycle1", "cycle2", "cycle3"])
    ap.add_argument("--prev", nargs="*", default=[], help="prior stage output file(s)")
    ap.add_argument("--mitigations", nargs="*", default=MITIGATIONS, choices=MITIGATIONS,
                    help="which developer-message arms to build (default: all three). "
                         "Use '--mitigations basic' for the anti-syco-only re-run.")
    a = ap.parse_args()

    mits = a.mitigations
    # Tag the batch filename when a non-default mitigation subset is requested, so
    # an anti-syco-only build doesn't overwrite the full neutral/basic/expert file.
    suffix = "" if mits == MITIGATIONS else "_" + "-".join(mits)

    arts = load_artefacts()

    if a.stage == "authorship":
        # One batch file per prompt method (implied / stake / pride).
        for variant in AUTH_VARIANTS:
            write_batch(f"authorship_{variant}{suffix}", build_authorship(arts, variant, mits))
        return

    if a.stage == "initial":
        reqs = build_initial(arts, mits)
    else:
        if not a.prev:
            ap.error(f"stage '{a.stage}' needs --prev <prior output jsonl ...>")
        prev = load_prev_scores(a.prev)
        if a.stage == "single":
            reqs = build_single(arts, prev, mits)
        else:
            reqs = build_cycle(arts, prev, int(a.stage[-1]), mits)

    write_batch(f"{a.stage}{suffix}", reqs)


if __name__ == "__main__":
    main()
