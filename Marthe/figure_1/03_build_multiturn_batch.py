"""Build multi-turn batch files reusing the existing 20 S0 runs per artefact.

One batch per turn, chained by injecting each prior turn's output as an
assistant message (stateless replay — same approach as
scripts/03_build_pushback_batch.py).

Turn 1 is equivalent to the existing 1-shot (singleshot) data, so no turn1
batch is needed.  Turn 2 and 3 accept singleshot output files directly as the
turn-1 input, mapping their custom_ids to the multi-turn format internally:

  singleshot  {aid}|{dir}|{val}|idx{i}|r{run}
  →  turn1    {aid}|{dir}|{val}|r{run}|ord{cyclic(i)}|t1

  where cyclic(0)="012", cyclic(1)="120", cyclic(2)="201"

S0 already exists (20 runs per artefact from the live multi-turn experiment in
results/pushback/live/live_runlog.jsonl).  From each S0 run we branch into 3
trajectories — one starting with each argument — giving:

  3 artefacts × 2 directions × 2 validities × 3 trajectories × 20 runs
  = 720 requests per turn level

Each trajectory follows a cyclic ordering starting from argument s:
  s=0 → (0, 1, 2)
  s=1 → (1, 2, 0)
  s=2 → (2, 0, 1)

Stage pipeline (no turn1 batch needed — reuse singleshot data):

  python 03_build_multiturn_batch.py turn2 \\
      --t1-orig <singleshot_orig_output.jsonl> \\
      --t1-opp  <batch_oneshot_opp_output.jsonl>
  # submit results/batch_mt_t2_in.jsonl  →  <t2_output.jsonl>

  python 03_build_multiturn_batch.py turn3 \\
      --t1-orig <singleshot_orig_output.jsonl> \\
      --t1-opp  <batch_oneshot_opp_output.jsonl> \\
      --t2 <t2_output.jsonl>
  # submit results/batch_mt_t3_in.jsonl  →  <t3_output.jsonl>

custom_ids:
  turn k: {aid}|{direction}|{validity}|r{run}|ord{o0}{o1}{o2}|t{k}
  e.g.    L01|lower|valid|r3|ord120|t2
"""
import argparse
import json
import os
import re

from common import (
    ARTEFACTS, EFFORT, MODEL, OPPOSITE_DIR, ORIGINAL_DIR,
    RESULTS_DIR, VALIDITIES, load_all_artefacts, load_s0_from_runlog,
)

N_RUNS = 20
# 3 cyclic orderings: each argument appears as the first exactly once
ORDERINGS_3 = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]


# ── helpers ──────────────────────────────────────────────────────────────────

def extract_text(body):
    return "\n".join(
        c.get("text", "")
        for item in body.get("output", [])
        if item.get("type") == "message"
        for c in item.get("content", [])
        if c.get("type") == "output_text"
    )


def parse_score(text):
    m = re.search(r"-?\d+", text)
    return int(m.group()) if m else None


def load_batch_scores(path):
    """Return {custom_id: score} from a batch output JSONL (turn2/turn3 format)."""
    scores = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        scores[r["custom_id"]] = parse_score(extract_text(r["response"]["body"]))
    return scores


# singleshot idx → cyclic ordering string (arg i starts this ordering)
_IDX_TO_ORD = {0: "012", 1: "120", 2: "201"}


def load_t1_from_singleshot(*paths):
    """Load turn-1 scores from singleshot output files, mapping custom_ids.

    Singleshot format:  {aid}|{dir}|{val}|idx{i}|r{run}[|...]
    Mapped to turn1:    {aid}|{dir}|{val}|r{run}|ord{cyclic(i)}|t1
    """
    scores = {}
    for path in paths:
        if path is None:
            continue
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            parts = r["custom_id"].split("|")
            if len(parts) < 5 or not parts[3].startswith("idx"):
                continue
            aid, d, val = parts[0], parts[1], parts[2]
            idx = int(parts[3][3:])
            run = int(parts[4][1:])
            t1_cid = f"{aid}|{d}|{val}|r{run}|ord{_IDX_TO_ORD[idx]}|t1"
            scores[t1_cid] = parse_score(extract_text(r["response"]["body"]))
    return scores


def make_request(custom_id, messages):
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": MODEL,
            "reasoning": {"effort": EFFORT},
            "input": messages,
            "store": True,
        },
    }


def write_batch(name, requests):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"batch_mt_{name}_in.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in requests:
            json.dump(r, f, ensure_ascii=False)
            f.write("\n")
    print(f"Wrote {len(requests)} requests → {path}")
    return path


def turn_cid(aid, direction, validity, run, ordering, k):
    ord_str = "".join(str(i) for i in ordering)
    return f"{aid}|{direction}|{validity}|r{run}|ord{ord_str}|t{k}"


def init_msg(art):
    return {"role": "user",
            "content": f"{art['initial_prompt'].strip()}\n\nArtefact: {art['artefact']}\n"}


def arg_msg(art, direction, validity, arg_idx):
    return {"role": "user",
            "content": art["pushback"][direction]["cycles"][validity][arg_idx].strip()}


def score_msg(score):
    return {"role": "assistant", "content": str(score)}


# ── stages ────────────────────────────────────────────────────────────────────

def build_turn1(arts, s0_scores):
    """Replay existing S0, then add the first argument of each trajectory."""
    requests, missing = [], 0
    for aid in ARTEFACTS:
        art = arts[aid]
        for direction in [ORIGINAL_DIR[aid], OPPOSITE_DIR[aid]]:
            for validity in VALIDITIES:
                for run in range(N_RUNS):
                    s0 = s0_scores.get(aid, {}).get(run)
                    if s0 is None:
                        missing += 1
                        continue
                    for ordering in ORDERINGS_3:
                        msgs = [init_msg(art), score_msg(s0),
                                arg_msg(art, direction, validity, ordering[0])]
                        requests.append(make_request(
                            turn_cid(aid, direction, validity, run, ordering, 1), msgs
                        ))
    if missing:
        print(f"WARNING: {missing} (artefact, run) combos missing S0 in runlog")
    return requests


def build_turn2(arts, s0_scores, t1_scores):
    """Replay S0 + turn-1 output, then add the second argument."""
    requests, missing = [], 0
    for aid in ARTEFACTS:
        art = arts[aid]
        for direction in [ORIGINAL_DIR[aid], OPPOSITE_DIR[aid]]:
            for validity in VALIDITIES:
                for run in range(N_RUNS):
                    s0 = s0_scores.get(aid, {}).get(run)
                    for ordering in ORDERINGS_3:
                        t1 = t1_scores.get(turn_cid(aid, direction, validity, run, ordering, 1))
                        if s0 is None or t1 is None:
                            missing += 1
                            continue
                        msgs = [
                            init_msg(art), score_msg(s0),
                            arg_msg(art, direction, validity, ordering[0]), score_msg(t1),
                            arg_msg(art, direction, validity, ordering[1]),
                        ]
                        requests.append(make_request(
                            turn_cid(aid, direction, validity, run, ordering, 2), msgs
                        ))
    if missing:
        print(f"WARNING: {missing} conversations missing prior scores")
    return requests


def build_turn3(arts, s0_scores, t1_scores, t2_scores):
    """Replay full 2-turn history, then add the third argument."""
    requests, missing = [], 0
    for aid in ARTEFACTS:
        art = arts[aid]
        for direction in [ORIGINAL_DIR[aid], OPPOSITE_DIR[aid]]:
            for validity in VALIDITIES:
                for run in range(N_RUNS):
                    s0 = s0_scores.get(aid, {}).get(run)
                    for ordering in ORDERINGS_3:
                        t1 = t1_scores.get(turn_cid(aid, direction, validity, run, ordering, 1))
                        t2 = t2_scores.get(turn_cid(aid, direction, validity, run, ordering, 2))
                        if s0 is None or t1 is None or t2 is None:
                            missing += 1
                            continue
                        msgs = [
                            init_msg(art), score_msg(s0),
                            arg_msg(art, direction, validity, ordering[0]), score_msg(t1),
                            arg_msg(art, direction, validity, ordering[1]), score_msg(t2),
                            arg_msg(art, direction, validity, ordering[2]),
                        ]
                        requests.append(make_request(
                            turn_cid(aid, direction, validity, run, ordering, 3), msgs
                        ))
    if missing:
        print(f"WARNING: {missing} conversations missing prior scores")
    return requests


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["turn2", "turn3"])
    ap.add_argument("--runlog", default=None,
                    help="path to live_runlog.jsonl (default: auto-resolved)")
    ap.add_argument("--t1-orig", default=None,
                    help="singleshot output for the ORIGINAL direction "
                         "(results/singleshot/gpt55/*args*output*.jsonl)")
    ap.add_argument("--t1-opp", default=None,
                    help="singleshot output for the OPPOSITE direction "
                         "(figure_1/results/batch_*oneshot_opp*_output.jsonl)")
    ap.add_argument("--t2", default=None, help="turn2 batch output JSONL")
    a = ap.parse_args()

    arts = load_all_artefacts()
    s0_scores = load_s0_from_runlog(a.runlog)

    if not a.t1_orig and not a.t1_opp:
        ap.error("provide at least one of --t1-orig / --t1-opp "
                 "(singleshot output files used as turn-1 input)")
    t1_scores = load_t1_from_singleshot(a.t1_orig, a.t1_opp)
    print(f"Loaded {len(t1_scores)} turn-1 scores from singleshot files")

    if a.stage == "turn2":
        reqs = build_turn2(arts, s0_scores, t1_scores)
        write_batch("t2", reqs)
        print("3 artefacts × 2 directions × 2 validities × 3 orderings × 20 runs = 720 requests")

    elif a.stage == "turn3":
        if not a.t2:
            ap.error("turn3 needs --t2 <t2_output.jsonl>")
        reqs = build_turn3(arts, s0_scores, t1_scores, load_batch_scores(a.t2))
        write_batch("t3", reqs)
        print("720 requests")


if __name__ == "__main__":
    main()
