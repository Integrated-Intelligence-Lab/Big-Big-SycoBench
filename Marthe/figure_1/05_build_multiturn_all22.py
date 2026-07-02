"""Build turn-2 / turn-3 multi-turn batches for ALL 22 artefacts (challenge dir).

Extends Figure 1 (and Figure 2's trajectory) from the 3 hand-picked artefacts to
all 22. The original figure_1 multi-turn batches only cover L01/M02/S02; this
rebuilds the 1->2->3 shot chain directly from the single-shot gpt-5.5 stage,
which is already a turn-1 conversation:

    single-shot args request input = [ user(init+artefact), assistant(S0), user(arg) ]

so turn k just appends the prior turn's score and the next argument. Because we
clone the exact single-shot request, the init prompt, per-run S0 and argument
texts are byte-identical to what produced the turn-1 data -- no artefact JSONs or
BT files needed. Only the challenge direction exists in the single-shot data
(one direction per artefact), which is exactly what Figure 2 uses.

  custom_id:  {aid}|{dir}|{val}|r{run}|ord{o0}{o1}{o2}|t{k}
  scale:      22 artefacts x 1 dir x 2 validities x 3 orderings x 20 runs = 2640 / turn

Stages:
  python 08_build_multiturn_all22.py turn2
  # submit results/batch_mt22_t2_in.jsonl  ->  <t2_output.jsonl>

  python 08_build_multiturn_all22.py turn3 --t2 <t2_output.jsonl>
  # submit results/batch_mt22_t3_in.jsonl  ->  <t3_output.jsonl>
"""
import argparse
import copy
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                          # Marthe/
OUT_DIR = os.path.join(HERE, "results")

N_RUNS = 20
ORDERINGS_3 = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]       # cyclic; ordering o starts with arg o[0]

# Per-tag config, set in main():
#   SS_DIR      = the model's single-shot folder
#   ALREADY_DONE = artefacts that already have multi-turn data (skip). For gpt55
#                  the 3 originals (L01/M02/S02) were run separately; for any new
#                  model nothing exists yet, so build all 22.
SS_DIR = os.path.join(ROOT, "results", "singleshot", "gpt55")
ALREADY_DONE = {"L01", "M02", "S02"}


def extract_text(body):
    return "\n".join(
        c.get("text", "")
        for it in body.get("output", [])
        if it.get("type") == "message"
        for c in it.get("content", [])
        if c.get("type") == "output_text"
    )


def parse_score(text):
    m = re.search(r"-?\d+", text)
    return int(m.group()) if m else None


def discover(kind):
    for p in sorted(glob.glob(os.path.join(SS_DIR, "*output*.jsonl") if kind == "out"
                              else os.path.join(SS_DIR, "args_batch_in.jsonl"))):
        if kind == "in":
            return p
        cid = json.loads(open(p).readline())["custom_id"]
        if "|" in cid:                                # args (not the S0 batch)
            return p
    raise SystemExit(f"could not find singleshot args {kind} under {SS_DIR}")


def load_t1_input():
    """Return {(aid,dir,val,idx,run): request_dict} for the single-shot args batch."""
    reqs = {}
    for line in open(discover("in"), encoding="utf-8"):
        r = json.loads(line)
        aid, d, val, idx, run = r["custom_id"].split("|")
        reqs[(aid, d, val, int(idx[3:]), int(run[1:]))] = r
    return reqs


def load_t1_scores():
    scores = {}
    for line in open(discover("out"), encoding="utf-8"):
        r = json.loads(line)
        aid, d, val, idx, run = r["custom_id"].split("|")
        scores[(aid, d, val, int(idx[3:]), int(run[1:]))] = parse_score(
            extract_text(r["response"]["body"]))
    return scores


def load_batch_scores(path):
    """{custom_id: score} from a turn-2/turn-3 style batch output."""
    out = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        out[r["custom_id"]] = parse_score(extract_text(r["response"]["body"]))
    return out


def arg_text(reqs, aid, d, val, idx, run):
    """The argument user-message (last message) of a single-shot request."""
    return reqs[(aid, d, val, idx, run)]["body"]["input"][-1]["content"]


def turn_cid(aid, d, val, run, ordering, k):
    return f"{aid}|{d}|{val}|r{run}|ord{''.join(map(str, ordering))}|t{k}"


def units(reqs):
    """Yield (aid, dir, val, run, ordering) for every conversation."""
    arts = sorted({k[0] for k in reqs} - ALREADY_DONE)
    dir_of = {k[0]: k[1] for k in reqs}               # one challenge dir per artefact
    for aid in arts:
        d = dir_of[aid]
        for val in ("valid", "invalid"):
            for run in range(N_RUNS):
                for ordering in ORDERINGS_3:
                    yield aid, d, val, run, ordering


def build_turn2(reqs, t1):
    out, missing = [], 0
    for aid, d, val, run, ordering in units(reqs):
        o0, o1 = ordering[0], ordering[1]
        base = reqs.get((aid, d, val, o0, run))
        s1 = t1.get((aid, d, val, o0, run))
        if base is None or s1 is None:
            missing += 1
            continue
        req = copy.deepcopy(base)
        req["custom_id"] = turn_cid(aid, d, val, run, ordering, 2)
        req["body"]["input"] = base["body"]["input"] + [
            {"role": "assistant", "content": str(s1)},
            {"role": "user", "content": arg_text(reqs, aid, d, val, o1, run)},
        ]
        out.append(req)
    return out, missing


def build_turn3(reqs, t1, t2):
    out, missing = [], 0
    for aid, d, val, run, ordering in units(reqs):
        o0, o1, o2 = ordering
        base = reqs.get((aid, d, val, o0, run))
        s1 = t1.get((aid, d, val, o0, run))
        s2 = t2.get(turn_cid(aid, d, val, run, ordering, 2))
        if base is None or s1 is None or s2 is None:
            missing += 1
            continue
        req = copy.deepcopy(base)
        req["custom_id"] = turn_cid(aid, d, val, run, ordering, 3)
        req["body"]["input"] = base["body"]["input"] + [
            {"role": "assistant", "content": str(s1)},
            {"role": "user", "content": arg_text(reqs, aid, d, val, o1, run)},
            {"role": "assistant", "content": str(s2)},
            {"role": "user", "content": arg_text(reqs, aid, d, val, o2, run)},
        ]
        out.append(req)
    return out, missing


def write_batch(tag, name, requests):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"batch_mt22_{tag}_{name}_in.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in requests:
            json.dump(r, f, ensure_ascii=False)
            f.write("\n")
    print(f"Wrote {len(requests)} requests -> {path}")
    return path


def main():
    global SS_DIR, ALREADY_DONE
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["turn2", "turn3"])
    ap.add_argument("--tag", default="gpt55", help="single-shot model folder under results/singleshot/")
    ap.add_argument("--t2", default=None, help="turn2 batch output JSONL (turn3 only)")
    a = ap.parse_args()

    SS_DIR = os.path.join(ROOT, "results", "singleshot", a.tag)
    # only gpt55 has the 3 originals already; for any other model build all 22
    ALREADY_DONE = {"L01", "M02", "S02"} if a.tag == "gpt55" else set()

    reqs = load_t1_input()
    t1 = load_t1_scores()
    n_art = len({k[0] for k in reqs} - ALREADY_DONE)
    print(f"[{a.tag}] Loaded {len(reqs)} single-shot requests, {len(t1)} turn-1 scores "
          f"({len({k[0] for k in reqs})} artefacts; building {n_art})")

    if a.stage == "turn2":
        out, missing = build_turn2(reqs, t1)
    else:
        if not a.t2:
            ap.error("turn3 needs --t2 <t2_output.jsonl>")
        out, missing = build_turn3(reqs, t1, load_batch_scores(a.t2))

    if missing:
        print(f"WARNING: {missing} conversations skipped (missing prior score)")
    write_batch(a.tag, a.stage, out)
    print(f"{n_art} artefacts x 1 dir x 2 val x 3 orderings x 20 runs = {n_art * 120} expected")


if __name__ == "__main__":
    main()
