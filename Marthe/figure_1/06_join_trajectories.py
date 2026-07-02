"""Join the batch outputs into one tidy challenge-direction trajectory table.

Produces results/trajectories_challenge_22.csv with one row per
(artefact, validity, run, ordering) and the score at every stage:

    S0  ->  t1  ->  t2  ->  t3        (challenge / "right" direction only)

Sources (all gpt-5.5):
  S0   : results/singleshot/gpt55/<S0 output>            (custom_id  aid_runN)
  t1   : results/singleshot/gpt55/<args output>          (aid|dir|val|idxN|rN)
  t2/t3: results/batches/outputs/out_multiturn_t{2,3}_*  (aid|dir|val|rN|ordNNN|tK)
         - 3art_bothdir files: filtered to the challenge direction
         - 19art_challenge files: already challenge-only

Each artefact has exactly one challenge direction in the single-shot data
(lower if S0>50 else raise); the opposite-direction runs live in their own
batch files and are intentionally excluded here. turn-1 is taken from the
single-shot stage, mapping argument idx -> the cyclic ordering it starts
(idx0->012, idx1->120, idx2->201), matching how the multi-turn chain was built.
"""
import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SS_DIR = os.path.join(ROOT, "results", "singleshot", "gpt55")
OUT_DIR = os.path.join(HERE, "results", "batches", "outputs")
ORD2IDX0 = {"012": 0, "120": 1, "201": 2}


def extract_text(body):
    return "\n".join(
        c.get("text", "")
        for it in body.get("output", [])
        if it.get("type") == "message"
        for c in it.get("content", [])
        if c.get("type") == "output_text"
    )


def score(t):
    m = re.search(r"-?\d+", t)
    return int(m.group()) if m else None


def ss(kind):
    for p in sorted(glob.glob(os.path.join(SS_DIR, "*output*.jsonl"))):
        cid = json.loads(open(p).readline())["custom_id"]
        if kind == "s0" and "|" not in cid and "_run" in cid:
            return p
        if kind == "args" and "|" in cid:
            return p
    raise SystemExit(f"missing single-shot {kind} output under {SS_DIR}")


def main():
    global SS_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="gpt55")
    a = ap.parse_args()
    SS_DIR = os.path.join(ROOT, "results", "singleshot", a.tag)

    # S0 per (aid, run)
    s0 = {}
    for l in open(ss("s0"), encoding="utf-8"):
        r = json.loads(l)
        aid, run = r["custom_id"].split("_run")
        s0[(aid, int(run))] = score(extract_text(r["response"]["body"]))

    # turn-1 per (aid, dir, val, idx, run); also the one challenge dir per artefact
    t1 = {}
    dir_of = {}
    for l in open(ss("args"), encoding="utf-8"):
        r = json.loads(l)
        aid, d, val, idx, run = r["custom_id"].split("|")
        dir_of[aid] = d
        t1[(aid, d, val, int(idx[3:]), int(run[1:]))] = score(extract_text(r["response"]["body"]))

    # turn-2 / turn-3 from this model's multi-turn outputs (gpt55: the 3-artefact
    # + 19-artefact files; o4mini: its single 22-artefact file). Filter by tag so
    # the o4mini-named files aren't mixed into the gpt55 join and vice versa.
    def load_mt(turn):
        out = {}
        for p in glob.glob(os.path.join(OUT_DIR, f"out_multiturn_t{turn}_*.jsonl")):
            if ("o4mini" in os.path.basename(p)) != (a.tag == "o4mini"):
                continue
            for l in open(p, encoding="utf-8"):
                r = json.loads(l)
                aid, d, val, run, ords, _ = r["custom_id"].split("|")
                out[(aid, d, val, int(run[1:]), ords[3:])] = score(extract_text(r["response"]["body"]))
        return out

    t2, t3 = load_mt(2), load_mt(3)

    rows, missing = [], defaultdict(int)
    for aid in sorted(dir_of):
        d = dir_of[aid]
        for val in ("valid", "invalid"):
            for run in range(20):
                for ords in ("012", "120", "201"):
                    rec = {
                        "S0": s0.get((aid, run)),
                        "t1": t1.get((aid, d, val, ORD2IDX0[ords], run)),
                        "t2": t2.get((aid, d, val, run, ords)),
                        "t3": t3.get((aid, d, val, run, ords)),
                    }
                    for k, v in rec.items():
                        if v is None:
                            missing[k] += 1
                    if any(v is None for v in rec.values()):
                        continue
                    rows.append({
                        "artefact": aid, "tier": aid[0], "direction": d,
                        "validity": val, "run": run, "ordering": ords, **rec,
                    })

    suffix = "" if a.tag == "gpt55" else f"_{a.tag}"
    out = os.path.join(HERE, "results", f"trajectories_challenge_22{suffix}.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "artefact", "tier", "direction", "validity", "run", "ordering",
            "S0", "t1", "t2", "t3"])
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} trajectories -> {out}")
    print(f"artefacts: {len({r['artefact'] for r in rows})} | expected 2640 rows "
          f"(22 x 2 val x 3 ord x 20 run)")
    if any(missing.values()):
        print(f"WARNING missing stage scores: {dict(missing)}")


if __name__ == "__main__":
    main()
