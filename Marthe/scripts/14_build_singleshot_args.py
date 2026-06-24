"""Single-shot isolation, STAGE 2: fire each cycle argument ONCE from S0.

For every artefact, each of the 6 cycle arguments (3 valid + 3 invalid) is
presented alone as a single pushback turn, replayed statelessly on top of that
run's own S0 score:

    [ user: initial_prompt + artefact ,  assistant: <S0> ,  user: <argument> ]  -> rescore

No accumulation, no turn order -> the rescore minus S0 is that argument's
isolated effect, cleanly matchable to its BT validity rating (one point per
argument, vs the confounded per-turn deltas of the multi-turn cycles).

Push direction is chosen per artefact from the model's own mean S0 (lower if
S0>50 else raise) so the score has headroom to move; the matching BT scores are
that direction's. Borderline artefacts (S0 near 50) are printed.

Argument text is rescaled in place: 1-10 -> 1-100 and justification asks dropped
(all 264 cycle args carry a 1-10 ref + a justification ask, so this is reliable
and avoids stripping the heterogeneous rescore tail).

custom_id: {artefact}|{direction}|{validity}|idx{idx}|r{run}

    python 14_build_singleshot_args.py --model gpt-5.5-2026-04-23 --tag gpt55 \
        --s0-output Marthe/results/singleshot/gpt55/<s0_output>.jsonl
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict
from statistics import mean

ARTEFACT_DIR = "Vincent/sycophancy-benchmark/artefacts/json"
N_RUNS = 20
EFFORT = "medium"
VALIDITIES = ["valid", "invalid"]


def scale_initial(p):
    s = p.replace("1 to 10", "1 to 100").replace("10 is excellent", "100 is excellent")
    return s.replace(" and a brief justification", "")


def scale_arg(s):
    s = re.sub(r"1\s*-\s*10\b", "1-100", s)          # "1-10" / "1 - 10" -> "1-100"
    s = s.replace("1 to 10", "1 to 100")
    for j in (" and a brief justification", ", with a brief justification",
              " with a brief justification"):
        s = s.replace(j, "")
    return s.strip()


def extract_text(body):
    return "\n".join(
        c.get("text", "")
        for it in body.get("output", [])
        if it.get("type") == "message"
        for c in it.get("content", [])
        if c.get("type") == "output_text"
    )


def parse_score(t):
    m = re.search(r"-?\d+", t)
    return int(m.group()) if m else None


def load_s0(path):
    out = {}
    for l in open(path, encoding="utf-8"):
        r = json.loads(l)
        aid, run = r["custom_id"].split("_run")
        out[(aid, int(run))] = parse_score(extract_text(r["response"]["body"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--s0-output", required=True, help="this model's stage-1 S0 batch output")
    a = ap.parse_args()

    s0 = load_s0(a.s0_output)
    arts = {json.load(open(f, encoding="utf-8"))["id"]: json.load(open(f, encoding="utf-8"))
            for f in glob.glob(os.path.join(ARTEFACT_DIR, "*.json"))}

    # per-artefact mean S0 -> headroom direction
    mean_s0 = {aid: mean([s0[(aid, r)] for r in range(N_RUNS) if s0.get((aid, r)) is not None])
               for aid in arts}
    direction = {aid: ("lower" if m > 50 else "raise") for aid, m in mean_s0.items()}
    borderline = [aid for aid, m in mean_s0.items() if 40 <= m <= 60]
    if borderline:
        print("borderline S0 (40-60), direction may be ambiguous:",
              ", ".join(f"{a}={mean_s0[a]:.0f}->{direction[a]}" for a in sorted(borderline)))

    reqs, missing = [], 0
    for aid, art in arts.items():
        d = direction[aid]
        for val in VALIDITIES:
            for idx, raw in enumerate(art["pushback"][d]["cycles"][val]):
                arg = scale_arg(raw)
                for run in range(N_RUNS):
                    s = s0.get((aid, run))
                    if s is None:
                        missing += 1
                        continue
                    reqs.append({
                        "custom_id": f"{aid}|{d}|{val}|idx{idx}|r{run}",
                        "method": "POST",
                        "url": "/v1/responses",
                        "body": {
                            "model": a.model,
                            "reasoning": {"effort": EFFORT},
                            "input": [
                                {"role": "user",
                                 "content": f"{scale_initial(art['initial_prompt']).strip()}\n\nArtefact: {art['artefact']}\n"},
                                {"role": "assistant", "content": str(s)},
                                {"role": "user", "content": arg},
                            ],
                            "store": True,
                        },
                    })
    if missing:
        print(f"WARNING: skipped {missing} runs with missing S0")

    out_dir = os.path.join("Marthe/results/singleshot", a.tag)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "args_batch_in.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in reqs:
            json.dump(r, f, ensure_ascii=False)
            f.write("\n")
    print(f"Wrote {len(reqs)} requests "
          f"({len(arts)} artefacts x 6 args x {N_RUNS} runs, model={a.model}) to {out_path}")


if __name__ == "__main__":
    main()
