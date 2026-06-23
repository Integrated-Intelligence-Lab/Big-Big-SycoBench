"""Run the multi-turn pushback cycles via previous_response_id (SYNCHRONOUS).

This is the experiment the Batch API could not do: chain each turn off the prior
turn's stored response (`previous_response_id`) so the server carries the full
conversation -- including the model's prior reasoning -- instead of replaying a
hand-built message array. Same setup as the original neutral cycle experiment
(no developer message, original prompt, cycles arguments, both validity arms),
so the resulting trajectory can be compared against the stateless-replay batch.

Per (artefact, run): create S0 once (store=true), then fork the valid and
invalid chains from it; each chain is 3 sequential turns, every turn pointing at
the previous turn's response id. S0 is freshly sampled here (batch S0s aren't
chainable) -- fine because S0 is near-deterministic at this effort.

Requires OPENAI_API_KEY in the environment. Independent conversations run in a
thread pool; turns within a conversation are necessarily sequential. Results are
appended to a JSONL as each conversation finishes (safe to Ctrl-C and inspect).

    python 09_run_cycles_prid.py                 # all 3 artefacts, 20 runs
    python 09_run_cycles_prid.py --runs 1        # quick smoke test
    python 09_run_cycles_prid.py --artefacts M02 --runs 5 --workers 4
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

ARTEFACT_FILES = {"L01": "L01_scale100", "M02": "M02_scale100", "S02": "S02_scale100"}
ARTEFACT_DIR = "Marthe/artefacts"
OUT_PATH = "Marthe/results/pushback/prid_cycles_results.jsonl"
VALIDITIES = ["valid", "invalid"]
MODEL = "gpt-5.5-2026-04-23"
EFFORT = "medium"

client = None  # set in main() so --help / import work without credentials
_write_lock = threading.Lock()


def load_artefacts(ids):
    arts = {}
    for aid in ids:
        d = json.load(open(os.path.join(ARTEFACT_DIR, ARTEFACT_FILES[aid] + ".json"), encoding="utf-8"))
        d["_direction"] = "lower" if d["quality"] == "good" else "raise"
        arts[aid] = d
    return arts


def parse_score(text):
    m = re.search(r"-?\d+", text)
    return int(m.group()) if m else None


def create(input_text, previous_response_id=None, tries=3):
    """One Responses API call; retries a few times on transient errors."""
    last = None
    for _ in range(tries):
        try:
            kw = dict(model=MODEL, reasoning={"effort": EFFORT},
                      input=[{"role": "user", "content": input_text}], store=True)
            if previous_response_id:
                kw["previous_response_id"] = previous_response_id
            return client.responses.create(**kw)
        except Exception as e:  # noqa: BLE001  (toy script: retry then surface)
            last = e
    raise last


def run_conversation(art, run):
    """S0 + valid chain + invalid chain for one (artefact, run). Returns records."""
    aid = art["id"]
    direction = art["_direction"]
    recs = []

    s0 = create(f"{art['initial_prompt'].strip()}\n\nArtefact: {art['artefact']}\n")
    recs.append({"artefact": aid, "arm": "shared", "run": run, "turn": 0,
                 "score": parse_score(s0.output_text), "response_id": s0.id})

    for arm in VALIDITIES:
        prev_id = s0.id
        for k in (1, 2, 3):
            turn_text = art["pushback"][direction]["cycles"][arm][k - 1].strip()
            resp = create(turn_text, previous_response_id=prev_id)
            recs.append({"artefact": aid, "arm": arm, "run": run, "turn": k,
                         "score": parse_score(resp.output_text), "response_id": resp.id})
            prev_id = resp.id
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artefacts", nargs="*", default=list(ARTEFACT_FILES), choices=list(ARTEFACT_FILES))
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=OUT_PATH)
    a = ap.parse_args()

    global client
    client = OpenAI()

    arts = load_artefacts(a.artefacts)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w").close()  # fresh file

    jobs = [(art, run) for art in arts.values() for run in range(a.runs)]
    done, total = 0, len(jobs)
    print(f"Running {total} conversations ({total * 7} API calls) with {a.workers} workers -> {a.out}")

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futs = {pool.submit(run_conversation, art, run): (art["id"], run) for art, run in jobs}
        for fut in as_completed(futs):
            aid, run = futs[fut]
            try:
                recs = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"  FAILED {aid} run{run}: {e}")
                continue
            with _write_lock:
                with open(a.out, "a", encoding="utf-8") as f:
                    for r in recs:
                        json.dump(r, f, ensure_ascii=False)
                        f.write("\n")
            done += 1
            print(f"  [{done}/{total}] {aid} run{run} done")
    print("Done.")


if __name__ == "__main__":
    main()
