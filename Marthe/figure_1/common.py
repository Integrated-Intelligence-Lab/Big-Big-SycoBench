"""Shared constants and loaders for the figure_1 multi-shot experiment.

Three artefacts (L01, M02, S02) × both directions × valid + invalid × all 6
argument orderings, plus 0-shot (argument in initial prompt) and 1-shot
(argument after a neutral S0).

Scale: all artefacts loaded from Marthe/artefacts/*_scale100.json, which
already have the rescore prompt and cycle texts on the 1-100 scale.  No
further text substitution is needed.
"""
import itertools
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ARTEFACT_DIR = os.path.join(HERE, "..", "artefacts")
RESULTS_DIR = os.path.join(HERE, "results")

ARTEFACTS = ["L01", "M02", "S02"]

# Direction chosen for the original multi-turn/singleshot experiments
# (based on S0: lower if S0 > 50, raise if S0 < 50)
ORIGINAL_DIR = {"L01": "lower", "M02": "raise", "S02": "lower"}
OPPOSITE_DIR = {"L01": "raise",  "M02": "lower", "S02": "raise"}

VALIDITIES = ["valid", "invalid"]

# All 6 orderings of the 3 cycle arguments (0, 1, 2)
ORDERINGS = list(itertools.permutations(range(3)))          # 6 tuples
ORIGINAL_ORDERING = (0, 1, 2)   # already exists in live/prid multi-turn data

MODEL = "gpt-5.5-2026-04-23"
EFFORT = "medium"


def load_artefact(aid):
    return json.load(
        open(os.path.join(ARTEFACT_DIR, f"{aid}_scale100.json"), encoding="utf-8")
    )


def load_all_artefacts():
    return {aid: load_artefact(aid) for aid in ARTEFACTS}


def initial_prompt_for_zeroshot(art):
    """Initial prompt with the trailing 'Reply with the number.' stripped.

    For 0-shot, the cycle text already ends with its own rescore ask, so we
    avoid having two competing asks in the same message.
    """
    p = art["initial_prompt"].rstrip()
    p = re.sub(r"[.]*\s*Reply with the number\.?\s*$", "", p, flags=re.I).strip()
    return p


def load_s0_from_runlog(path=None):
    """Load S0 scores from the live multi-turn runlog.

    Returns {artefact_id: {run_index: score}}.
    The runlog lives at Marthe/results/pushback/live/live_runlog.jsonl.
    """
    if path is None:
        path = os.path.join(HERE, "..", "results", "pushback", "live", "live_runlog.jsonl")
    s0 = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r["kind"] != "s0":
            continue
        aid, run_str = r["custom_id"].split("_run")
        if aid not in s0:
            s0[aid] = {}
        s0[aid][int(run_str)] = r["score"]
    return s0
