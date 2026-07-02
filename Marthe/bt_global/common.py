"""Shared loading for the BT argument-validity check on ONE GLOBAL POOL.

Same items and machinery as ../bt_validation -- Vincent's pushback CYCLE turns
(3 valid + 3 invalid per artefact x direction), the actual messages sent to the
model under test in the multi-turn experiments. The only difference is the
pooling: bt_validation fits one Bradley-Terry scale per artefact x direction
(scores comparable only within a pool); here every argument goes into a SINGLE
pool, so the fitted scores live on one common scale and differences are
interpretable across artefacts and across push directions.

For that single scale to mean one thing, the judge can no longer assume both
arguments concern the same artefact and push the same way: a pair may now mix
artefacts and mix directions. So each item keeps its `artefact_id` and
`direction` as first-class fields (the pair builder carries them per side, and
the judge prompt names them) -- see 01_build_pairs.py / 02_build_judge_batch.py.
`pool_id` (artefact|direction) is kept only as an analysis label, not as a
fitting boundary.

Each cycle message ends with a rescore request ("...what score would you now
give?"), phrased differently per artefact/cycle -- not a fixed literal suffix,
so it isn't stripped; the judge prompt instructs the model to judge the
argument, not the rescore question.
"""
import json
import os

ARTEFACT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "Vincent", "sycophancy-benchmark", "artefacts", "json",
)
DIRECTIONS = ["lower", "raise"]
VALIDITIES = ["valid", "invalid"]


def load_artefacts():
    arts = {}
    for fname in sorted(os.listdir(ARTEFACT_DIR)):
        if not fname.endswith(".json"):
            continue
        d = json.load(open(os.path.join(ARTEFACT_DIR, fname), encoding="utf-8"))
        arts[d["id"]] = d
    return arts


def pool_id(artefact_id, direction):
    """Kept as an analysis label only -- the global fit ignores it."""
    return f"{artefact_id}|{direction}"


def item_id(artefact_id, direction, validity, idx):
    return f"{artefact_id}|{direction}|{validity}|{idx}"


def load_items(arts):
    """One item per cycle turn, across every artefact x direction.

    Returns {item_id: {item_id, pool_id, artefact_id, direction, validity, idx,
    text, fallacy_types}}. `fallacy_types` is the tag list for that invalid
    cycle (e.g. ["authority"]) and None for valid items. Identical to
    ../bt_validation; only the downstream pooling differs.
    """
    items = {}
    for art in arts.values():
        for direction in DIRECTIONS:
            pb = art["pushback"][direction]
            for validity in VALIDITIES:
                cycles = pb["cycles"][validity]
                for idx, text in enumerate(cycles):
                    iid = item_id(art["id"], direction, validity, idx)
                    fallacy_types = (
                        pb["invalid_fallacy_types"][idx] if validity == "invalid" else None
                    )
                    items[iid] = {
                        "item_id": iid,
                        "pool_id": pool_id(art["id"], direction),
                        "artefact_id": art["id"],
                        "direction": direction,
                        "validity": validity,
                        "idx": idx,
                        "text": text.strip(),
                        "fallacy_types": fallacy_types,
                    }
    return items
