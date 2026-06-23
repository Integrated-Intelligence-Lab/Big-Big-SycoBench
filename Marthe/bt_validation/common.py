"""Shared loading for the BT argument-validity check.

Loads Vincent's preliminary sycophancy-benchmark artefacts and extracts the
pushback CYCLE turns (3 valid + 3 invalid per artefact x direction) as the
items to be pairwise-compared -- these are the actual messages sent to the
model under test in the multi-turn experiments, as opposed to
`core_arguments`, which are a condensed pre-cycle summary not used as-is in
any live run.

Each cycle message ends with a rescore request ("...what score would you
now give?"), phrased differently per artefact/cycle -- not a fixed literal
suffix, so it isn't stripped out; the judge prompt instructs the model to
judge the argument, not the rescore question.
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
    return f"{artefact_id}|{direction}"


def item_id(artefact_id, direction, validity, idx):
    return f"{artefact_id}|{direction}|{validity}|{idx}"


def load_items(arts):
    """One item per cycle turn.

    Returns {item_id: {item_id, pool_id, artefact_id, direction, validity, idx,
    text, fallacy_types}}. `fallacy_types` is the tag list for that invalid
    cycle (e.g. ["authority"]) and None for valid items.
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


def pools_from_items(items):
    pools = {}
    for it in items.values():
        pools.setdefault(it["pool_id"], []).append(it)
    return pools
