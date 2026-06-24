"""Shared loading for the BT *persuasion*-axis check on Arne's arguments.

Same comparative-judgment + Bradley-Terry machinery as ../bt_validation, but the
items are Arne's tier x level argument set and the axis being recovered is
PERSUASION (how convincing), not validity. Each pool is one artefact x push
direction; an item is one argument.

Arne's CSV (arne_arguments.csv) carries argument text + design labels but only
the artefact *title*, so artefact bodies (needed for judge context) are pulled
from Vincent's set -- all 5 of Arne's ids (S01-S05) match Vincent's by id+title.

Item fields kept for the level-separation analysis: validity (GOOD/BAD), tier
(T0-T6, the substance variant), level (L0-L4) and persuasion_load (operator
count, the persuasion axis).
"""
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "arne_arguments.csv")
ARTEFACT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(HERE)), "Vincent", "sycophancy-benchmark", "artefacts", "json"
)


def pool_id(artefact_id, direction):
    return f"{artefact_id}|{direction}"


def load_artefacts():
    """{id: artefact dict} from Vincent, restricted to the ids in Arne's CSV."""
    ids = {r["artefact_id"] for r in csv.DictReader(open(CSV_PATH, encoding="utf-8"))}
    arts = {}
    for fname in sorted(os.listdir(ARTEFACT_DIR)):
        if not fname.endswith(".json"):
            continue
        d = json.load(open(os.path.join(ARTEFACT_DIR, fname), encoding="utf-8"))
        if d["id"] in ids:
            arts[d["id"]] = d
    missing = ids - set(arts)
    if missing:
        raise SystemExit(f"artefact bodies missing from Vincent set: {sorted(missing)}")
    return arts


def load_items(arts=None):
    """{argument_id: {item fields}} -- one item per CSV row."""
    items = {}
    for r in csv.DictReader(open(CSV_PATH, encoding="utf-8")):
        iid = r["argument_id"]
        items[iid] = {
            "item_id": iid,
            "pool_id": pool_id(r["artefact_id"], r["direction"]),
            "artefact_id": r["artefact_id"],
            "direction": r["direction"],
            "validity": r["validity"],            # GOOD / BAD
            "tier": r["tier"],                    # substance variant
            "level": r["level"],                  # persuasion level
            "persuasion_load": int(r["persuasion_load"]),
            "operators": r["operators"],
            "text": r["argument"].strip(),
        }
    return items


def pools_from_items(items):
    pools = {}
    for it in items.values():
        pools.setdefault(it["pool_id"], []).append(it)
    return pools
