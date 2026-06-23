"""Build all within-pool argument pairs for the BT argument-validity check.

Each artefact x push-direction is its own pool of 6 cycle turns (3 valid,
3 invalid). Comparisons only ever happen within a pool, never across
artefacts: judging "which argument is more valid" requires the artefact
context, and two different artefacts can't share one comparison meaningfully.

At this pool size (6 items -> 15 unordered pairs) we just run the complete
graph rather than sparsifying. Each unordered pair is issued once (slot A/B
assignment follows itertools.combinations order), matching the prior
difficulty-ranking project's pairing convention -- no order-swap duplicate.
"""
import itertools
import json
import os

from common import load_artefacts, load_items, pools_from_items

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def build_pairs(pools):
    pairs = []
    for pid, items in pools.items():
        for a, b in itertools.combinations(items, 2):
            pairs.append({
                "pair_id": f"{a['item_id']}__vs__{b['item_id']}",
                "pool_id": pid,
                "item_id_a": a["item_id"],
                "item_id_b": b["item_id"],
                "text_a": a["text"],
                "text_b": b["text"],
                "validity_a": a["validity"],
                "validity_b": b["validity"],
            })
    return pairs


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    arts = load_artefacts()
    items = load_items(arts)
    pools = pools_from_items(items)
    pairs = build_pairs(pools)

    path = os.path.join(OUT_DIR, "pairs.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for p in pairs:
            json.dump(p, f, ensure_ascii=False)
            f.write("\n")

    print(f"{len(arts)} artefacts, {len(items)} cycle-turn arguments, {len(pools)} pools")
    print(f"Wrote {len(pairs)} pairs to {path}")


if __name__ == "__main__":
    main()
