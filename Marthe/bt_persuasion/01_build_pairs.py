"""Build within-pool argument pairs for the persuasion-axis BT check.

Pools are big here (70 items = validity x 7 tiers x 5 levels), so a complete
graph is 2415 pairs/pool = ~24k judge calls across 10 pools. By default we
sparsify with a random d-regular graph (each item compared to `--degree`
others), which keeps every item equally connected at a fraction of the cost.
Pass --complete to issue the full graph instead.

Comparisons stay within a pool (artefact x direction); the judge needs the
artefact context and cross-artefact comparisons aren't meaningful.

    python 01_build_pairs.py                 # d-regular, degree 14
    python 01_build_pairs.py --degree 20
    python 01_build_pairs.py --complete
"""
import argparse
import itertools
import json
import os
import random

from common import load_items, pools_from_items

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def regular_pairs(n, degree, rng):
    """Near-d-regular simple graph as the union of `degree` random matchings.

    Each random matching gives every node exactly one edge; unioning `degree`
    of them (and deduping) lands every node at ~degree comparisons (a few fewer
    where two matchings happen to draw the same pair). Far more even than
    whole-graph rejection sampling, which is hopeless at this degree.
    """
    if degree >= n - 1:
        return list(itertools.combinations(range(n), 2))
    edges = set()
    for _ in range(degree):
        perm = list(range(n))
        rng.shuffle(perm)
        for k in range(0, n - 1, 2):
            a, b = perm[k], perm[k + 1]
            edges.add((min(a, b), max(a, b)))
    return sorted(edges)


def build_pairs(pools, degree, complete, seed):
    rng = random.Random(seed)
    pairs = []
    for pid, items in pools.items():
        n = len(items)
        idx_pairs = (list(itertools.combinations(range(n), 2)) if complete
                     else regular_pairs(n, degree, rng))
        for ia, ib in idx_pairs:
            a, b = items[ia], items[ib]
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--degree", type=int, default=14)
    ap.add_argument("--complete", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    items = load_items()
    pools = pools_from_items(items)
    pairs = build_pairs(pools, a.degree, a.complete, a.seed)

    path = os.path.join(OUT_DIR, "pairs.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for p in pairs:
            json.dump(p, f, ensure_ascii=False)
            f.write("\n")

    sizes = [len(v) for v in pools.values()]
    mode = "complete" if a.complete else f"d-regular (degree {a.degree})"
    print(f"{len(items)} arguments, {len(pools)} pools ({sizes[0]} items each), graph: {mode}")
    print(f"Wrote {len(pairs)} pairs to {path}  (~{len(pairs)} judge calls)")


if __name__ == "__main__":
    main()
