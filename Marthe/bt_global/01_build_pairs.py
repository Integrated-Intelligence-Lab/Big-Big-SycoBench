"""Build pairs for the SINGLE-POOL BT argument-validity check.

Unlike ../bt_validation (which pairs only within an artefact x direction pool
of 6), every argument here lives in one global pool of ~264 items, so the fit
yields one common scale. A complete graph on 264 items is C(264,2)=34,716 pairs
-- far too many -- so by default we sparsify with a random `--degree`-regular
graph: each item is compared to *exactly* `--degree` others, so no argument is
judged more (or less) often than any other. (This uses
`nx.random_regular_graph`, the same exact-regular construction as the prior
difficulty-ranking project's `src/pairs.py`, rather than ../bt_persuasion's
union-of-matchings `regular_pairs`, which only lands each node at ~degree.)
Drawing those edges over the *whole* pool is what wires arguments from different
artefacts and different directions into one connected comparison graph, which is
exactly what a single interpretable scale requires. We assert connectivity
before writing (a disconnected graph would leave sub-scales that don't share a
zero).

Single round only: if you later want to *extend* a played round to a higher
degree while reusing every judgment already paid for, that needs the nested
CP-SAT b-matching (`incremental_samples` in that same `src/pairs.py`); a fresh
random-regular graph at a higher degree would not be nested.

Pairs now mix artefacts and directions, so each side carries its own
`artefact_id` / `direction` / `validity` for the judge builder to render.

    python 01_build_pairs.py                  # exact 20-regular
    python 01_build_pairs.py --degree 30
    python 01_build_pairs.py --complete       # full graph (slow/expensive)
"""
import argparse
import itertools
import json
import os

import networkx as nx

from common import load_artefacts, load_items

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def regular_pairs(n, degree, seed):
    """Exactly `degree`-regular simple graph: every node in exactly `degree`
    edges. Requires n*degree even (always true here, n is even) and degree < n.
    Same `nx.random_regular_graph` construction as the difficulty project's
    src/pairs.py."""
    if degree >= n - 1:
        return list(itertools.combinations(range(n), 2))
    g = nx.random_regular_graph(degree, n, seed=seed)
    return sorted((min(a, b), max(a, b)) for a, b in g.edges())


def is_connected(n, idx_pairs):
    """Union-find connectivity check over the index graph."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in idx_pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    return len({find(i) for i in range(n)}) == 1


def pair_record(items, ia, ib):
    a, b = items[ia], items[ib]
    return {
        "pair_id": f"{a['item_id']}__vs__{b['item_id']}",
        "item_id_a": a["item_id"],
        "item_id_b": b["item_id"],
        "artefact_id_a": a["artefact_id"],
        "artefact_id_b": b["artefact_id"],
        "direction_a": a["direction"],
        "direction_b": b["direction"],
        "validity_a": a["validity"],
        "validity_b": b["validity"],
        "text_a": a["text"],
        "text_b": b["text"],
    }


def build_pairs(items, degree, complete, seed):
    n = len(items)
    idx_pairs = (list(itertools.combinations(range(n), 2)) if complete
                 else regular_pairs(n, degree, seed))
    if not is_connected(n, idx_pairs):
        raise SystemExit("comparison graph is disconnected -- raise --degree")
    return [pair_record(items, ia, ib) for ia, ib in idx_pairs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--degree", type=int, default=20)
    ap.add_argument("--complete", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    arts = load_artefacts()
    items_map = load_items(arts)
    items = [items_map[k] for k in sorted(items_map)]  # deterministic index order

    pairs = build_pairs(items, a.degree, a.complete, a.seed)

    path = os.path.join(OUT_DIR, "pairs.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for p in pairs:
            json.dump(p, f, ensure_ascii=False)
            f.write("\n")

    mode = "complete" if a.complete else f"exact {a.degree}-regular (every argument in {a.degree} comparisons)"
    print(f"{len(arts)} artefacts, {len(items)} cycle-turn arguments, one global pool")
    print(f"graph: {mode}")
    print(f"Wrote {len(pairs)} pairs to {path}  (~{len(pairs)} judge calls)")


if __name__ == "__main__":
    main()
