"""Fit a Bradley-Terry model per pool from the judge's pairwise verdicts.

Each artefact x direction pool is fit independently -- pools never share a
comparison, so there's no joint scale to estimate. See bt.py for the fitting
function (same `choix.ilsr_pairwise`-based implementation as the prior
difficulty-ranking project's `src/bt.py`).
"""
import argparse
import os

import pandas as pd

from common import load_artefacts, load_items
from bt import compute_bt_ratings

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.01,
                    help="choix.ilsr_pairwise regularization strength")
    a = ap.parse_args()

    arts = load_artefacts()
    items = load_items(arts)

    pairs = pd.read_json(os.path.join(RESULTS_DIR, "pairs_with_results.jsonl"), lines=True)

    rows = []
    for pool_id, g in pairs.groupby("pool_id"):
        pool_df = g.rename(columns={"item_id_a": "id_1", "item_id_b": "id_2"})[["id_1", "id_2", "a_wins"]]
        ratings = compute_bt_ratings(pool_df, alpha=a.alpha)
        for iid, rating in ratings.items():
            rows.append({**items[iid], "bt_rating": rating})

    result = pd.DataFrame(rows)
    out_path = os.path.join(RESULTS_DIR, "bt_scores.csv")
    result.to_csv(out_path, index=False)
    print(f"Wrote {len(result)} item ratings across {result['pool_id'].nunique()} pools to {out_path}")


if __name__ == "__main__":
    main()
