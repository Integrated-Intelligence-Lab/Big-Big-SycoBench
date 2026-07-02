"""Fit ONE Bradley-Terry model over all judged pairs -- the whole point of this
folder. Every argument is a player in a single pool, so the ratings share one
zero and differences are interpretable as log-odds across artefacts and
directions: rating(i) - rating(j) = 1 means the judge prefers i over j at about
e:1 (~2.7:1) odds. (Contrast ../bt_validation, which fits one scale per
artefact x direction and so can't compare across pools.)

The `pool_id` column is retained on each row as an analysis label only; it is
NOT used to split the fit.
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
    fit_df = pairs.rename(columns={"item_id_a": "id_1", "item_id_b": "id_2"})[["id_1", "id_2", "a_wins"]]

    ratings = compute_bt_ratings(fit_df, alpha=a.alpha)

    rows = [{**items[iid], "bt_rating": rating} for iid, rating in ratings.items()]
    result = pd.DataFrame(rows).sort_values("bt_rating", ascending=False)

    out_path = os.path.join(RESULTS_DIR, "bt_scores.csv")
    result.to_csv(out_path, index=False)
    print(f"Wrote {len(result)} item ratings on one global scale to {out_path}")
    print(f"  range [{result['bt_rating'].min():.3f}, {result['bt_rating'].max():.3f}]")


if __name__ == "__main__":
    main()
