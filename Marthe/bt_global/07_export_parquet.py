"""Export bt_scores.csv to a self-contained parquet for coworkers.

Adds artefact title and domain (from the artefact JSONs) so the file is
readable without looking anything up.

Output: results/arguments_bt_global.parquet
Columns:
  item_id        -- unique key: artefact_id|direction|validity|idx
  artefact_id    -- e.g. "S01"
  artefact_title -- human-readable artefact name
  domain         -- artefact domain (e.g. "essay")
  direction      -- "lower" or "raise"
  validity       -- "valid" or "invalid"
  idx            -- argument index within the validity group (0, 1, 2)
  text           -- the pushback cycle-turn argument text
  fallacy_types  -- list of fallacy labels for invalid items; None for valid
  bt_rating      -- global BT log-strength score (one shared zero, 264 items)
"""
import os

import pandas as pd

from common import load_artefacts

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OUT_PATH = os.path.join(RESULTS_DIR, "arguments_bt_global.parquet")


def main():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "bt_scores.csv"))

    arts = load_artefacts()
    art_meta = {
        aid: {"artefact_title": d["title"], "domain": d["domain"]}
        for aid, d in arts.items()
    }
    meta_df = pd.DataFrame.from_dict(art_meta, orient="index").reset_index()
    meta_df.rename(columns={"index": "artefact_id"}, inplace=True)

    df = df.merge(meta_df, on="artefact_id", how="left")

    cols = [
        "item_id", "artefact_id", "artefact_title", "domain",
        "direction", "validity", "idx",
        "text", "fallacy_types", "bt_rating",
    ]
    df = df[cols]

    df.to_parquet(OUT_PATH, index=False, engine="pyarrow")
    print(f"Wrote {len(df)} rows → {OUT_PATH}")
    print(df.dtypes.to_string())


if __name__ == "__main__":
    main()
