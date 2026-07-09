"""Prepare BT scores in a clean local format.

The official ADS hinged weights are model-specific because the challenge
direction is model-specific. Script 03 computes those weights from this table
after it knows which direction each model used for each artefact.
"""

from __future__ import annotations

import pandas as pd

from common import DATA_DIR, RESULTS_DIR, ensure_dirs


BT_INPUT = DATA_DIR / "bt_scores.csv"
WEIGHT_OUTPUT = RESULTS_DIR / "prepared_bt_scores.csv"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    if "artefact" in df.columns and "artefact_id" not in df.columns:
        rename["artefact"] = "artefact_id"
    if "bt_score" in df.columns and "bt_rating" not in df.columns:
        rename["bt_score"] = "bt_rating"
    df = df.rename(columns=rename)

    required = {"artefact_id", "direction", "validity", "idx", "bt_rating"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{BT_INPUT} is missing columns: {sorted(missing)}")
    return df


def main() -> None:
    ensure_dirs()
    bt = normalize_columns(pd.read_csv(BT_INPUT))
    bt = bt.copy()
    bt["idx"] = bt["idx"].astype(int)
    bt["bt_rating"] = bt["bt_rating"].astype(float)

    used = bt[["artefact_id", "direction", "validity", "idx", "bt_rating"]].copy()
    used["raw_bt_abs_weight"] = used["bt_rating"].abs()
    used.to_csv(WEIGHT_OUTPUT, index=False)

    print(f"Wrote {WEIGHT_OUTPUT}")
    print(f"Rows: {len(used)}")
    print("Script 03 will compute model-specific ADS hinged weights.")


if __name__ == "__main__":
    main()
