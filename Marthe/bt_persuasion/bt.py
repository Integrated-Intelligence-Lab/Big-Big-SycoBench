"""Bradley-Terry fit -- same implementation as
`code-Marthe-gpt-does-a-good-job/difficulty GitHub/src/bt.py` (on OneDrive,
not in this repo), which used choix.ilsr_pairwise to rank problems by
difficulty from pairwise "which is harder" judgments. Swapped here for
argument validity instead of problem difficulty.

Requires the `choix` package (`pip install choix`).
"""
import choix
import numpy as np
import pandas as pd


def compute_bt_ratings(df, alpha=0.01):
    # identify unique player IDs
    players = np.unique(df[['id_1', 'id_2']].values.ravel())
    pid_map = {pid: i for i, pid in enumerate(players)}

    # build win/lose pairs for ilsr_pairwise
    pairs = []
    for a, b, aw in zip(df.id_1, df.id_2, df.a_wins):
        if aw:
            pairs.append((pid_map[a], pid_map[b]))
        else:
            pairs.append((pid_map[b], pid_map[a]))

    # compute strengths
    strengths = choix.ilsr_pairwise(len(players), pairs, alpha=alpha)

    return pd.Series(strengths, index=players, name='bt_rating')
