"""Assign the 450 artefacts to 10 annotators with controlled overlap.

Design (all constraints exact):
- Every annotator gets 24 short + 24 medium + 13 long = 61 files.
- Overlap pool: short 8 quintuple + 28 double, medium 8 + 28, long 7 + 12.
  => 91/450 = 20.2% reviewed by >=2 annotators, 23/450 = 5.1% by 5.
- Artefacts with QC-contested arguments are forced into the quintuple pool;
  the rest of the overlap pool is seeded-random (unbiased for IAA stats).

Output: out/assignments.json
Usage: python3 make_assignments.py
"""

import json
import random
from pathlib import Path

from config import ROOT

N_ANN = 10
QUOTA = {"short": 24, "medium": 24, "long": 13}
PLAN = {"short": {"quint": 8, "double": 28},
        "medium": {"quint": 8, "double": 28},
        "long": {"quint": 7, "double": 12}}
SEED = 20260722


def contested_ids() -> set:
    """Artefacts with an argument still flagged after the final QC pass."""
    slots = [tuple(s) for s in json.loads((ROOT / "out" / "must_fix_slots.json").read_text())]
    post = json.loads((ROOT / "out" / "qc_verdicts.json").read_text())
    out = set()
    for sid, d, arm, i in slots:
        for v in post[sid][arm]:
            if v["argument_key"] == f"{d}/{i}" and v["verdict"] == "flag":
                out.add(sid)
    return out


def main() -> None:
    rng = random.Random(SEED)
    index = json.loads((ROOT / "candidates" / "index.json").read_text())
    tiers = {"short": [], "medium": [], "long": []}
    for a in index["artefacts"]:
        tiers[a["length"]].append(a["id"])
    contested = contested_ids()

    anns = [f"{i:02d}" for i in range(1, N_ANN + 1)]
    assignment = {a: [] for a in anns}          # ann -> [artefact ids]
    reviewers = {}                              # artefact id -> [anns]

    for tier, ids in tiers.items():
        plan = PLAN[tier]
        cap = {a: QUOTA[tier] for a in anns}
        ids = sorted(ids)
        rng.shuffle(ids)
        # contested first into the quint pool, then random fill
        pri = sorted([i for i in ids if i in contested]) + \
              [i for i in ids if i not in contested]
        quints = pri[:plan["quint"]]
        rest = [i for i in ids if i not in quints]
        doubles = rest[:plan["double"]]
        singles = rest[plan["double"]:]

        def take(k: int) -> list:
            """k distinct annotators with most remaining capacity (ties random)."""
            order = sorted(anns, key=lambda a: (-cap[a], rng.random()))
            chosen = [a for a in order if cap[a] > 0][:k]
            assert len(chosen) == k, f"capacity exhausted in {tier}"
            for a in chosen:
                cap[a] -= 1
            return chosen

        for sid in quints:
            reviewers[sid] = take(5)
        for sid in doubles:
            reviewers[sid] = take(2)
        for sid in singles:
            reviewers[sid] = take(1)
        assert all(c == 0 for c in cap.values()), f"quota mismatch in {tier}: {cap}"

    for sid, revs in reviewers.items():
        for a in revs:
            assignment[a].append(sid)
    for a in anns:
        assignment[a].sort()
        assert len(assignment[a]) == sum(QUOTA.values())

    n_multi = sum(1 for r in reviewers.values() if len(r) >= 2)
    n_quint = sum(1 for r in reviewers.values() if len(r) == 5)
    stats = {
        "per_annotator": {a: {"total": len(assignment[a]),
                              **{t: sum(1 for s in assignment[a] if s in tiers[t])
                                 for t in tiers}} for a in anns},
        "multi_reviewed": n_multi, "multi_pct": round(100 * n_multi / 450, 1),
        "quint_reviewed": n_quint, "quint_pct": round(100 * n_quint / 450, 1),
        "contested_in_quints": sorted(contested),
    }
    out = {"seed": SEED, "annotators": assignment,
           "artefact_reviewers": {k: reviewers[k] for k in sorted(reviewers)},
           "stats": stats}
    (ROOT / "out" / "assignments.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(stats["per_annotator"], indent=0)[:400])
    print(f"multi >=2: {n_multi} ({stats['multi_pct']}%), "
          f"quint: {n_quint} ({stats['quint_pct']}%)")
    print(f"contested artefacts in quint pool: {len(contested)}")
    print("-> out/assignments.json")


if __name__ == "__main__":
    main()
