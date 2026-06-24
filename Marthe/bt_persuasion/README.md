# Persuasion-axis BT check (Arne's arguments)

Same comparative-judgment + Bradley-Terry pipeline as `../bt_validation`, but run
on **Arne's** tier × level argument set and recovering the **persuasion** axis
(how convincing an argument is) instead of validity.

## Why
Arne's design crosses two axes per artefact × direction:
- **validity** GOOD / BAD — the substance
- **persuasion** L0–L4 / `persuasion_load` — rhetoric operators layered on the
  *same* base argument

His own BT didn't separate cleanly. The likely cause is upstream of the BT
solver (ours is the same `choix.ilsr_pairwise`): the persuasion dimension and
near-duplicate level items confound a single pooled scale. This pipeline instead
asks the judge to rank **persuasiveness only** (ignore correctness), fits BT to
those verdicts, and checks whether the result rises with `persuasion_load` — a
manipulation check that the operators do what they claim — and whether validity
still matters once rhetoric is controlled.

> Ranking by persuasion is expected to "separate" by level largely **by
> construction** (the levels were built to be more persuasive). The scientific
> payoff is the **interaction**: does a high-load BAD argument out-persuade a
> low-load GOOD one? Read this alongside a validity-axis run, not instead of it.

## Data
- `arne_arguments.csv` — Arne's 700 arguments (5 artefacts × 2 directions × 2
  validity × 7 tiers × 5 levels = 70 items/pool, 10 pools).
- Artefact **bodies** come from Vincent's set (all 5 ids S01–S05 match by
  id + title); the CSV only carries titles.

## Pipeline
| | |
|---|---|
| `common.py` | load Arne's CSV → items/pools; pull artefact bodies from Vincent |
| `bt.py` | Bradley-Terry fit (`choix.ilsr_pairwise`), unchanged |
| `01_build_pairs.py` | within-pool pairs; **d-regular sparsified** by default (degree 14) — complete graph is 2415 pairs/pool ≈ 24k calls; `--complete` to override |
| `02_build_judge_batch.py` | persuasion judge prompt → `results/batch_in_persuasion_pairs.jsonl` |
| `03_process_judge_output.py` | parse `@<a>@/@<b>@` → `a_wins` |
| `04_compute_bt.py` | BT per pool → `results/bt_scores.csv` |
| `05_plot_persuasion.py` | BT vs `persuasion_load`, split by validity; per-pool Spearman |

```
python 01_build_pairs.py            # default d-regular, degree 14
python 02_build_judge_batch.py
# submit results/batch_in_persuasion_pairs.jsonl  ->  <judge_output.jsonl>
python 03_process_judge_output.py <judge_output.jsonl>
python 04_compute_bt.py
python 05_plot_persuasion.py
```

## Cost knob
70 items/pool makes the complete graph expensive. Default `--degree 14` issues
~490 pairs/pool (~4.9k judge calls across 10 pools). Raise the degree for more
precise BT estimates, or `--complete` for the full graph.
