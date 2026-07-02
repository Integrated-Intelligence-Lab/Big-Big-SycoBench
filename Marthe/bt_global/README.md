# Single-pool argument validity via Bradley-Terry

Same items, judge, and BT machinery as [`../bt_validation`](../bt_validation),
but every argument is fitted in **one global pool** instead of one pool per
artefact x direction. The point is interpretability: with a single scale, a
difference in BT rating is a log-odds the judge prefers one argument over
another, and that comparison holds **across artefacts and across push
directions** — not just inside one 6-item pool.

## What changes vs `bt_validation`

| | `bt_validation` | `bt_global` (here) |
|---|---|---|
| pools | one per artefact x direction | **one global pool** |
| a pair | same artefact, same direction | may mix artefact and/or direction |
| BT fit | per pool, no shared zero | **single fit, one shared zero** |
| scores comparable | only within a pool | **across everything** |
| judge prompt | "two arguments for {lowering/raising} its score" | artefact-aware + direction-agnostic (below) |
| graph | complete (15 pairs/pool) | exact d-regular over all ~264 items (every argument in d comparisons) |

The items are identical: Vincent's pushback **cycle turns** (3 valid + 3 invalid
per artefact x direction), the messages actually sent to the model under test.

## Two design choices that make a single scale valid

- **Direction-agnostic prompt.** The judge is *not* told which argument argues
  to raise and which to lower. Each cycle message already makes its direction
  plain in its text, so the label is redundant — and stating it risks nudging
  the judge to grade whether the score *should* move that way
  (direction-appropriateness) rather than how sound the argument is. Since
  "valid" is a property of the reasoning, we keep the judge on the reasoning.
  The criterion still never depends on direction, so all 264 arguments sit on
  one ruler; you can refit within a single direction afterward using each item's
  `direction` field, and a systematic lower-vs-raise gap is then a *finding*
  (one side is harder to argue well), surfaced by `05`. *(Worth a spot-check
  after a real run: read a few cross-direction verdicts to confirm the judge
  isn't misreading what an argument argues — validity is largely
  direction-independent, so this should be rare.)*

- **Artefact-aware prompt.** When the two arguments target different artefacts,
  both bodies are shown and each argument is bound to its own — context the
  judge needs to check the claims (this is separate from direction, which stays
  unnamed). See `VALIDITY_PROMPT` / `build_input` in `02_build_judge_batch.py`.

## Connectivity and cost

A complete graph on ~264 items is 34,716 pairs. We sparsify with an **exact
d-regular** graph (`nx.random_regular_graph`, the same construction as the
difficulty project's `src/pairs.py`): every argument is in **exactly** `--degree`
comparisons — none judged more often than another — unlike `../bt_persuasion`'s
union-of-matchings, which only lands each node at *roughly* the degree. Drawing
edges over the whole pool is what bridges artefacts and directions into one
connected graph. `01` asserts connectivity (union-find) before writing — a
disconnected graph would leave sub-scales with no shared zero. Default
`--degree 20` → exactly `264·20/2 = 2640` pairs. Each unordered pair is judged
once (no order-swap position-bias control), matching `bt_validation`'s
convention.

This is a **single-round** design. To extend a played round to a higher degree
while reusing every judgment already collected, you'd need the nested CP-SAT
b-matching (`incremental_samples` in that same `src/pairs.py`) — a fresh
random-regular graph at a higher degree is not nested. Not wired in here.

## Pipeline

```
pip install choix networkx
python 01_build_pairs.py            # --degree N | --complete
python 02_build_judge_batch.py
# submit results/batch_in_validity_pairs.jsonl with the judge model's batch API -> <judge_output.jsonl>
python 03_process_judge_output.py <judge_output.jsonl>
python 04_compute_bt.py             # single global fit -> results/bt_scores.csv
python 05_plot_validation.py        # same plots as ../bt_validation
```

| file | role |
|---|---|
| `common.py` | loads artefacts + the cycle-turn arguments (keeps `artefact_id`/`direction` per item; `pool_id` is an analysis label only) |
| `bt.py` | Bradley-Terry fit via `choix.ilsr_pairwise` |
| `01_build_pairs.py` | d-regular pairs over the whole pool, connectivity-checked |
| `02_build_judge_batch.py` | artefact-aware, direction-agnostic validity-judge batch |
| `03_process_judge_output.py` | parses judge output into `a_wins` (keyed on `custom_id`) |
| `04_compute_bt.py` | one BT fit over all pairs -> `results/bt_scores.csv` |
| `05_plot_validation.py` | separation + outliers + raincloud, same as `../bt_validation` (now on one global scale) |
| `06_plot_singleshot_bt_vs_delta.py` | optional: single-shot isolated score shift vs the global BT rating, per SUT model (`--tag gpt55`) — mirrors `../scripts/15_...` but on the global scale |

## Outputs

- `results/bt_scores.csv` — every argument with its `bt_rating` on the one
  global scale, sorted high to low (plus `artefact_id`, `direction`,
  `validity`, `fallacy_types` for slicing).
- `results/separation_summary.csv` — per pool (`artefact|direction`, an analysis
  label): `clean_separation` (all valid items outrank all invalid) and `auc`.
- `results/outliers.csv` — every crossover item (an invalid argument rated above
  its pool's weakest valid one, or vice versa), with `fallacy_types`.
- `results/bt_separation.png` — pooled valid (green) vs invalid (red) KDE on the
  global scale, with a raincloud strip of every argument and the pooled AUC.
- `results/bt_vs_delta_<tag>.png` (from `06`) — per argument, the single-shot
  isolated score shift (raw points, and headroom-normalized) against the global
  BT rating, coloured by validity, with the Spearman fit. Covers the 132
  arguments that have single-shot runs (one direction per artefact).
