# Argument-validity check via Bradley-Terry

Verifies that Vincent's preliminary sycophancy-benchmark labels — "valid" vs
"invalid" pushback argument — actually hold up under independent pairwise
judgment, rather than resting on the hand-assigned label alone.

## Idea

For each artefact x push-direction (`lower`/`raise`), there are 6 cycle
turns: 3 labeled `valid`, 3 labeled `invalid`
(`Vincent/sycophancy-benchmark/artefacts/json/<id>.json`, see its
`SCHEMA.md`) — these are the actual messages sent turn-by-turn to the model
under test, as opposed to `core_arguments` (a condensed pre-cycle summary
not used as-is in any live run). Instead of asking a judge to rate each
argument's validity in isolation (an absolute judgment, known to be noisier
and more anchoring-prone than a relative one), we have a judge compare
arguments pairwise — "which of these two is the more valid/substantive
reason to revise the score" — and fit a
[Bradley-Terry model](https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model)
to turn those win/loss judgments into a continuous validity-strength score
per argument. If the resulting scores cleanly separate the `valid`- and
`invalid`-labeled arguments within each pool, that's independent evidence
the dichotomy isn't just our own call. Any crossover (an "invalid" argument
out-scoring a "valid" one) is exactly the kind of outlier worth a closer
look — the same thing flagged informally for L01-invalid in the original
sycophancy experiments (see `Marthe/README.md`).

This reuses the comparative-judgment idea, and the BT fit itself, from the
prior difficulty-ranking project
(`code-Marthe-gpt-does-a-good-job/difficulty GitHub`, on OneDrive, not in
this repo): `bt.py` is the same `choix.ilsr_pairwise`-based
`compute_bt_ratings(df, alpha)` that project used to rank problems by
difficulty from pairwise "which is harder" judgments, swapped here for
argument validity instead of problem difficulty.

## Design choices

- **Comparisons stay within a pool** (one artefact x direction). Judging
  "more valid" needs the artefact's text for context, and two different
  artefacts can't share one comparison meaningfully — so there's no
  cross-artefact BT scale, only a per-pool one.
- **Complete graph, not sparsified.** At n=6 items (15 unordered pairs) per
  pool there's no need for the random-regular-graph sampling the difficulty
  project used for hundreds of items — just run every pair.
- **No bipartite-only design.** Pairs aren't restricted to valid-vs-invalid;
  a pool's 3 valid-vs-valid and 3 invalid-vs-invalid pairs are included too
  (the full `C(6,2)=15`). Showing the judge only deliberately-mismatched
  pairs would let it learn the schedule's structure ("there's always a
  clear winner here") rather than judging each pair on its merits — and it
  loses a free check on whether a label group is internally homogeneous.
- **Each unordered pair is judged once** (15 comparisons per pool), matching
  the prior difficulty project's pairing convention. No order-swap control
  for judge position bias (the judge favoring whichever argument happens to
  be "A") -- a possible follow-up if the results look position-sensitive.
- **Judge model differs from the system under test.** Grading "is this
  argument valid" with the same model whose sycophancy is being measured
  (`gpt-5.5-2026-04-23`) wouldn't be independent evidence. `JUDGE_MODEL` in
  `02_build_judge_batch.py` is currently `gpt-5.4-mini-2026-03-17`. For
  stronger evidence, repeat the whole pipeline with a second judge from a
  different model family and compare.
- **Cycle turns, not core arguments.** `core_arguments` are a condensed
  summary that never gets sent to the model under test as-is; the cycle
  messages are what's actually used in the multi-turn experiments, so
  they're what needs validating. Each cycle message ends with a rescore
  request phrased differently per cycle (not a fixed literal string, so it
  can't be reliably stripped). The judge prompt does not currently tell the
  model to ignore that trailing request -- worth checking the judge's raw
  responses for cases where it answers with a score instead of a letter
  (`03_process_judge_output.py` drops any unparseable response with a
  warning, so check that warning count after a real run).

## Pipeline

| | |
|---|---|
| `common.py` | loads artefacts, extracts the 264 cycle-turn arguments (22 artefacts x 2 directions x 3 valid + 3 invalid), assigns pool/item ids |
| `bt.py` | Bradley-Terry fit via `choix.ilsr_pairwise` (requires `pip install choix`) |
| `01_build_pairs.py` | builds the within-pool complete-graph pairs (one per unordered pair) -> `results/pairs.jsonl` |
| `02_build_judge_batch.py` | turns pairs into an OpenAI-batch-shaped input with the validity-judge prompt -> `results/batch_in_validity_pairs.jsonl` |
| `03_process_judge_output.py` | parses the judge's batch output into `a_wins` -> `results/pairs_with_results.jsonl` |
| `04_compute_bt.py` | fits BT per pool -> `results/bt_scores.csv` |
| `05_plot_validation.py` | per-pool clean-separation check, pooled/per-pool AUC, crossover-outlier list, scatter plot |

```
pip install choix
python 01_build_pairs.py
python 02_build_judge_batch.py
# submit results/batch_in_validity_pairs.jsonl with the judge model's batch API -> <judge_output.jsonl>
python 03_process_judge_output.py <judge_output.jsonl>
python 04_compute_bt.py
python 05_plot_validation.py
```

Currently 22 of the planned 24 artefacts are present (`L05`, `L06` missing
from Vincent's preliminary set) -> 22 x 2 directions = 44 pools, 264
cycle-turn arguments, 660 pairwise judgments.

## Outputs

- `results/separation_summary.csv` — per pool: `clean_separation` (all 3
  valid items outrank all 3 invalid items) and `auc` (P(random valid item
  outranks random invalid item), 1.0 = perfect, 0.5 = chance).
- `results/outliers.csv` — every crossover item (an invalid argument rated
  above its pool's weakest valid one, or vice versa) — the candidates for
  manual re-reading. Includes `fallacy_types` for invalid items, so you can
  see which fallacy family tends to land the crossovers.
- `results/bt_separation.png` — one column per pool, valid (green) vs
  invalid (red) ratings, for a global at-a-glance view.

This validates the label, it doesn't replace the small human audit
discussed earlier — that's still worth doing on whatever the outlier list
surfaces, plus a spot-check of a clean-separation pool, since a judge model
agreeing with the label isn't the same as a human agreeing with it.
