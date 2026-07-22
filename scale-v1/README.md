# scale-v1: 300-artefact benchmark generation pipeline

Generates 450 candidate artefacts (180 short / 180 medium / 90 long, drafted at
1.5x so the pilot screen can select the final 300 at 40/40/20), each with
2 directions x 2 arms x 3 standalone arguments, via the OpenAI Batch API.

Aligned with the ads_v2 protocol, not the old repo SCHEMA.md: 0-100 scale,
three distinct standalone arguments per arm (turn-1 attribution under cyclic
orderings), every argument ends with a rescore request.

## Cost controls

- Model: `gpt-5.6-sol` (config.py), Batch API pricing $2.50/$15 per 1M tokens.
- Hard cap: **$500** (`BUDGET_USD`). `submit.py` refuses any submission whose
  projection plus the ledger of actual spend would exceed it.
- Actual usage is read from each batch's results and appended to
  `out/ledger.json`.
- Projection: ~$180 artefacts + ~$65 pushbacks + ~$65 QC + regeneration ≈ $270.

## Key

Put the OpenAI key in `scale-v1/.openai_key` (single line) or export
`OPENAI_API_KEY`. `submit.py submit` first runs a model-id preflight against
/v1/models and aborts with the available gpt-5* ids if `gpt-5.6-sol` is wrong.

## Pipeline

```bash
cd scale-v1
python3 make_specs.py                      # 450 specs -> specs/specs.jsonl (done)

# Stage 1: artefacts
python3 build_batches.py artefacts         # -> out/batch_artefacts.jsonl (done)
python3 submit.py submit artefacts         # needs key; ~24h completion window
python3 submit.py status
python3 submit.py fetch artefacts
python3 parse_results.py artefacts         # validates length + anchor band
#   failures listed in out/failed_artefacts.txt; rebuild subset with --only

# Stage 2: pushbacks (12 arguments per artefact)
python3 build_batches.py pushbacks
python3 submit.py submit pushbacks && ... fetch && python3 parse_results.py pushbacks
#   -> merged records in candidates/json/C*.json

# Stage 3: adversarial QC (2 attack calls per artefact)
python3 build_batches.py qc
python3 submit.py submit qc && ... fetch && python3 parse_results.py qc
#   -> out/qc_verdicts.json + out/review_queue.md (human sign-off checklist)

python3 render_markdown.py                 # human-readable mirrors
```

Flagged arguments: fix by rebuilding those artefacts' pushbacks
(`build_batches.py pushbacks --only C012,C044`), or edit by hand and re-run QC
on them. Everything that passes QC + human review enters the candidate pool;
the final 300 are selected after the pilot screen (initial-score runs on 2-3
subject models, screening out artefacts whose mean initial score sits within
10 points of 50 for any pilot model).

## Provenance

Every record embeds a canary GUID (`config.CANARY`), the generator model id,
and `dataset_version`. Candidates are not the benchmark: the frozen v1 set is
selected after pilot screening; screened-out candidates seed the private
held-out pool.

## Not in this pipeline (by design)

- Pilot screening runs (subject models, not generation).
- BT judging of arguments (evaluation side).
- Human review: `out/review_queue.md` is the checklist; every flag needs a
  human decision plus a random 10% audit of passes.
