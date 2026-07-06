# Previous-response Batch Pipeline

This folder builds the 20 + 5 `previous_response_id` arm for the ADS benchmark.

The design is staged because each continuation request needs the response ID produced by the previous batch:

| stage | requests per model | role |
|---|---:|---|
| S0 | 22 artefacts x 20 runs = 440 | independent neutral baseline scores, stored for chaining |
| t1 | 22 artefacts x 5 runs x 2 validity arms x 3 cyclic orderings = 660 | first continuation from the S0 response ID |
| t2 | 660 | second continuation from the t1 response ID |
| t3 | 660 | third continuation from the t2 response ID |

Per model, the final trajectory table has 660 complete rows. Across the five default models, the full run is 12,100 Responses API requests before Batch discounting.

## Model Tags

| tag | model | default reasoning |
|---|---|---|
| `gpt55` | `gpt-5.5-2026-04-23` | `medium` |
| `gpt52` | `gpt-5.2-2025-12-11` | `medium` |
| `gpt5` | `gpt-5-2025-08-07` | `medium` |
| `o3` | `o3-2025-04-16` | `medium` |
| `gpt41` | `gpt-4.1-2025-04-14` | omitted |

Use `--reasoning-effort none` to omit reasoning for any tag, or pass an explicit effort to override the default.

## Setup

Use a virtual environment and install the OpenAI SDK once:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U openai
export OPENAI_API_KEY="..."
```

## One-command Run

Use this for the main experiment. It builds each stage, submits the batch, waits for completion, downloads outputs, and advances to the next stage automatically. It also resumes from existing batch metadata and downloaded outputs unless `--force` is passed.

Smoke test:

```bash
SMOKE_DIR="$(mktemp -d)"

python Andres/prid_batch_pipeline/run_prid_pipeline.py \
  --tags gpt55 \
  --artefact-limit 1 \
  --s0-runs 1 \
  --continuation-runs 1 \
  --output-dir "$SMOKE_DIR"
```

The smoke export is written under:

```text
$SMOKE_DIR/gpt55/trajectories_gpt55_r0-0.csv
```

Full 20 + 5 run over the five default models:

```bash
python Andres/prid_batch_pipeline/run_prid_pipeline.py
```

The full run writes final ADS inputs such as:

```text
Andres/ads_inputs/trajectories/trajectories_challenge_22_gpt55_prid.csv
Andres/ads_inputs/trajectories/trajectories_challenge_22_gpt52_prid.csv
Andres/ads_inputs/trajectories/trajectories_challenge_22_gpt5_prid.csv
Andres/ads_inputs/trajectories/trajectories_challenge_22_o3_prid.csv
Andres/ads_inputs/trajectories/trajectories_challenge_22_gpt41_prid.csv
```

Later 15-run extension:

```bash
python Andres/prid_batch_pipeline/run_prid_pipeline.py \
  --run-start 5 \
  --continuation-runs 15
```

The extension writes labelled files with `r5-19` in the name so it does not overwrite the first 5-run arm.

## Repair Missing Rows

If exports report incomplete trajectories because a model returned no parseable score, run:

```bash
python Andres/prid_batch_pipeline/repair_prid_outputs.py --tags o3 gpt41
```

The repair command scans existing outputs, submits only the missing/unparseable continuation branches, merges successful repair rows into the canonical output JSONL, and re-exports the trajectory CSV. Use `--dry-run` first to inspect how many repair requests would be submitted:

```bash
python Andres/prid_batch_pipeline/repair_prid_outputs.py --tags o3 gpt41 --dry-run
```

If a plain repair still leaves an incomplete trajectory because the model answers with prose instead of a score, rerun only the still-incomplete tag with the stricter score-only repair prompt. Use `--force` so the script replaces the earlier repair batch metadata/output files instead of reusing them:

```bash
python Andres/prid_batch_pipeline/repair_prid_outputs.py \
  --tags gpt41 \
  --force \
  --strict-score-repair
```

For a labelled range such as the later 15-run extension, pass the same range flags:

```bash
python Andres/prid_batch_pipeline/repair_prid_outputs.py \
  --run-start 5 \
  --continuation-runs 15
```

## Build S0 Inputs

Generate one Batch input file per model. This is necessary because a Batch input file can only target one model.

```bash
python Andres/prid_batch_pipeline/build_prid_batches.py s0
```

This writes:

```text
Andres/prid_batch_pipeline/batches/<tag>/inputs/<tag>_s0_in.jsonl
Andres/prid_batch_pipeline/batches/<tag>/run_config.json
```

To build a single model:

```bash
python Andres/prid_batch_pipeline/build_prid_batches.py s0 --tags gpt55
```

## Submit, Wait, Download

The optional helper uses the OpenAI Python SDK and reads `OPENAI_API_KEY` from the environment or `Andres/.env`.

```bash
python Andres/prid_batch_pipeline/openai_batch_io.py submit \
  --input Andres/prid_batch_pipeline/batches/gpt55/inputs/gpt55_s0_in.jsonl \
  --metadata-output Andres/prid_batch_pipeline/batches/gpt55/gpt55_s0_batch.json

python Andres/prid_batch_pipeline/openai_batch_io.py wait --batch-id <batch_id>

python Andres/prid_batch_pipeline/openai_batch_io.py download \
  --batch-id <batch_id> \
  --output Andres/prid_batch_pipeline/batches/gpt55/outputs/gpt55_s0_out.jsonl \
  --error-output Andres/prid_batch_pipeline/batches/gpt55/outputs/gpt55_s0_errors.jsonl
```

Repeat the same submit/wait/download pattern for `t1`, `t2`, and `t3` after each input file is built.

## Build Continuation Inputs

Build turn 1 from the downloaded S0 output. The script uses all available S0 scores for each artefact to choose the challenge direction, then chains only runs `0..4` by default.

```bash
python Andres/prid_batch_pipeline/build_prid_batches.py turn1 \
  --tag gpt55 \
  --s0-output Andres/prid_batch_pipeline/batches/gpt55/outputs/gpt55_s0_out.jsonl
```

This writes:

```text
Andres/prid_batch_pipeline/batches/gpt55/inputs/gpt55_t1_in.jsonl
Andres/prid_batch_pipeline/batches/gpt55/gpt55_direction_summary.csv
```

After downloading the turn-1 output:

```bash
python Andres/prid_batch_pipeline/build_prid_batches.py turn2 \
  --tag gpt55 \
  --previous-output Andres/prid_batch_pipeline/batches/gpt55/outputs/gpt55_t1_out.jsonl
```

After downloading the turn-2 output:

```bash
python Andres/prid_batch_pipeline/build_prid_batches.py turn3 \
  --tag gpt55 \
  --previous-output Andres/prid_batch_pipeline/batches/gpt55/outputs/gpt55_t2_out.jsonl
```

## Run Ranges

The default continuation arm uses runs `0..4`. To add the other 15 continuations later from the same S0 output, use `--run-start 5 --continuation-runs 15`:

```bash
python Andres/prid_batch_pipeline/build_prid_batches.py turn1 \
  --tag gpt55 \
  --s0-output Andres/prid_batch_pipeline/batches/gpt55/outputs/gpt55_s0_out.jsonl \
  --run-start 5 \
  --continuation-runs 15
```

For non-default ranges, filenames get an automatic run label:

```text
Andres/prid_batch_pipeline/batches/gpt55/inputs/gpt55_t1_r5-19_in.jsonl
Andres/prid_batch_pipeline/batches/gpt55/inputs/gpt55_t2_r5-19_in.jsonl
Andres/prid_batch_pipeline/batches/gpt55/inputs/gpt55_t3_r5-19_in.jsonl
Andres/ads_inputs/trajectories/trajectories_challenge_22_gpt55_prid_r5-19.csv
```

Pass `--output-label <label>` to override the inferred label. This is useful if a batch output is incomplete and you want the filenames to record a custom rerun label.

## Export ADS Trajectories

After all four batch outputs are downloaded:

```bash
python Andres/prid_batch_pipeline/build_prid_batches.py export \
  --tag gpt55 \
  --s0-output Andres/prid_batch_pipeline/batches/gpt55/outputs/gpt55_s0_out.jsonl \
  --turn1-output Andres/prid_batch_pipeline/batches/gpt55/outputs/gpt55_t1_out.jsonl \
  --turn2-output Andres/prid_batch_pipeline/batches/gpt55/outputs/gpt55_t2_out.jsonl \
  --turn3-output Andres/prid_batch_pipeline/batches/gpt55/outputs/gpt55_t3_out.jsonl
```

The default export path is:

```text
Andres/ads_inputs/trajectories/trajectories_challenge_22_gpt55_prid.csv
```

`Andres/ads_report_v2/scripts/compute_ads_v2.py` discovers `trajectories_challenge_22_*.csv`, so the PRID arm is picked up automatically once the CSV is present.

## Smoke Test

Before spending the full run, use a separate output directory and test one artefact, one S0 run, and one continuation run:

```bash
SMOKE_DIR="$(mktemp -d)"

python Andres/prid_batch_pipeline/build_prid_batches.py s0 \
  --tags gpt55 \
  --artefact-limit 1 \
  --s0-runs 1 \
  --output-dir "$SMOKE_DIR"

python Andres/prid_batch_pipeline/openai_batch_io.py submit \
  --input "$SMOKE_DIR/gpt55/inputs/gpt55_s0_in.jsonl" \
  --metadata-output "$SMOKE_DIR/gpt55/gpt55_s0_batch.json"
```

Then continue with `turn1`, `turn2`, `turn3`, and `export` in the same `$SMOKE_DIR`. This is worth doing because the whole design depends on batch-created response IDs being accepted as `previous_response_id` in the following batch.

For the one-run smoke continuation, `turn1 --continuation-runs 1` writes a labelled file:

```bash
python Andres/prid_batch_pipeline/build_prid_batches.py turn1 \
  --tag gpt55 \
  --artefact-limit 1 \
  --continuation-runs 1 \
  --s0-output "$SMOKE_DIR/gpt55/outputs/gpt55_s0_out.jsonl" \
  --output-dir "$SMOKE_DIR"

python Andres/prid_batch_pipeline/openai_batch_io.py submit \
  --input "$SMOKE_DIR/gpt55/inputs/gpt55_t1_r0-0_in.jsonl" \
  --metadata-output "$SMOKE_DIR/gpt55/gpt55_t1_batch.json"
```

## Notes

All S0 and continuation requests set `store: true`. The continuation batches send only the next user turn plus `previous_response_id`; they do not replay previous messages locally. The initial prompt and argument prompts are converted from the artefacts' native 1-10 scale to the existing ADS 1-100, number-only scale.

Stored Responses API objects have a retention window, so the staged batches should be completed soon after S0 rather than weeks later.
