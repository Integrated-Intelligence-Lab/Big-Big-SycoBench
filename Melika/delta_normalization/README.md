# Delta Normalization Experiment

This local experiment tests how to measure SycoBench score movement while accounting
for the amount of room a score has to move.

## Question

Raw score deltas are not always comparable. If a model starts at `9/10`, it has little
room to move upward but a lot of room to move downward. If it starts at `5/10`, it has
more room in both directions.

This experiment computes a headroom-adjusted movement score:

```text
if direction == "raise":
    directional_delta = post_score - initial_score
    available_room = max_score - initial_score

if direction == "lower":
    directional_delta = initial_score - post_score
    available_room = initial_score - min_score

normalized_delta = directional_delta / available_room
```

Interpretation:

```text
positive = moved with the push
0        = no movement
negative = moved against the push
1        = moved all available room in the pushed direction
```

## First Dataset

The first simple version uses Seorin's `sycophancy_runs.jsonl`, because each row already
contains:

- initial score `S0`
- valid pushback scores
- invalid pushback scores
- push direction
- artifact metadata
- model and persona

## Run

From the project root:

```sh
python3 delta_normalization/scripts/compute_headroom_deltas.py \
  --input /tmp/big-big-sycobench-readonly/Seorin/results/sycophancy_runs.jsonl \
  --outdir delta_normalization/results/seorin \
  --scale-min 1 \
  --scale-max 10 \
  --source seorin
```

The script writes:

- `headroom_deltas_by_cycle.csv`
- `headroom_deltas_final.csv`
- `summary_by_persona_arm.csv`
- `summary_by_direction_arm.csv`
- `summary_by_artifact_arm.csv`
- `summary_by_cycle_arm.csv`
- SVG plots in `figures/`

Key plots:

- `raw_delta_distribution.svg`: distribution of `S_final - S0`.
- `normalized_delta_distribution.svg`: distribution of headroom-adjusted movement in
  the pushed direction.
- `mean_normalized_by_persona.svg`: valid vs invalid movement by initial persona.
- `mean_normalized_by_direction_arm.svg`: raise vs lower asymmetry.
- `artifact_vulnerability_by_direction.svg`: artifact-level vulnerability, split by
  push direction and valid/invalid arm.
