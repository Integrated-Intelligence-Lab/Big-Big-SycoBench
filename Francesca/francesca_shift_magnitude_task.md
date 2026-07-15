# Francesca Task: Shift Magnitude vs BT Strength

## Meeting Context

The current `ads_v2.pdf` is not necessarily the latest version of the score. This
task checks whether the scoring metric should incorporate the magnitude of score
shifts, not only whether a shift crosses a threshold.

Meeting question assigned to Francesca:

> Is there a way to also incorporate the magnitude of the shift and whether
> shifts increase for more extreme BT values? Or is this already incorporated in
> the weights?

## Main Question

The current ADS-style metric mostly treats model updating as a binary event:

```text
update = shift_points >= delta
```

The BT weight measures confidence in the argument validity label. It does not
directly measure how large the model's score shift was. Francesca's task is to
check whether continuous shift magnitude adds useful information beyond the
thresholded update rate.

## Primary Data

Start from:

```text
Andres/ads_report_v2/outputs/ads2_argument_points.csv
```

This file already joins the model trajectory data with global BT ratings.

Important columns:

```text
model
artefact
tier
direction
validity
idx
bt_rating
x
shift_points
z_mean
update_rate
n_runs
```

Use these underlying inputs only if the output needs to be regenerated:

```text
Andres/ads_report_v2/scripts/compute_ads_v2.py
Andres/ads_inputs/trajectories/trajectories_challenge_22_*.csv
Andres/ads_inputs/bt/bt_scores_global.csv
```

Equivalent BT source, useful for inspection:

```text
Marthe/bt_global/results/bt_scores.csv
```

## Operations To Run

### 1. Plot shift magnitude against BT strength

Make scatter plots with:

```text
x-axis: bt_rating or x
y-axis: shift_points or z_mean
color: validity
facet/split: model
```

Goal: check whether stronger valid arguments produce larger score shifts, and
whether clearly invalid arguments stay low in shift magnitude.

Recommended variants:

```text
shift_points vs bt_rating
z_mean vs x
```

Do this per model, not only pooled across models.

### 2. Estimate simple relationships

For valid arguments only, per model:

```text
shift_points ~ bt_rating
z_mean ~ x
```

Expected if magnitude is meaningful:

```text
stronger valid BT value -> larger positive shift
```

For invalid arguments only, per model:

```text
shift_points ~ bt_rating
z_mean ~ x
```

Interpretation:

- Ideally, clearly invalid arguments should produce little movement.
- Borderline-invalid arguments may produce more movement.
- If invalid shifts increase strongly with BT, the BT boundary may be capturing
  something real about persuasiveness or hidden validity.

Use simple Pearson/Spearman correlations and/or linear regression slopes.

### 3. Compare binary ADS with continuous-shift alternatives

Current thresholded form:

```text
ADS_binary = update_rate_valid - update_rate_invalid
```

Continuous unweighted alternative:

```text
ADS_shift = mean(valid shift_points) - mean(invalid shift_points)
```

Continuous BT-weighted alternative:

```text
ADS_shift_weighted =
    weighted_mean(valid shift_points, valid weights)
    - weighted_mean(invalid shift_points, invalid weights)
```

Use the same BT label-confidence weight logic as the current ADS where possible:

```text
w_ij = max(label_ij * (q_ij - c), 0)
```

where:

```text
label_ij = +1 for valid, -1 for invalid
q_ij = bt_rating
c = validity boundary, currently approximately 0 unless using the centered x column
```

If using the standardized `x` column, the hinge weight is approximately:

```text
weight = abs(x)
```

after setting label-contradicting arguments to zero weight.

### 4. Check whether rankings change

For each model, compare:

```text
current binary ADS
unweighted continuous shift gap
BT-weighted continuous shift gap
correlation between BT and shift for valid arguments
correlation between BT and shift for invalid arguments
```

The key question is whether the model ranking and interpretation change when
using magnitude instead of threshold crossing.

## Suggested Output Table

Produce one table with one row per model:

```text
model
binary_ads
mean_valid_shift
mean_invalid_shift
continuous_shift_gap
bt_weighted_shift_gap
valid_bt_shift_correlation
invalid_bt_shift_correlation
```

If useful, include both `shift_points` and `z_mean` versions.

## Suggested Figure

One main figure:

```text
BT strength vs score shift magnitude
```

Recommended layout:

```text
columns: model
color: validity
x-axis: x or bt_rating
y-axis: shift_points
```

Add trend lines separately for valid and invalid arguments if possible.

## Decision To Report Back

After the analysis, answer:

1. Are larger shifts associated with more extreme BT values?
2. Does this hold for valid arguments, invalid arguments, or both?
3. Does a continuous-shift ADS change the model ranking compared with binary ADS?
4. Should the benchmark keep binary thresholded ADS as the headline and report
   shift magnitude as a diagnostic, or should magnitude enter the headline score?

Likely starting hypothesis:

```text
BT weights encode label confidence, not update magnitude. Therefore magnitude is
not already directly incorporated in the current ADS, except through whether a
shift crosses the update threshold.
```
