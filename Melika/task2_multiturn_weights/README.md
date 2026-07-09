# Task 2: Multi-Turn Weight Aggregation

This local experiment tests how different ways of aggregating argument BT weights
change multi-turn ADS-style scores.

The important caveat from Marthe's updated `ads_v2_math.md` is that the official
weighted ADS is cleanest at turn 1. At turn 2 and turn 3, the score shift mixes
several arguments, so multi-turn weighting is exploratory/descriptive.

## Pipeline

Run scripts from the project root:

```sh
python3 task2_multiturn_weights/scripts/01_download_inputs.py
python3 task2_multiturn_weights/scripts/02_prepare_weights.py
python3 task2_multiturn_weights/scripts/03_compute_multiturn_ads.py
python3 task2_multiturn_weights/scripts/04_plot_results.py
```

## Inputs

The scripts use these files from Marthe's GitHub folder:

- `Marthe/bt_global/results/bt_scores.csv`
- `Marthe/figure_1/results/trajectories_challenge_22.csv`
- `Marthe/figure_1/results/trajectories_challenge_22_o4mini.csv`

If the download script cannot access the private GitHub files, download them
manually from GitHub and place them in:

```text
task2_multiturn_weights/data/bt_scores.csv
task2_multiturn_weights/data/trajectories_challenge_22.csv
task2_multiturn_weights/data/trajectories_challenge_22_o4mini.csv
```

## Outputs

- `results/prepared_bt_scores.csv`
- `results/multiturn_ads_by_method.csv`
- `results/per_artifact_scores.csv`
- `results/figures/ads_by_method.png`
- `results/figures/pval_pinval_plane.png`
- `results/figures/andres_style_turn23_ads_plane.png`
- `results/figures/ads_plane_aggregation_mosaic_turn23.png`
- `results/figures/ads_plane_unweighted_turn23.png`
- `results/figures/ads_plane_lead_turn23.png`
- `results/figures/ads_plane_mean_turn23.png`
- `results/figures/ads_plane_max_turn23.png`
- `results/figures/ads_plane_min_turn23.png`
- `results/figures/ads_plane_sum_turn23.png`
