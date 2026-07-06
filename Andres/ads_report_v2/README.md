# ADS: BT-weighted one-turn Argument Discernment Score

A well-calibrated evaluator updates when the argument is valid and does not update when the argument is invalid. ADS measures that gap, weighting each argument by how clearly it earns its validity label:

- **Headline = BT-weighted one-turn ADS.** Arguments are weighted by `|x|`, their standardized distance from the model's valid/invalid BT boundary (hinged: label-contradicting arguments get weight 0). The weights read as *label confidence*, not error severity — the report states the full decomposition honestly (the weighted gain is entirely invalid-side: compliance concentrates on borderline-invalid arguments, which carry the least weight) and acknowledges that the BT judge is load-bearing in the headline.
- **One-turn (t1) is the primary horizon** — the only one where every observation is attributable to exactly one argument, so the BT join is exact.
- **Multi-turn is secondary and unweighted by construction** (cyclic orderings make BT weighting degenerate beyond t1): cumulative z-shift trajectory curves plus per-turn unweighted ADS. Figure 3 also draws the label-confidence-weighted curves (weight = mean `|x|` of arguments seen so far), which converge to the unweighted curves at t3 exactly, by construction.
- **The unweighted (judge-free) score is reported adjacent everywhere**, alongside two surgical label-uncertainty variants: drop-S01 (the one crossover pool) and boundary-trimmed (`|x| < 0.25`).

Inputs are the shared trajectories and BT ratings in `../ads_inputs/`.

## Layout

- `scripts/compute_ads_v2.py` — metric + figure data. Reads `../ads_inputs/trajectories/trajectories_challenge_22_*.csv` and `../ads_inputs/bt/bt_scores_global.csv`. Writes `outputs/ads2_summary.csv` (all variants + cluster-bootstrap CIs), `ads2_argument_points.csv` (per-argument t1: BT, x, shift, z, update rate), `ads2_turn_curves.csv` (unweighted + label-confidence-weighted variants), `ads2_dose_response.csv`, `ads2_delta_sensitivity.csv`, `ads2_zero_weight.csv` (update rates on label-contradicting vs other arguments), `ads2_order_effects.csv` (t3 ordering spreads + per-position increments), `ads2_run_variance.csv` (run-subsampling, artefact/run variance decomposition, direction stability), `ads2_config.json`.
- `scripts/plot_quadrants.py` — Figure 1: the update-rate plane (quadrant taxonomy) with both models' weighted + unweighted t1 operating points.
- `scripts/plot_bt_validation.py` — Figure 2: per-argument mean t1 z-shift vs global BT rating, binned means, weighted headline boxes.
- `scripts/plot_turn_trajectories.py` — Figure 3: cumulative z-drift by arguments seen, valid vs invalid, solid unweighted with bootstrap bands plus dashed label-confidence-weighted curves.
- `scripts/plot_drift_vs_threshold.py` — Appendix Figure A1: continuous cumulative drift and thresholded update rates side by side.
- `scripts/plot_run_variance.py` — Figure 4: SD by design level (artefact / argument / run), run-subsampled headline vs the artefact-sampling CI, direction-flip risks at R=5.
- `ads_v2.tex` — the report (metric definition, design paragraphs incl. the weighting caveat, BT-scale validation, results, robustness).

## Reproduce

```bash
python3 scripts/compute_ads_v2.py --delta 5 --bootstrap 1000 --seed 0
python3 scripts/plot_quadrants.py
python3 scripts/plot_bt_validation.py
python3 scripts/plot_turn_trajectories.py
python3 scripts/plot_drift_vs_threshold.py
python3 scripts/plot_run_variance.py
latexmk -pdf ads_v2.tex
```

Full-model diagnostic versions:

```bash
python3 scripts/plot_quadrants.py --models all --output-path outputs/ads2_quadrants_full
python3 scripts/plot_drift_vs_threshold.py --models all --output-path outputs/ads2_drift_vs_threshold_full
```

`compute_ads_v2.py` is stdlib-only; the plots need matplotlib. All outputs are deterministic given `--seed`.

## Headline result (t1, delta = 5)

| Model   | ADS_w (headline) | unweighted t1 | unweighted t2 | unweighted t3 |
|---------|------------------|---------------|---------------|---------------|
| gpt-5.5 | **68.8** [55.0, 81.3] | 64.2 [50.6, 76.8] | 71.7 | 72.3 |
| o4-mini | **37.0** [21.4, 51.7] | 32.9 [17.5, 47.9] | 41.5 | 45.8 |

Both models take up valid arguments at the ceiling (p_val ~0.90-0.92 weighted); the entire separation is invalid-side compliance (p_inv,w 0.23 vs 0.53). Every robustness variant (two-sided, drop-S01, boundary-trimmed, delta in {1, 2, 10}) preserves the ordering and the rough factor of two; trimming the contested boundary band raises both models (72.1 / 40.7), and at delta = 1 ("any movement") the gap narrows because gpt-5.5 makes small 1-4-point concessions that delta = 5 filters out. Multi-turn: gpt-5.5's invalid drift plateaus after the first argument (2.4 -> 2.6 -> 2.6 z) while o4-mini's keeps accumulating (5.1 -> 7.4 -> 8.0).

The run count barely binds: the CIs are artefact-sampling widths (run noise is ~1% of the between-artefact variance), and 5 runs per arm reproduce the headline to within ±2 points with a ~2% wider CI (`ads2_run_variance.csv`). Replications can cut continuation runs but should keep ~20 cheap baseline scorings: three o4-mini artefacts have mean initial scores within 4 points of 50, and a 5-run direction choice flips them in 15-32% of draws.
