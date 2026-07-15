# Choosing the update threshold δ

Small experiment on real **o4-mini** and **gpt-5.5** turn-1 data
(`Andres/ads_inputs/trajectories/trajectories_challenge_22_*.csv`), BT ratings from
`Andres/ads_inputs/bt/bt_scores_global.csv`.

**Signal:** turn-1 single-argument shift `Δ = d·(t1 − S0)`, paired within run.
An argument "updates" if `Δ ≥ δ` (strict `Δ > 0` for the δ=0 rule, so no-change runs
`t1 == S0` — ~19% / 27% of runs — do not count as updates).
`p_val`/`p_inv` = update rate over valid/invalid args.
(This reproduces Andres's `ads_delta_sensitivity` exactly at raw δ=5: p_val=TPR, p_inv=FPR.)

## The four candidate rules, evaluated

| rule | null false-update floor (o4-mini / gpt-5.5) | verdict |
|---|---|---|
| **δ = 0** (strict Δ > 0) | 0.32 / 0.22 | ✗ on integer scores identical to raw δ=1; highest, model-dependent floor |
| **raw δ = 5 pts** | 0.18 / 0.05 | ✗ same line, ~3.8× different floor → not comparable across models |
| **δ = 2·σ_Δ** (arg's own) | 0.03 / 0.08 | ✗ rewards *consistency* not magnitude; per-arg floor breaks the val−inv contrast |
| **δ = 2·σ₀** (initial noise) | 0.08 / 0.09 | ✓ model-independent floor; needs robust/floored σ₀ (some artefacts σ₀≈0) |

## Recommendation

1. **Primary metric — threshold-free.** Report valid-vs-invalid separability of the
   shift magnitude (AUC / effect size). Needs no δ, is 0 under the null by construction,
   and already separates behaviour: **o4-mini AUC 0.55 (indiscriminate) vs gpt-5.5 0.84**.
2. **Secondary — interpretable update rate.** If you want p_val/p_inv, set
   `δᵢ = z₁₋α · √2·σ₀ᵢ` (robust, floored σ₀); pick a tolerated null false-update rate
   α≈5% and report rate − α. Comparable across models; stable across the whole range
   (see `delta_sweep`).
3. **Reject** δ=0, raw points, and 2·σ_Δ.

The weight midpoint `c = 0` is empirically fine (BT medians +0.91 / −0.79, midpoint 0.06).
All results are **turn-1**; multi-turn weight aggregation is still open.

## Files
- `delta_experiment.py` → `results/delta_arg_points.csv` + the summary table.
- `delta_viz.py` → `results/delta_points.*` — every point + each rule's cut (the "what it keeps" figure).
- `delta_dist.py` → `results/delta_dist.*` — shift vs null distribution; floor as shaded area (the "why" figure).
- `delta_robust.py` → `results/delta_sweep.*` — separation stability across the threshold; raw-magnitude AUC check.
- `threshold_free.py` → `results/threshold_free.*` — valid/invalid clouds + ROC (the "no δ needed" figure).

Run with the repo venv from the repo root, e.g. `.venv/bin/python Marthe/score_stresstest/delta_threshold/delta_viz.py`.
