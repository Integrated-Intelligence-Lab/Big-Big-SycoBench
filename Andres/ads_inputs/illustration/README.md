# Synthetic Shape-Gallery Data

This folder contains fake y-values for illustration only. The x-axis values are real global BT ratings copied from Marthe's challenge-direction argument set: 22 artefacts x 1 direction x 2 validities x 3 argument indices = 132 arguments.

Files:

- `shape_gallery_synthetic_points.csv`: one row per synthetic shape x real argument. Use `bt_rating` as x and `synthetic_shift` as the fake normalized shift y.
- `shape_gallery_synthetic_curves.csv`: grid values for the noiseless shape and the pinned sigmoid overlay.
- `shape_gallery_synthetic_parameters.csv`: shape labels, noise scale, and seed.
- `generate_shape_gallery_synthetic.py`: deterministic generator.
- `shape_gallery_data_only.png` / `shape_gallery_data_only.pdf`: 12-panel data-only gallery from the synthetic point CSV. These contain no true-shape, fitted, or overlay curves.
- `plot_shape_gallery_data_only.py`: plotting script for the data-only gallery.
- `shape_gallery_ads.png` / `shape_gallery_ads.pdf`: 12-panel gallery with per-panel update rates and ADS, the update threshold, and the legacy diagnostic overlays (fitted valid-uptake curve, invalid-compliance level).
- `shape_gallery_ads_scores.csv`: ADS (update rates) plus legacy ACSL component scores for each synthetic shape.
- `plot_shape_gallery_ads.py`: plotting and scoring script for the fitted illustration gallery.

These rows should not be used as model evidence. They are only for making schematic figures that explain possible response shapes.
