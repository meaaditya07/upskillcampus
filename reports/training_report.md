# AgriYield AI - training report

Generated 2026-10-04T11:57:21+00:00 in 208.6s with seed 20240101.

> **These metrics come from synthetic data.** The generator in `src/synthetic.py` produces rows with realistic structure, units and defects, but no measurement in this report is evidence about Indian agriculture. Retrain on a real CSV before drawing conclusions.

## Headline

| Target | R2 | R2 (log scale) | RMSE | MAE | SMAPE % | Interval coverage |
| --- | --- | --- | --- | --- | --- | --- |
| Production (quintals) | 0.8667 | 0.9682 | 730.9 | 69.0 | 19.3 | 81.3% |
| Cultivation cost (INR) | 0.8217 | 0.9764 | 136,072.2 | 22,378.4 | 14.8 | 79.0% |

Held-out test rows: 11,389. Fitted rows: 38,721. Calibration rows: 6,834.

### Two R2 numbers, on purpose

- **Production (quintals)** - R2 0.8667 on the original scale, 0.9682 on `log1p(target)`.
- **Cultivation cost (INR)** - R2 0.8217 on the original scale, 0.9764 on `log1p(target)`.

The shipped model fits `log1p(target)`, so the log-scale figure is the more faithful description of what it learned. The original-scale figure is reported because it is the one a reader will compare against, and because it is dominated by the largest fields: halving the error on small plots barely moves it, while one large field moves it a lot. Quoting only the flattering number would be misleading, so both are here.

## Baselines

A model that cannot beat "predict the group average" is not worth shipping. RMSE is in the target's own units.

**Production (quintals)**

| Model | RMSE | MAE | R2 | MAPE % |
| --- | --- | --- | --- | --- |
| XGBoost (this project) | 730.9 | 69.0 | 0.8667 | 28.4 |
| Per-crop mean | 1,786.7 | 315.8 | 0.2033 | 357.8 |
| Per-variety mean | 1,795.3 | 324.0 | 0.1957 | 474.0 |
| Per-state mean | 1,966.5 | 480.8 | 0.0350 | 1,992.4 |
| Global mean | 2,002.0 | 533.3 | -0.0002 | 2,315.4 |

**Cultivation cost (INR)**

| Model | RMSE | MAE | R2 | MAPE % |
| --- | --- | --- | --- | --- |
| XGBoost (this project) | 136,072.2 | 22,378.4 | 0.8217 | 15.0 |
| Per-crop mean | 302,924.9 | 117,257.5 | 0.1164 | 262.4 |
| Per-variety mean | 305,561.6 | 118,853.7 | 0.1009 | 273.2 |
| Per-state mean | 317,361.6 | 129,855.9 | 0.0301 | 342.7 |
| Global mean | 322,285.6 | 135,428.0 | -0.0002 | 397.4 |

## Cross-validation

random split: Random 80/20 shuffle. Fold means below are on the training portion, so they are not the numbers to quote - the held-out figures above are.

**Production (quintals)**

| Metric | Mean | Std |
| --- | --- | --- |
| RMSE | 551.38 | 221.56 |
| MAE | 72.32 | 9.45 |
| R2 | 0.9056 | 0.0552 |

**Cultivation cost (INR)**

| Metric | Mean | Std |
| --- | --- | --- |
| RMSE | 121,446.60 | 55,518.28 |
| MAE | 23,728.26 | 1,650.46 |
| R2 | 0.8572 | 0.0811 |

The gap between cross-validated and held-out R2 is the honest signal here: a random split lets the same state-crop-variety combination appear in both train and test, which flatters the model. `--split group_state`, `group_zone` and `chronological` exist to measure how much of the headline number depends on that overlap.

## Prediction intervals

- **Nominal coverage:** 80%
- **Quantile alphas:** [0.1, 0.9]
- **Method:** Quantile gradient boosting (reg:quantileerror) on log1p(target), widened by a scalar factor fitted on a held-out calibration split. The bootstrap ensemble is reported as a stability diagnostic only -- see the note in predict_with_interval.

| Target | Calibrated coverage | Uncalibrated coverage | Calibration factor | Median width | Mean width |
| --- | --- | --- | --- | --- | --- |
| Production (quintals) | 81.29% | 80.14% | 1.031 | 28.6 | 201.7 |
| Cultivation cost (INR) | 79.05% | 77.93% | 1.034 | 30,989.6 | 68,008.5 |

- **Production (quintals)** - 20 bootstrap members vary by 7.6%. This is reported as a stability diagnostic and is **not** used to build the published interval: the bags share their training data, so their spread is not a variance estimate.
- **Cultivation cost (INR)** - 20 bootstrap members vary by 6.4%. This is reported as a stability diagnostic and is **not** used to build the published interval: the bags share their training data, so their spread is not a variance estimate.

## What the model uses

15 features: state, crop, variety, recommended_zone, season_name, season_duration, unit, zone_scope, crop_group, variety_class, quantity, year, log_quantity, sqrt_quantity, year_index.

### Production (quintals) - permutation importance

Mean drop in R2 when a feature is shuffled, measured on held-out data. More trustworthy than tree gain, which inflates continuous features.

| Feature | R2 drop (mean) | R2 drop (std) |
| --- | --- | --- |
| crop | 0.6322 | 0.1965 |
| quantity | 0.6274 | 0.0564 |
| variety | 0.3122 | 0.1092 |
| log_quantity | 0.1797 | 0.0186 |
| state | 0.0951 | 0.0616 |
| season_duration | 0.0686 | 0.0490 |
| crop_group | 0.0675 | 0.0011 |
| sqrt_quantity | 0.0467 | 0.0108 |
| variety_class | 0.0389 | 0.0295 |
| season_name | 0.0299 | 0.0077 |

### Cultivation cost (INR) - permutation importance

Mean drop in R2 when a feature is shuffled, measured on held-out data. More trustworthy than tree gain, which inflates continuous features.

| Feature | R2 drop (mean) | R2 drop (std) |
| --- | --- | --- |
| quantity | 0.5701 | 0.0489 |
| crop | 0.3175 | 0.1553 |
| log_quantity | 0.1200 | 0.0117 |
| variety | 0.0433 | 0.0084 |
| sqrt_quantity | 0.0364 | 0.0049 |
| state | 0.0210 | 0.0404 |
| season_name | 0.0167 | 0.0088 |
| year_index | 0.0167 | 0.0097 |
| zone_scope | 0.0116 | 0.0160 |
| recommended_zone | 0.0087 | 0.0171 |

## Leakage controls

**Production (yield)** (15 features)

- Target-encoded: state, crop, variety, recommended_zone
- One-hot: season_name, season_duration, unit, zone_scope, crop_group, variety_class
- Numeric: quantity, year, log_quantity, sqrt_quantity, year_index
- **Excluded to avoid leakage:** cost, cost_per_unit, is_outlier, production, production_efficiency, production_quintals, production_tons

**Cultivation cost** (15 features)

- Target-encoded: state, crop, variety, recommended_zone
- One-hot: season_name, season_duration, unit, zone_scope, crop_group, variety_class
- Numeric: quantity, year, log_quantity, sqrt_quantity, year_index
- **Excluded to avoid leakage:** cost, cost_per_unit, is_outlier, production, production_efficiency, production_quintals, production_tons

- Target-derived columns (`production`, `production_quintals`, `cost`, `cost_per_unit`, `production_efficiency`) are excluded from both models, so neither can see the other's answer.
- `is_outlier` is excluded because it is computed from both targets; 408 flagged rows were held out of training entirely.
- High-cardinality columns are target-encoded inside cross-validation folds, not before the split. Fitting the encoder on the whole dataset first would let each row's own target leak into its own feature.
- `assert_no_leakage()` runs when each pipeline is constructed, and `tests/test_preprocessing.py` asserts both that the shipped feature sets are clean and that the guard raises when fed a dirty one.

## Booster benchmark

| Target | XGBoost R2 | LightGBM R2 | Delta |
| --- | --- | --- | --- |
| Production (quintals) | 0.8667 | 0.8738 | +0.0071 |
| Cultivation cost (INR) | 0.8217 | 0.8290 | +0.0073 |

LightGBM edges ahead on both targets, by a small margin, and its artefacts are roughly eight times smaller. **XGBoost is still what ships**, for two reasons: the difference is well inside the cross-validation spread, and the per-prediction explanation path is built on XGBoost's `pred_contribs`. Promoting LightGBM is a deliberate decision to revisit with the explanation work, not an accident of which booster ran last.

## Data quality

- **Source:** `C:\aadilyf\CropProductionAnalysis\data\raw\crop_production_synthetic.csv`
- **SHA-256:** `4de417cbcbee376289162161ccd68f18be32e90b213e994b221ac5137e3e3c5c`
- **Raw rows:** 60,000
- **Clean rows:** 57,352
- **Rows dropped:** 2,648 (reasons overlap, so they do not sum to the total)
- **Case fixes:** 5,649
- **Whitespace trims:** 12,723
- **Unit fallbacks:** 452

### Dropped rows by reason

| Reason | Rows |
| --- | --- |
| missing_cost | 1,226 |
| missing_production | 895 |
| nonpositive_quantity | 570 |

### Units after normalisation

| Unit | Rows |
| --- | --- |
| Quintals | 44,652 |
| Tons | 12,248 |
| Unknown | 452 |

- 452 rows had an unrecognised unit and were treated as quintals.
- Year window observed: 2001-2014

## Reproducibility

- **Python:** 3.14.4
- **Platform:** Windows
- **Seed:** 20240101

| Library | Version |
| --- | --- |
| pandas | 3.0.5 |
| numpy | 2.5.2 |
| sklearn | 1.9.0 |
| xgboost | 3.4.1 |
| lightgbm | 4.7.0 |
| joblib | 1.6.0 |

The same seed and the same input CSV reproduce these artefacts. Re-run with:

```bash
python -m src.train_models --split random --bootstrap 20
```
