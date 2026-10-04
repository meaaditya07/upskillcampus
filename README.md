<div align="center">

# 🌾 AgriYield AI

**Smart Crop Yield Forecasting, Cost Optimization & Agricultural Insights for India**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-ML-green.svg)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Machine Learning platform predicting crop production, forecasting cultivation costs, and delivering data-driven crop recommendations across Indian states and agricultural zones.

[Key Features](#-key-features) • [Quick Start](#-quick-start) • [Architecture](#%EF%B8%8F-architecture) • [Custom Datasets](#-using-your-own-data) • [Testing](#-testing--quality)

---

</div>

## 🌟 Key Features

* **📈 Precision Yield & Cost Forecasting:** Powered by XGBoost with log-transformed target optimization and calibrated 80% quantile prediction intervals.
* **💡 Smart Crop & Zone Recommendations:** Ranks optimal crops using a multi-factor scoring algorithm (50% Efficiency, 30% Cost, 20% Reliability).
* **🔍 Honest Diagnostics & Model Audits:** Real-time visibility into feature importance, residuals, calibration error, and leakage prevention audits.
* **🗺️ Interactive Geographic Visualizations:** Interactive maps displaying nationwide cultivation metrics and regional insights.
* **⚙️ Clean Decoupled Architecture:** Streamlit UI completely separated from core service logic — ready to be exposed as a FastAPI backend effortlessly.

---

## 🚀 Quick Start

### 1. Prerequisites & Installation

Clone the repository and install dependencies:

```bash
git clone [https://github.com/meaaditya07/upskillcampus.git](https://github.com/meaaditya07/upskillcampus.git)
cd upskillcampus
python -m pip install -r requirements.txt

### Recommendations

Options are reduced to recency-weighted **medians** of efficiency and cost, plus
a robust coefficient of variation of their own yields, min-max normalised
across the candidates on screen and combined 50% efficiency / 30% cost / 20%
reliability. Groups with fewer than 30 observations or fewer than 3 distinct
years lose 0.5 of score, so a single lucky season cannot top the list.

Normalisation is relative: the score ranks the visible candidates against each
other and is not an absolute figure.

### Prices

Profit uses an **indicative** price catalogue, not official MSP data, and older
vintages are scaled down because nominal prices rose over the window. Every
profit figure in the UI is accompanied by this caveat. Swap
`INDICATIVE_PRICE_PER_QUINTAL` in `config/settings.py` for real figures before
drawing any conclusion from profit.

## Tests

```bash
python -m pytest tests/ -q
```

296 tests, about a minute, no network access required.

| File | Covers |
| --- | --- |
| `test_schema.py` | canonicalisation, unit conversion, zone and season parsing |
| `test_data_loader.py` | header resolution, coercion, cleaning, provenance |
| `test_preprocessing.py` | features, the leakage firewall, outlier flagging |
| `test_services.py` | forecasting, profitability, ranking, insights |
| `test_geo.py` | state keys, dissolving, and dataset-to-map coverage |
| `test_end_to_end.py` | CSV to forecast, and artefact/data drift guards |
| `test_ui.py` | every page renders under a headless Streamlit runtime |

Two of these are drift guards worth knowing about: if `metrics.json` records a
different clean row count than the pipeline currently produces, or if held-out
interval coverage falls outside a sane band, the suite fails. That is the
signal that the shipped artefacts predate the current code.

`test_ui.py` calls each page's `render()` for real rather than importing it,
and separately boots `app.py` itself. That distinction earned its keep: an
earlier version registered four `st.Page` callables that were all named
`render`, so Streamlit inferred the same URL path for each and refused to start
— while every per-page test passed, because each bypassed the navigation.

Linting is pinned in `ruff.toml` so the rule set does not depend on config
inherited from outside the repository:

```bash
python -m ruff check .
```

## Layout

```
app.py                  entry point and navigation
page_overview.py        national picture
page_predictor.py       per-field forecast and economics
page_recommend.py       crop and zone rankings
page_diagnostics.py     accuracy, calibration, leakage audit, data quality
components/             theme, charts, KPI tiles, sidebar
src/                    library code (no Streamlit imports)
  schema.py             canonical states, crops, units, zones, seasons
  data_loader.py        CSV to cleaned canonical frame
  synthetic.py          deterministic demo data generator
  preprocessing.py      features, splits, target encoding, leakage guard
  models.py             estimators, intervals, contributions
  evaluate.py           metrics, baselines, CV, permutation importance
  train_models.py       training CLI
  geo.py                GeoJSON fetch, dissolve, coverage
  services/             registry, predictor, profitability, recommend, insights
config/settings.py      paths, units, palette, prices, weights, seed
ruff.toml               pinned lint rule set
data/raw/               input CSVs
data/geo/               cached basemaps
data/processed/         prediction cache
models/                 trained artefacts and metrics.json
reports/                training_report.md
scripts/bootstrap.py    first-run setup
scripts/write_report.py regenerates reports/training_report.md
tests/                  pytest suite
```

The four `page_*.py` modules only *export* a `render()` callable; they run
nothing on import. `app.py` declares them with `st.navigation` and hands over
the callables. Do not link to a `page_*.py` path with `st.page_link` — that
makes Streamlit execute the file as its own entry point, which renders nothing.
Each `st.Page` needs an explicit `url_path`, since the URL is otherwise inferred
from the callable name and all four are called `render`.

## Reproducibility

`SEED = 20240101` in `config/settings.py` fixes every random draw. The same CSV
and seed produce the same artefacts, and the dataset SHA-256 is recorded in
`metrics.json` so a run can be tied back to its input.

`reports/training_report.md` is generated from `metrics.json`, never written by
hand, so it cannot drift away from the artefacts it describes:

```bash
python scripts/write_report.py
```

## Known limitations

- **The shipped metrics are from synthetic data.** Shapes and relative
  magnitudes are realistic; absolute values are not evidence about Indian
  agriculture. The sidebar says so on every page.
- A **random** split is optimistic for panel data, because the same
  state–crop–variety combination can appear in both train and test.
  `--split group_state` and `--split chronological` exist to measure that, and
  the Diagnostics page reports the random-split numbers as the shipped ones.
- Profit figures depend on an **indicative** price catalogue.
- Zone recommendations are only as good as the zone labels in the source data.
