# AgriYield AI

Crop production and cultivation-cost forecasting for India, with profitability
analysis, crop recommendations and an honest diagnostics page.

A Streamlit front end over a Streamlit-free service layer, so the same code
backs the UI today and a FastAPI endpoint later without moving any logic.

---

## Quick start

```bash
python -m pip install -r requirements.txt
python scripts/bootstrap.py
python -m streamlit run app.py
```

`bootstrap.py` generates the dataset (synthetic, if you have no CSV), downloads
and dissolves the India basemap, trains both models and verifies the artefacts.
It is safe to re-run; each step is skipped if its output already exists.

```bash
python scripts/bootstrap.py --force        # regenerate everything and retrain
python scripts/bootstrap.py --fast         # quick models while iterating (5 bags, no CV)
python scripts/bootstrap.py --data-only    # dataset and map, no training
```

> Use `python -m streamlit`, not the bare `streamlit` command. User script
> directories are frequently not on `PATH` on Windows and macOS.

## Using your own data

Drop a CSV anywhere in `data/raw/`. It wins over the synthetic file
automatically, and the sidebar relabels itself from **synthetic demo data** to
the file path. Required columns, with the header spellings that are recognised:

| Meaning | Accepted headers |
| --- | --- |
| Crop | `Crop`, `crop` |
| Variety | `Variety`, `variety` |
| State | `State`, `state` |
| Area (hectares) | `Quantity`, `Area`, `area_ha` |
| Production (in `Unit`) | `Production`, `production` |
| Unit | `Unit`, `unit` |
| Cultivation cost (total INR) | `Cost`, `cost` |
| Season | `Season`, `season` |
| Zone | `Recommended Zone`, `Zone` |
| Year | `Year`, `year` |

Then retrain: `python -m src.train_models --data data/raw/your_file.csv`.

Column matching is case- and separator-insensitive, so `Recommended Zone`,
`recommended_zone` and `RECOMMENDEDZONE` all resolve. Anything unmapped is
reported on the Diagnostics page rather than silently ignored.

## Pages

| Page | What it answers |
| --- | --- |
| **Overview** | Where does India grow what, and at what cost? |
| **Yield & cost forecast** | What will *my* field produce and cost, and what is that worth? |
| **Recommendations** | Where and what should I plant, and how sure are we? |
| **Diagnostics** | How good is the model, and what can it not do? |

## How it works

```
data/raw/*.csv
     │  src/data_loader.py      resolve headers, coerce, clean, canonicalise
     │  src/schema.py           states, crops, units, zones, seasons
     ▼
  cleaned frame  ── 57,352 rows
     │  src/preprocessing.py    features, outlier flags, leakage firewall
     │                         high-cardinality target encoding inside CV folds
     ▼
  src/train_models.py
     │  src/models.py           XGBoost on log1p(target) + calibrated quantile pair
     │  src/evaluate.py         baselines, CV, permutation importance, residuals
     ▼
  models/*.joblib + models/metrics.json
     │  src/services/registry.py    stamp-aware artifact cache
     ▼
  src/services/{predictor,profitability,recommend,insights}.py   ← no Streamlit imports
     ▼
  components/ + page_*.py + app.py
```

### Data semantics

These are the decisions that quietly corrupt everything downstream if they are
wrong, so they are enforced in one place and asserted in tests:

- **Area** is `quantity`, in cultivated hectares. Cultivation cost is a **total**
  for that area, so `cost_per_unit` is per hectare.
- **Production** is reported in the source `unit` and converted **once**, at
  load, to `production_quintals`.
- **An Indian quintal is 50 kg**, so one tonne is exactly 20 quintals. `Tons`
  is a unit of mass in the source and is treated as 20 quintals. Kilograms,
  170 kg bales and 100 kg bundles are handled too.
- **Year** may be `2001`, `2001-02`, `2001-2002` or `Kharif 2001`.
- **Zones** are stored as `"<Scope> - <Name>"` (`Mandal - Ludhiana`). Hyphens,
  en dashes, em dashes, colons, slashes and pipes all parse.
- **A missing or unreadable unit becomes `Unknown`** and is counted in
  `report.unit_fallbacks`, not silently coerced.

### Leakage prevention

Production, cost and every ratio derived from them are excluded from both
feature sets, as is `is_outlier` — it is computed from the targets. The
high-cardinality columns (`state`, `crop`, `variety`, `recommended_zone`) are
target-encoded **inside cross-validation folds**; fitting the encoder on the
whole dataset first would leak each row's own target into its own feature.

`assert_no_leakage()` runs at pipeline construction, and
`tests/test_preprocessing.py` asserts both directions: that the real feature
sets are clean, and that the guard actually raises when fed a dirty one.

### Prediction intervals

The published 80% range is **not** the bootstrap ensemble. It is two
`reg:quantileerror` XGBoost models at the 10th and 90th percentiles of
`log1p(target)`, widened by a scalar factor fitted on a calibration split that
neither model saw. The 20 bootstrap bags are reported as a stability diagnostic
(~7% spread) and are not used for the interval, because the bags are too
correlated for their spread to be a variance estimate.

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
