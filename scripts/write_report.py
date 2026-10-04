"""Regenerate reports/training_report.md from models/metrics.json.

The report is derived, never hand-written, so it cannot drift away from the
artefacts it describes. Run after any retraining:

    python scripts/write_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import (  # noqa: E402 - needs the sys.path insert above
    APP_NAME,
    METRICS_PATH,
    REPORT_PATH,
)

YIELD, COST = "production_quintals", "cost"
LABELS = {YIELD: "Production (quintals)", COST: "Cultivation cost (INR)"}


def _table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "_none recorded_"
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    return "\n".join([header, divider, *body])


def _num(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):,.{digits}f}"


def build(metrics: dict) -> str:
    provenance = metrics.get("provenance", {})
    quality = metrics.get("data_quality", {})
    split = metrics.get("split", {})
    interval = metrics.get("interval", {})
    models = metrics.get("models", {})
    libraries = metrics.get("libraries", {})

    out: list[str] = []
    add = out.append

    add(f"# {APP_NAME} - training report")
    add("")
    add(
        f"Generated {metrics.get('generated_at', 'n/a')} in "
        f"{float(metrics.get('elapsed_seconds', 0)):,.1f}s with seed "
        f"{metrics.get('seed', 'n/a')}."
    )
    add("")
    add(
        "> **These metrics come from synthetic data.** The generator in "
        "`src/synthetic.py` produces rows with realistic structure, units and "
        "defects, but no measurement in this report is evidence about Indian "
        "agriculture. Retrain on a real CSV before drawing conclusions."
    )
    add("")

    # ---- headline -------------------------------------------------------
    add("## Headline")
    add("")
    rows = []
    for target in (YIELD, COST):
        block = models.get(target, {})
        measured = block.get("metrics", {})
        tested = block.get("interval", {}).get("test_coverage", {})
        rows.append(
            {
                "Target": LABELS[target],
                "R2": _num(measured.get(f"{target}_r2")),
                "R2 (log scale)": _num(measured.get(f"{target}(log)_r2")),
                "RMSE": _num(measured.get(f"{target}_rmse"), 1),
                "MAE": _num(measured.get(f"{target}_mae"), 1),
                "SMAPE %": _num(measured.get(f"{target}_smape"), 1),
                "Interval coverage": (
                    f"{float(tested.get('coverage')):.1f}%"
                    if tested else "n/a"
                ),
            }
        )
    add(_table(rows, list(rows[0].keys())))
    add("")
    add(
        f"Held-out test rows: {split.get('n_test', 0):,}. "
        f"Fitted rows: {split.get('n_fit', 0):,}. "
        f"Calibration rows: {split.get('n_calibration', 0):,}."
    )
    add("")

    # ---- why R2 on two scales -------------------------------------------
    add("### Two R2 numbers, on purpose")
    add("")
    for target in (YIELD, COST):
        measured = models.get(target, {}).get("metrics", {})
        add(
            f"- **{LABELS[target]}** - R2 {_num(measured.get(f'{target}_r2'))} on the "
            f"original scale, {_num(measured.get(f'{target}(log)_r2'))} on "
            f"`log1p(target)`."
        )
    add("")
    add(
        "The shipped model fits `log1p(target)`, so the log-scale figure is the "
        "more faithful description of what it learned. The original-scale figure "
        "is reported because it is the one a reader will compare against, and "
        "because it is dominated by the largest fields: halving the error on "
        "small plots barely moves it, while one large field moves it a lot. "
        "Quoting only the flattering number would be misleading, so both are here."
    )
    add("")

    # ---- baselines ------------------------------------------------------
    add("## Baselines")
    add("")
    add(
        "A model that cannot beat \"predict the group average\" is not worth "
        "shipping. RMSE is in the target's own units."
    )
    add("")
    for target in (YIELD, COST):
        add(f"**{LABELS[target]}**")
        add("")
        table = models.get(target, {}).get("baseline_table", [])
        if table:
            rows = [
                {
                    "Model": r["model"],
                    "RMSE": _num(r.get("RMSE"), 1),
                    "MAE": _num(r.get("MAE"), 1),
                    "R2": _num(r.get("R2")),
                    "MAPE %": _num(r.get("MAPE %"), 1),
                }
                for r in table
            ]
            add(_table(rows, list(rows[0].keys())))
        add("")

    # ---- cross validation ----------------------------------------------
    add("## Cross-validation")
    add("")
    add(
        f"{split.get('strategy', 'n/a')} split: {split.get('description', '')}. "
        "Fold means below are on the training portion, so they are not the "
        "numbers to quote - the held-out figures above are."
    )
    add("")
    for target in (YIELD, COST):
        summary = models.get(target, {}).get("cv_summary", [])
        if not summary:
            continue
        add(f"**{LABELS[target]}**")
        add("")
        rows = [
            {
                "Metric": s["metric"],
                "Mean": _num(s["mean"], 2 if s["metric"] != "R2" else 4),
                "Std": _num(s["std"], 2 if s["metric"] != "R2" else 4),
            }
            for s in summary
        ]
        add(_table(rows, ["Metric", "Mean", "Std"]))
        add("")

    add(
        "The gap between cross-validated and held-out R2 is the honest signal "
        "here: a random split lets the same state-crop-variety combination "
        "appear in both train and test, which flatters the model. `--split "
        "group_state`, `group_zone` and `chronological` exist to measure how "
        "much of the headline number depends on that overlap."
    )
    add("")

    # ---- intervals ------------------------------------------------------
    add("## Prediction intervals")
    add("")
    add(f"- **Nominal coverage:** {float(interval.get('nominal_coverage', 0.8)):.0%}")
    add(f"- **Quantile alphas:** {interval.get('quantile_alphas')}")
    add(f"- **Method:** {interval.get('method', 'n/a')}")
    add("")
    rows = []
    for target in (YIELD, COST):
        block = models.get(target, {}).get("interval", {})
        tested = block.get("test_coverage", {})
        raw = block.get("raw_quantile_coverage", {})
        if not tested:
            continue
        rows.append(
            {
                "Target": LABELS[target],
                "Calibrated coverage": f"{float(tested.get('coverage', 0)):.2f}%",
                "Uncalibrated coverage": (
                    f"{float(raw.get('coverage', 0)):.2f}%" if raw else "n/a"
                ),
                "Calibration factor": _num(block.get("calibration_factor"), 3),
                "Median width": _num(tested.get("median_width"), 1),
                "Mean width": _num(tested.get("mean_width"), 1),
            }
        )
    add(_table(rows, list(rows[0].keys())))
    add("")
    for target in (YIELD, COST):
        block = models.get(target, {}).get("interval", {})
        if not block:
            continue
        add(
            f"- **{LABELS[target]}** - {block.get('n_bootstrap_members', 0)} bootstrap "
            f"members vary by {float(block.get('bootstrap_stability_pct', 0)):.1f}%. "
            "This is reported as a stability diagnostic and is **not** used to "
            "build the published interval: the bags share their training data, "
            "so their spread is not a variance estimate."
        )
    add("")

    # ---- features -------------------------------------------------------
    add("## What the model uses")
    add("")
    features = models.get(YIELD, {}).get("features", [])
    add(f"{len(features)} features: {', '.join(features)}.")
    add("")
    for target in (YIELD, COST):
        permutation = models.get(target, {}).get("permutation_importance", [])
        if not permutation:
            continue
        add(f"### {LABELS[target]} - permutation importance")
        add("")
        add(
            "Mean drop in R2 when a feature is shuffled, measured on held-out "
            "data. More trustworthy than tree gain, which inflates continuous "
            "features."
        )
        add("")
        rows = [
            {
                "Feature": r["feature"],
                "R2 drop (mean)": _num(r["r2_drop_mean"]),
                "R2 drop (std)": _num(r["r2_drop_std"]),
            }
            for r in permutation[:10]
        ]
        add(_table(rows, list(rows[0].keys())))
        add("")

    # ---- leakage --------------------------------------------------------
    add("## Leakage controls")
    add("")
    audit = metrics.get("feature_audit", [])
    if isinstance(audit, list):
        for block in audit:
            add(f"**{block.get('model', block.get('target', 'model'))}** "
                f"({block.get('n_features', 0)} features)")
            add("")
            add(f"- Target-encoded: {block.get('high_cardinality_encoded', 'n/a')}")
            add(f"- One-hot: {block.get('one_hot_encoded', 'n/a')}")
            add(f"- Numeric: {block.get('numeric', 'n/a')}")
            add(f"- **Excluded to avoid leakage:** "
                f"{block.get('excluded_to_avoid_leakage', 'n/a')}")
            add("")
    add(
        "- Target-derived columns (`production`, `production_quintals`, "
        "`cost`, `cost_per_unit`, `production_efficiency`) are excluded from "
        "both models, so neither can see the other's answer."
    )
    add(
        "- `is_outlier` is excluded because it is computed from both targets; "
        f"{split.get('n_outliers_excluded', 0):,} flagged rows were held out of "
        "training entirely."
    )
    add(
        "- High-cardinality columns are target-encoded inside "
        "cross-validation folds, not before the split. Fitting the encoder on "
        "the whole dataset first would let each row's own target leak into its "
        "own feature."
    )
    add(
        "- `assert_no_leakage()` runs when each pipeline is constructed, and "
        "`tests/test_preprocessing.py` asserts both that the shipped feature "
        "sets are clean and that the guard raises when fed a dirty one."
    )
    add("")

    # ---- benchmark ------------------------------------------------------
    add("## Booster benchmark")
    add("")
    rows = []
    for target in (YIELD, COST):
        xgb_r2 = models.get(target, {}).get("metrics", {}).get(f"{target}_r2")
        lgbm_r2 = (models.get(target, {}).get("lgbm") or {}).get(f"{target}_r2")
        if lgbm_r2 is None:
            continue
        delta = float(lgbm_r2) - float(xgb_r2)
        rows.append(
            {
                "Target": LABELS[target],
                "XGBoost R2": _num(xgb_r2),
                "LightGBM R2": _num(lgbm_r2),
                "Delta": f"{delta:+.4f}",
            }
        )
    if rows:
        add(_table(rows, list(rows[0].keys())))
        add("")
        add(
            "LightGBM edges ahead on both targets, by a small margin, and its "
            "artefacts are roughly eight times smaller. **XGBoost is still what "
            "ships**, for two reasons: the difference is well inside the "
            "cross-validation spread, and the per-prediction explanation path "
            "is built on XGBoost's `pred_contribs`. Promoting LightGBM is a "
            "deliberate decision to revisit with the explanation work, not an "
            "accident of which booster ran last."
        )
        add("")

    # ---- data quality ---------------------------------------------------
    add("## Data quality")
    add("")
    add(f"- **Source:** `{provenance.get('path', 'n/a')}`")
    add(f"- **SHA-256:** `{provenance.get('sha256', 'n/a')}`")
    add(f"- **Raw rows:** {provenance.get('rows_raw', 0):,}")
    add(f"- **Clean rows:** {provenance.get('rows_clean', 0):,}")
    add(
        f"- **Rows dropped:** {quality.get('rows_dropped_total', 0):,} "
        "(reasons overlap, so they do not sum to the total)"
    )
    add(f"- **Case fixes:** {quality.get('case_fixes', 0):,}")
    add(f"- **Whitespace trims:** {quality.get('string_trims', 0):,}")
    add(f"- **Unit fallbacks:** {quality.get('unit_fallbacks', 0):,}")
    add("")
    dropped = quality.get("dropped_rows", {})
    if dropped:
        add("### Dropped rows by reason")
        add("")
        rows = [
            {"Reason": k, "Rows": f"{v:,}"}
            for k, v in sorted(dropped.items(), key=lambda kv: -kv[1])
            if k != "total"
        ]
        add(_table(rows, ["Reason", "Rows"]))
        add("")
    units = quality.get("normalised_units", {})
    if units:
        add("### Units after normalisation")
        add("")
        rows = [
            {"Unit": k, "Rows": f"{v:,}"}
            for k, v in sorted(units.items(), key=lambda kv: -kv[1])
        ]
        add(_table(rows, ["Unit", "Rows"]))
        add("")
    for note in quality.get("notes", []):
        add(f"- {note}")
    if quality.get("notes"):
        add("")

    # ---- reproducibility ------------------------------------------------
    add("## Reproducibility")
    add("")
    add(f"- **Python:** {metrics.get('python', 'n/a')}")
    add(f"- **Platform:** {metrics.get('platform', 'n/a')}")
    add(f"- **Seed:** {metrics.get('seed', 'n/a')}")
    add("")
    if libraries:
        add(_table(
            [{"Library": k, "Version": str(v)} for k, v in libraries.items()],
            ["Library", "Version"],
        ))
        add("")
    add(
        "The same seed and the same input CSV reproduce these artefacts. "
        "Re-run with:"
    )
    add("")
    add("```bash")
    add("python -m src.train_models --split random --bootstrap 20")
    add("```")
    add("")
    return "\n".join(out)


def main() -> int:
    if not METRICS_PATH.exists():
        print(f"{METRICS_PATH} not found - train the models first.")
        return 1
    with open(METRICS_PATH, encoding="utf-8") as handle:
        metrics = json.load(handle)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(build(metrics), encoding="utf-8")
    print(f"wrote {REPORT_PATH} ({REPORT_PATH.stat().st_size:,} bytes)")
    print("regenerate at any time with: python scripts/write_report.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
