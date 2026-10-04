"""Diagnostics: what the model is worth, and what it cannot do.

This page exists to be the opposite of a sales pitch. It shows held-out
accuracy, interval calibration, what the model uses, the rows it refuses to
trust, and the baseline it has to beat to be worth anything.
"""

from __future__ import annotations

import streamlit as st

from components import charts, kpi_cards, sidebar, theme
from config.settings import METRICS_PATH
from src.preprocessing import COST_TARGET, YIELD_TARGET, audit_feature_sets
from src.services import insights, registry


def render() -> None:
    sidebar.sidebar("Diagnostics")

    st.markdown("## Diagnostics")
    st.caption(
        "Held-out accuracy, interval calibration, feature use and the rows "
        "excluded as suspect. Everything here comes from a test split the "
        "models never saw."
    )

    status = registry.training_status()
    metrics = registry.load_metrics() if METRICS_PATH.exists() else {}

    if not metrics:
        st.warning("metrics.json not found. Run training to populate this page.")
        return

    _model_accuracy(metrics)
    st.divider()
    _intervals(metrics)
    st.divider()
    _baselines(metrics)
    st.divider()
    _features(metrics)
    st.divider()
    _leakage_audit()
    st.divider()
    _predictions(metrics)
    st.divider()
    _data_quality(metrics, status)
    st.divider()
    _environment(metrics, status)


def _model_accuracy(metrics: dict) -> None:
    theme.section("Held-out accuracy")
    st.caption(
        "Random split, models fitted on the training portion only. R² above 0 "
        "means better than predicting the mean; the baselines below show what "
        "that is actually worth."
    )
    tiles = []
    for target, label in ((YIELD_TARGET, "Production"), (COST_TARGET, "Cost")):
        block = metrics.get("models", {}).get(target, {}).get("metrics", {})
        r2 = block.get(f"{target}_r2")
        mae = block.get(f"{target}_mae")
        smape = block.get(f"{target}_smape")
        if r2 is None:
            continue
        detail = []
        if mae is not None:
            detail.append(f"MAE {float(mae):,.1f}")
        if smape is not None:
            detail.append(f"SMAPE {float(smape):.1f}%")
        tiles.append((f"{label} R²", f"{float(r2):.3f}", " · ".join(detail)))
    if tiles:
        kpi_cards.row(tiles)

    log_block = metrics.get("models", {}).get(YIELD_TARGET, {}).get("metrics", {})
    log_r2 = log_block.get(f"{YIELD_TARGET}(log)_r2")
    if log_r2 is not None:
        theme.note(
            f"The shipped model fits `log1p(target)`, where R² is "
            f"**{float(log_r2):.3f}**. The headline {float(log_block[f'{YIELD_TARGET}_r2']):.3f} "
            "is on the original rupee/quintal scale, which is dominated by the "
            "largest fields. Both are reported because quoting only the "
            "flattering one would be misleading."
        )


def _intervals(metrics: dict) -> None:
    theme.section("Prediction intervals")
    method = metrics.get("interval", {})
    st.caption(
        f"Method: {method.get('method', 'n/a')}"
    )
    tiles = []
    for target, label in ((YIELD_TARGET, "Production"), (COST_TARGET, "Cost")):
        interval = metrics.get("models", {}).get(target, {}).get("interval", {})
        tested = interval.get("test_coverage", {})
        if not tested:
            continue
        tiles.append(
            (
                f"{label} coverage",
                kpi_cards.percent(float(tested.get("coverage", 0.0))),
                f"median width {float(tested.get('median_width', 0.0)):,.0f}",
            )
        )
        raw = interval.get("raw_quantile_coverage", {})
        if raw:
            tiles.append(
                (
                    f"{label} uncalibrated",
                    kpi_cards.percent(float(raw.get("coverage", 0.0))),
                    "before widening",
                )
            )
    if tiles:
        kpi_cards.row(tiles)

    members = metrics.get("models", {}).get(YIELD_TARGET, {}).get("interval", {})
    stability = members.get("bootstrap_stability_pct")
    if stability is not None:
        theme.note(
            f"A {int(members.get('n_bootstrap_members', 0))}-member bootstrap "
            f"ensemble varies by {float(stability):.1f}% across members. It is "
            "reported as a stability diagnostic only and is **not** used for the "
            "published intervals: the bags are too correlated for averaging "
            "them to be a variance estimate.",
            kind="info",
        )

    try:
        cache = registry.load_prediction_cache()
        st.plotly_chart(
            charts.calibration_curve(cache, YIELD_TARGET), width="stretch"
        )
    except Exception as exc:
        theme.note(f"No cached predictions: {exc}", kind="warning")


def _benchmark_summary(metrics: dict) -> str:
    """One-line verdict on the LightGBM benchmark, read from per-model blocks."""
    models = metrics.get("models", {})
    deltas = []
    for target, label in ((YIELD_TARGET, "production"), (COST_TARGET, "cost")):
        block = models.get(target, {})
        lgbm = block.get("lgbm")
        xgb_r2 = block.get("metrics", {}).get(f"{target}_r2")
        if not lgbm or xgb_r2 is None:
            continue
        lgbm_r2 = lgbm.get(f"{target}_r2")
        if lgbm_r2 is None:
            continue
        deltas.append((label, float(lgbm_r2) - float(xgb_r2)))
    if not deltas:
        return "not benchmarked (lightgbm not installed during training)"
    detail = ", ".join(f"{label} {delta:+.4f}" for label, delta in deltas)
    return f"ran; {detail} R2 vs XGBoost. Shipped: XGBoost."


def _baselines(metrics: dict) -> None:
    theme.section("Baselines")
    st.caption(
        "A model that cannot beat 'predict the average' is not worth shipping. "
        "These are the numbers it has to clear."
    )
    for target, label in ((YIELD_TARGET, "Production"), (COST_TARGET, "Cost")):
        table = metrics.get("models", {}).get(target, {}).get("baseline_table")
        if not table:
            continue
        st.markdown(f"**{label}**")
        st.dataframe(table, width="stretch", hide_index=True)


def _features(metrics: dict) -> None:
    theme.section("What the model uses")
    left, right = st.columns(2)
    for column, target, label in (
        (left, YIELD_TARGET, "Production"),
        (right, COST_TARGET, "Cost"),
    ):
        with column:
            importance = metrics.get("models", {}).get(target, {}).get("feature_importance")
            if importance is None:
                st.info(f"No feature importance recorded for {label.lower()}.")
                continue
            st.markdown(f"**{label}**")
            st.plotly_chart(
                charts.feature_importance(importance), width="stretch"
            )


def _leakage_audit() -> None:
    theme.section("Leakage audit")
    st.caption(
        "Columns each model is forbidden to see, and why. Production, cost and "
        "both derived ratios are excluded from both models; so is the outlier "
        "flag, because it is computed from the targets themselves."
    )
    audit = audit_feature_sets()
    st.dataframe(
        audit,
        width="stretch",
        hide_index=True,
        column_config={
            "one_hot_encoded": st.column_config.TextColumn(width="large"),
            "high_cardinality_encoded": st.column_config.TextColumn(width="large"),
        },
    )
    theme.note(
        "High-cardinality columns (state, crop, variety, zone) are target-encoded "
        "**inside cross-validation folds**. Encoding them on the full dataset "
        "first would let each row's own target leak into its own feature.",
        kind="ok",
    )


def _predictions(metrics: dict) -> None:
    theme.section("Held-out predictions")
    try:
        cache = registry.load_prediction_cache()
    except Exception as exc:
        theme.note(f"prediction cache unavailable: {exc}", kind="warning")
        return

    target = st.selectbox(
        "Target", [YIELD_TARGET, COST_TARGET],
        format_func=lambda t: "Production (quintals)" if t == YIELD_TARGET else "Cost (INR)",
    )
    left, right = st.columns(2)
    with left:
        st.markdown("**Predicted vs actual** (log scale)")
        st.plotly_chart(
            charts.actual_vs_predicted(cache, target), width="stretch"
        )
    with right:
        st.markdown("**Residuals**")
        st.plotly_chart(
            charts.residual_histogram(cache, target), width="stretch"
        )
    st.caption(
        "Points hugging the dashed diagonal are predictions the model gets "
        "right. A visible fan at high production means the model "
        "under-predicts the biggest fields."
    )


def _data_quality(metrics: dict, status: dict) -> None:
    theme.section("Data quality")
    quality = metrics.get("data_quality", {})
    dropped = quality.get("dropped_rows", {})
    tiles = [
        ("Raw rows", f"{quality.get('rows_in', 0):,}", "as read from CSV"),
        ("Clean rows", f"{quality.get('rows_out', 0):,}", "usable"),
        (
            "Dropped",
            f"{quality.get('rows_dropped_total', 0):,}",
            "reasons overlap, so reasons do not sum to the total",
        ),
        (
            "Case / trim fixes",
            f"{quality.get('case_fixes', 0) + quality.get('string_trims', 0):,}",
            "whitespace and capitalisation",
        ),
    ]
    kpi_cards.row(tiles)

    if dropped:
        st.markdown("**Why rows were dropped**")
        st.dataframe(
            sorted(dropped.items(), key=lambda kv: -kv[1]),
            width="stretch", hide_index=True,
            column_config={"0": "Reason", "1": "Rows"},
        )

    normalised = quality.get("normalised_units", {})
    if normalised:
        st.markdown("**Units after normalisation**")
        st.dataframe(
            sorted(normalised.items(), key=lambda kv: -kv[1]),
            width="stretch", hide_index=True,
            column_config={"0": "Unit", "1": "Rows"},
        )
        theme.note(
            "An Indian quintal is 50 kg, so one tonne is 20 quintals. "
            "Production is converted once, here, and never again downstream. "
            "Rows with an unreadable unit fall back to quintals and are counted "
            "in `unit_fallbacks`.",
            kind="info",
        )

    theme.section("Flagged outliers")
    report = insights.outlier_report(top=25)
    if report.empty:
        theme.note("No rows were flagged as outliers.", kind="ok")
    else:
        st.caption(
            "Held out of training and excluded from every chart. These look "
            "like entry errors - a decimal point in the wrong place, a unit "
            "typed twice - rather than real extremes."
        )
        st.dataframe(report, width="stretch", hide_index=True)


def _environment(metrics: dict, status: dict) -> None:
    theme.section("Reproducibility")
    libraries = metrics.get("libraries", {})
    # provenance is None (not {}) before training, so guard the lookup itself.
    provenance = status.get("provenance") or {}
    rows = [
        ("Generated at", metrics.get("generated_at", "n/a")),
        ("Seed", str(metrics.get("seed", "n/a"))),
        ("Python", metrics.get("python", "n/a")),
        ("Platform", metrics.get("platform", "n/a")),
        ("Dataset SHA-256", provenance.get("sha256") or "n/a"),
        ("Training time", f"{metrics.get('elapsed_seconds') or 0:,.1f} s"),
        (
            "LightGBM benchmark",
            _benchmark_summary(metrics),
        ),
    ]
    for name, value in libraries.items():
        rows.append((name, str(value)))
    st.dataframe(
        rows, width="stretch", hide_index=True,
        column_config={"0": "Setting", "1": "Value"},
    )
    theme.note(
        "Retraining is deterministic: the same seed and the same input CSV "
        "produce the same artefacts.",
        kind="ok",
    )

    with st.expander("Raw metrics.json"):
        st.json(metrics)
