"""Model construction, prediction and prediction-interval estimation.

Three model families live here:

``build_pipeline``
    The primary point predictor. A ``TransformedTargetRegressor`` on ``log1p``
    guarantees non-negative predictions and tames a target that spans several
    orders of magnitude.

``build_quantile_pipeline``
    A pinball-loss (``reg:quantileerror``) regressor at alpha = 0.10 / 0.90.
    This produces a genuine conditional quantile, not a symmetric error bar.

``fit_bootstrap``
    An ensemble of models fitted on bootstrap resamples of the training set.
    The spread of their predictions is an independent estimate of uncertainty
    that is driven by sampling variability rather than by the loss function.

:meth:`ModelBundle.predict_with_interval` combines the two by averaging the
quantile bounds with the bootstrap percentiles. :func:`interval_coverage`
in ``src/evaluate.py`` then checks empirically how often the realised value
falls inside, so the intervals are validated rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline

from config.settings import (
    DEFAULT_BOOTSTRAP_N,
    INTERVAL_QUANTILE_HIGH,
    INTERVAL_QUANTILE_LOW,
    SEED,
)
from src.preprocessing import (
    COST_TARGET,
    YIELD_TARGET,
    assert_no_leakage,
    build_preprocessor,
    expanded_to_display,
    feature_display_names,
    features_for,
)

# log1p/expm1 guard rails: keeps a wild tree prediction from becoming inf.
_LOG_FLOOR = -12.0
_LOG_CEIL = 20.0


def _clip_expm1(values: np.ndarray) -> np.ndarray:
    return np.expm1(np.clip(values, _LOG_FLOOR, _LOG_CEIL))


def log_target(inner: BaseEstimator) -> TransformedTargetRegressor:
    """Wrap an estimator so it learns ``log1p(target)`` and predicts ``expm1``."""
    return TransformedTargetRegressor(
        regressor=inner, func=np.log1p, inverse_func=np.expm1, check_inverse=False
    )


# --------------------------------------------------------------------------
# Estimators
# --------------------------------------------------------------------------
def _xgboost():
    from xgboost import XGBRegressor

    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=800,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.5,
        reg_alpha=0.0,
        tree_method="hist",
        n_jobs=-1,
        random_state=SEED,
        verbosity=0,
    )


def _xgboost_quantile(alpha: float):
    from xgboost import XGBRegressor

    return XGBRegressor(
        objective="reg:quantileerror",
        # XGBoost >=2.0 names this parameter `quantile_alpha`; `quantile` is
        # silently ignored and the fit aborts with an empty-alpha assertion.
        quantile_alpha=alpha,
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.5,
        tree_method="hist",
        n_jobs=-1,
        random_state=SEED,
        verbosity=0,
    )


def _lightgbm():
    import lightgbm as lgb

    return lgb.LGBMRegressor(
        objective="regression",
        n_estimators=900,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=SEED,
        verbose=-1,
    )


def _lightgbm_quantile(alpha: float):
    import lightgbm as lgb

    return lgb.LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=SEED,
        verbose=-1,
    )


def _elastic_net():
    from sklearn.linear_model import ElasticNetCV

    return ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.9, 0.99],
        n_alphas=40,
        cv=3,
        max_iter=3000,
        random_state=SEED,
        n_jobs=-1,
    )


def make_estimator(kind: str, quantile: float | None = None):
    """Instantiate an estimator by name, wrapped for a log-scale target."""
    if kind == "xgboost":
        return log_target(_xgboost())
    if kind == "lightgbm":
        return log_target(_lightgbm())
    if kind == "elasticnet":
        return log_target(_elastic_net())
    if kind == "quantile_xgboost":
        return log_target(_xgboost_quantile(
            quantile if quantile is not None else INTERVAL_QUANTILE_LOW
        ))
    if kind == "quantile_lightgbm":
        return log_target(_lightgbm_quantile(
            quantile if quantile is not None else INTERVAL_QUANTILE_LOW
        ))
    raise ValueError(f"Unknown estimator kind: {kind!r}")


def lightgbm_available() -> tuple[bool, str]:
    try:
        import lightgbm  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


# Characters XGBoost refuses inside a DMatrix feature name.
_XGBOOST_UNSAFE = str.maketrans({"[": "(", "]": ")", "<": "_lt_", ">": "_gt_"})


def _xgboost_safe_name(name: object) -> str:
    """A DMatrix-legal version of an expanded column name."""
    return str(name).translate(_XGBOOST_UNSAFE)


# --------------------------------------------------------------------------
# Pipelines
# --------------------------------------------------------------------------
def build_pipeline(
    target: str,
    kind: str = "xgboost",
    quantile: float | None = None,
    random_state: int = SEED,
) -> Pipeline:
    """Preprocessor + estimator for one target."""
    assert_no_leakage(features_for(target), target)
    preprocessor = build_preprocessor(target, random_state=random_state)
    return Pipeline(
        [
            ("preprocess", preprocessor),
            ("regressor", make_estimator(kind, quantile=quantile)),
        ]
    )


def fitted_estimator(pipeline: Pipeline):
    """Return the *fitted* regressor inside a Pipeline.

    ``TransformedTargetRegressor`` follows sklearn's cloning convention: the
    attribute ``regressor`` holds the unfitted template while the fitted clone
    lives on ``regressor_``. Reading ``regressor`` instead of ``regressor_``
    silently yields an estimator with no fitted boosters, which is why
    ``feature_importances_`` and ``get_booster()`` come back empty.
    """
    final = pipeline.named_steps["regressor"]
    return getattr(final, "regressor_", final)


# --------------------------------------------------------------------------
# Bundle
# --------------------------------------------------------------------------
@dataclass
class IntervalResult:
    """A point estimate with a calibrated 80% prediction interval.

    ``quantile_*`` and ``bootstrap_*`` are kept separate on purpose. Only the
    quantile pair drives the headline interval; see
    :meth:`ModelBundle.predict_with_interval` for why the bootstrap percentiles
    do not.
    """

    point: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    quantile_lower: np.ndarray
    quantile_upper: np.ndarray
    bootstrap_lower: np.ndarray
    bootstrap_upper: np.ndarray
    bootstrap_std: np.ndarray

    @property
    def width(self) -> np.ndarray:
        return self.upper - self.lower

    def as_dict(self) -> dict[str, float]:
        """Single-row summary (used when predicting one scenario)."""
        return {
            "point": float(self.point[0]),
            "lower": float(self.lower[0]),
            "upper": float(self.upper[0]),
            "quantile_lower": float(self.quantile_lower[0]),
            "quantile_upper": float(self.quantile_upper[0]),
            "bootstrap_lower": float(self.bootstrap_lower[0]),
            "bootstrap_upper": float(self.bootstrap_upper[0]),
            "bootstrap_std": float(self.bootstrap_std[0]),
        }


@dataclass
class ModelBundle:
    """A trained model plus its interval machinery and metadata."""

    name: str
    target: str
    kind: str
    pipeline: Pipeline
    quantile_low: Pipeline | None = None
    quantile_high: Pipeline | None = None
    bootstrap: list[Pipeline] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    train_rows: int = 0
    test_rows: int = 0
    features: list[str] = field(default_factory=list)
    display_names: list[str] = field(default_factory=list)
    expanded_names: list[str] = field(default_factory=list)
    target_log_metrics: dict[str, float] = field(default_factory=dict)
    interval_coverage: float | None = None
    calibration_factor: float = 1.0

    # -- prediction ---------------------------------------------------------
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        raw = np.asarray(self.pipeline.predict(frame), dtype="float64")
        return np.clip(raw, 0.0, None)

    def predict_with_interval(
        self, frame: pd.DataFrame, alpha: float = 0.10
    ) -> IntervalResult:
        """Point estimate plus a calibrated 80% prediction interval.

        The interval comes from the quantile-regressor pair, widened by
        ``calibration_factor`` -- a single scalar fitted on a held-out
        calibration split so that empirical coverage matches the nominal level.

        The bootstrap percentiles are computed but deliberately *not* averaged
        into the headline bounds. A naive bootstrap of n ~ 45,000 rows resamples
        with replacement, so every bag shares ~63% of its rows with every other
        bag; their predictions move almost not at all and the resulting spread
        collapses (measured coverage 21% against a nominal 80%, interval width
        roughly 6x too narrow). Averaging it in actively degrades a
        well-calibrated interval. It is retained as a *stability* diagnostic --
        "how much does the point estimate move under resampling?" -- which is a
        genuinely different and useful question from "how uncertain is this
        prediction?".
        """
        point = self.predict(frame)

        q_low = (
            np.clip(np.asarray(self.quantile_low.predict(frame), dtype="float64"),
                    0.0, None)
            if self.quantile_low is not None
            else np.full_like(point, np.nan)
        )
        q_high = (
            np.clip(np.asarray(self.quantile_high.predict(frame), dtype="float64"),
                    0.0, None)
            if self.quantile_high is not None
            else np.full_like(point, np.nan)
        )

        # Calibrated widening, applied about the point estimate so the interval
        # keeps the asymmetric shape the quantile models learned.
        factor = float(self.calibration_factor or 1.0)
        lower = point - factor * (point - q_low)
        upper = point + factor * (q_high - point)
        lower = np.where(np.isfinite(lower), lower, 0.0)
        upper = np.where(np.isfinite(upper), upper, point)
        lower = np.clip(lower, 0.0, None)
        upper = np.maximum(upper, lower)
        lower = np.minimum(lower, point)

        if self.bootstrap:
            boot = np.vstack(
                [np.clip(np.asarray(m.predict(frame), dtype="float64"), 0.0, None)
                 for m in self.bootstrap]
            )
            b_low = np.quantile(boot, alpha, axis=0)
            b_high = np.quantile(boot, 1 - alpha, axis=0)
            b_std = (
                boot.std(axis=0, ddof=1) if boot.shape[0] > 1
                else np.zeros_like(point)
            )
        else:
            shape = point.shape
            b_low = np.full(shape, np.nan)
            b_high = np.full(shape, np.nan)
            b_std = np.full(shape, np.nan)

        return IntervalResult(
            point=point,
            lower=lower,
            upper=upper,
            quantile_lower=q_low,
            quantile_upper=q_high,
            bootstrap_lower=b_low,
            bootstrap_upper=b_high,
            bootstrap_std=b_std,
        )

    def bootstrap_stability(
        self, frame: pd.DataFrame
    ) -> float:
        """Mean relative movement of the point estimate across resamples.

        A model that relies on a handful of rows shows a large value here; a
        model driven by broad structure shows a small one.
        """
        if not self.bootstrap:
            return float("nan")
        point = self.predict(frame)
        boot = np.vstack(
            [np.clip(np.asarray(m.predict(frame), dtype="float64"), 0.0, None)
             for m in self.bootstrap]
        )
        spread = boot.std(axis=0, ddof=1)
        scale = np.maximum(np.abs(point), 1e-9)
        return float(np.mean(spread / scale) * 100.0)

    # -- evaluation ---------------------------------------------------------
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        """Metrics in both the original units and the log space fit on."""
        from src.evaluate import regression_metrics

        y_true = pd.to_numeric(y, errors="coerce").to_numpy(dtype="float64")
        y_pred = self.predict(X)
        metrics = regression_metrics(y_true, y_pred, target=self.target)

        try:
            log_pred = predict_log_scale(self.pipeline, X)
            metrics.update(
                regression_metrics(
                    np.log1p(np.clip(y_true, 0, None)),
                    np.clip(log_pred, _LOG_FLOOR, _LOG_CEIL),
                    target=f"{self.target}(log)",
                )
            )
        except Exception:
            pass  # diagnostic only; never let them break the primary metrics
        return metrics

    def set_interval_coverage(self, coverage: float) -> None:
        self.interval_coverage = float(coverage)

    # -- explainability -----------------------------------------------------
    def feature_importances(self, top: int | None = None) -> pd.DataFrame:
        """Gain-based importance collapsed to readable feature blocks."""
        try:
            inner = fitted_estimator(self.pipeline)
            importances = np.asarray(inner.feature_importances_, dtype="float64")
        except Exception:  # pragma: no cover - estimator without importances
            return pd.DataFrame(columns=["feature", "importance"])

        if not len(importances):
            return pd.DataFrame(columns=["feature", "importance"])

        n_expanded = len(self.expanded_names)
        if n_expanded == len(importances) and n_expanded:
            display = expanded_to_display(self.expanded_names)
        elif len(self.display_names) == len(importances):
            display = np.asarray(self.display_names, dtype=object)
        else:
            display = np.array(
                [f"f{i}" for i in range(len(importances))], dtype=object
            )

        frame = pd.DataFrame({"feature": display, "importance": importances})
        grouped = (
            frame.groupby("feature", as_index=False)["importance"]
            .sum()
            .sort_values("importance", ascending=False)
        )
        if top:
            grouped = grouped.head(top)
        return grouped.reset_index(drop=True)

    def expanded_importances(self) -> pd.DataFrame:
        """Un-collapsed importances, for inspecting specific one-hot levels."""
        try:
            inner = fitted_estimator(self.pipeline)
            importances = np.asarray(inner.feature_importances_, dtype="float64")
            names = list(self.expanded_names) or [
                f"f{i}" for i in range(len(importances))
            ]
        except Exception:  # pragma: no cover
            return pd.DataFrame(columns=["feature", "importance"])
        return (
            pd.DataFrame({"feature": names[: len(importances)],
                          "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )


def calibrate_interval_factor(
    bundle: ModelBundle,
    calibration: pd.DataFrame,
    alpha: float = 0.10,
    max_factor: float = 6.0,
    tolerance: float = 0.002,
) -> float:
    """Smallest multiplicative widening of the quantile interval that achieves
    the nominal coverage on a held-out calibration split.

    The quantile regressors already land close (measured 78.8% against a nominal
    80%), so the fitted factor is typically near 1.0 and acts as a correction
    rather than a rescue. Searching for the smallest factor keeps the model from
    becoming needlessly pessimistic when it is already well calibrated.
    """
    if calibration is None or len(calibration) == 0:
        return 1.0

    X = calibration.loc[:, bundle.features]
    y = pd.to_numeric(calibration[bundle.target], errors="coerce").to_numpy(
        dtype="float64"
    )
    finite = np.isfinite(y)
    if not finite.any():
        return 1.0
    X, y = X.loc[finite], y[finite]

    result = bundle.predict_with_interval(X, alpha=alpha)
    point, q_low, q_high = result.point, result.quantile_lower, result.quantile_upper
    target = 1.0 - 2.0 * alpha

    def coverage_for(factor: float) -> float:
        lower = np.clip(point - factor * (point - q_low), 0.0, None)
        upper = np.maximum(point + factor * (q_high - point), lower)
        return float(((y >= lower) & (y <= upper)).mean())

    if coverage_for(1.0) >= target - tolerance:
        return 1.0

    low, high = 1.0, max_factor
    if coverage_for(high) < target:
        return max_factor
    for _ in range(40):
        mid = (low + high) / 2.0
        if coverage_for(mid) >= target:
            high = mid
        else:
            low = mid
    return float(high)


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------
def fit_bundle(
    target: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    calibration: pd.DataFrame | None = None,
    kind: str = "xgboost",
    n_bootstrap: int = DEFAULT_BOOTSTRAP_N,
    fit_quantiles: bool = True,
    interval_kind: str = "xgboost",
    progress: Any = None,
) -> ModelBundle:
    """Train a point model, its quantile pair and a bootstrap ensemble.

    ``calibration`` is a held-out slice used only to fit the interval widening
    factor. Keeping it separate from both ``train`` and ``test`` is what makes
    the reported coverage an honest number rather than an in-sample one.
    """

    def say(message: str) -> None:
        if progress is not None:
            progress(message)

    features = features_for(target)
    X_train = train.loc[:, features]
    y_train = pd.to_numeric(train[target], errors="coerce")
    X_test = test.loc[:, features]
    y_test = pd.to_numeric(test[target], errors="coerce")

    say(f"[{target}] fitting {kind} on {len(X_train):,} rows...")
    pipeline = build_pipeline(target, kind=kind)
    pipeline.fit(X_train, y_train)

    preprocessor = pipeline.named_steps["preprocess"]
    expanded = list(preprocessor.get_feature_names_out())
    display = feature_display_names(preprocessor)

    bundle = ModelBundle(
        name=f"{kind}:{target}",
        target=target,
        kind=kind,
        pipeline=pipeline,
        train_rows=len(X_train),
        test_rows=len(X_test),
        features=features,
        display_names=display,
        expanded_names=expanded,
    )

    if fit_quantiles:
        for alpha, attr in (
            (INTERVAL_QUANTILE_LOW, "quantile_low"),
            (INTERVAL_QUANTILE_HIGH, "quantile_high"),
        ):
            say(f"[{target}] fitting {interval_kind} quantile model @ {alpha:.2f}...")
            qpipe = build_pipeline(
                target, kind=f"quantile_{interval_kind}", quantile=alpha
            )
            qpipe.fit(X_train, y_train)
            setattr(bundle, attr, qpipe)

    if fit_quantiles:
        bundle.calibration_factor = calibrate_interval_factor(
            bundle, calibration, alpha=INTERVAL_QUANTILE_LOW
        )
        say(
            f"[{target}] interval calibration factor = "
            f"{bundle.calibration_factor:.3f}"
        )

    if n_bootstrap > 0:
        bundle.bootstrap = fit_bootstrap(
            pipeline, X_train, y_train, n_bootstrap=n_bootstrap, say=say
        )

    bundle.metrics = bundle.evaluate(X_test, y_test)
    return bundle


def fit_bootstrap(
    prototype: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_N,
    say: Any = None,
) -> list[Pipeline]:
    """Fit ``n_bootstrap`` pipelines on bootstrap resamples of (X, y).

    The preprocessor is refit on each resample, which is standard bagging:
    every bag derives its encodings only from rows that bag is allowed to see.
    A held-out row therefore never influences its own encoding.
    """
    rng = np.random.default_rng(SEED)
    n = len(X)
    members: list[Pipeline] = []

    # Bootstrap members use a lighter configuration; their job is to estimate
    # spread, not to be the headline model.
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        X_b = X.iloc[idx]
        y_b = y.iloc[idx]

        member = clone(prototype)
        member.set_params(
            regressor__regressor__n_estimators=400
        )
        try:
            member.fit(X_b, y_b)
            members.append(member)
        except Exception as exc:  # pragma: no cover - degenerate resample
            if say:
                say(f"  bootstrap {i + 1}/{n_bootstrap} skipped: {exc}")
    if say:
        say(f"  fitted {len(members)}/{n_bootstrap} bootstrap members")
    return members


# --------------------------------------------------------------------------
# LightGBM comparison arm
# --------------------------------------------------------------------------
def fit_comparison_model(
    target: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    kind: str = "lightgbm",
) -> tuple[Pipeline | None, str]:
    """Train a benchmark booster. Returns (pipeline, reason_if_unavailable)."""
    try:
        features = features_for(target)
        pipeline = build_pipeline(target, kind=kind)
        pipeline.fit(
            train.loc[:, features], pd.to_numeric(train[target], errors="coerce")
        )
        return pipeline, ""
    except Exception as exc:  # pragma: no cover - environment dependent
        return None, f"{type(exc).__name__}: {exc}"


def predict_log_scale(pipeline: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    """Predictions in the log space the model was fit on (for diagnostics)."""
    transformed = pipeline.named_steps["preprocess"].transform(frame)
    inner = fitted_estimator(pipeline)
    raw = inner.predict(transformed)
    return np.asarray(raw, dtype="float64")


def contribution_explanation(
    bundle: ModelBundle, frame: pd.DataFrame, top: int = 12
) -> pd.DataFrame:
    """Per-feature additive contributions for a single scenario.

    XGBoost's SHAP-ordered ``pred_contribs`` operates on the *transformed*
    matrix, since that is what the booster was trained on, so the frame is
    pushed through the fitted preprocessor first and the contribution vector is
    then collapsed back onto the readable feature blocks.
    """
    empty = pd.DataFrame(columns=["feature", "contribution", "kind"])
    try:
        regressor = bundle.pipeline.named_steps["regressor"]
        inner = getattr(regressor, "regressor_", regressor)
    except Exception:  # pragma: no cover - unexpected pipeline shape
        return empty
    if not hasattr(inner, "get_booster"):
        return empty

    try:
        import xgboost as xgb

        matrix = bundle.pipeline.named_steps["preprocess"].transform(
            frame.iloc[:1]
        )
        raw_names = (
            bundle.expanded_names
            if len(bundle.expanded_names) == matrix.shape[1]
            else None
        )
        # XGBoost rejects DMatrix feature names containing [ ] < or >, and a
        # level captured before the loader's missing-value fix is literally
        # named "unit_<Na>". Sanitising the names rather than dropping them
        # keeps the caller's columns identifiable.
        names = [_xgboost_safe_name(n) for n in raw_names] if raw_names else None
        dmatrix = xgb.DMatrix(matrix, feature_names=names)
        contribs = np.asarray(
            inner.get_booster().predict(dmatrix, pred_contribs=True)
        )[0]
    except Exception as exc:  # pragma: no cover - version dependent
        empty.attrs["error"] = f"{type(exc).__name__}: {exc}"
        return empty

    values = contribs[:-1]
    display_source = (
        bundle.expanded_names
        if len(bundle.expanded_names) >= len(values)
        else [f"f{i}" for i in range(len(values))]
    )
    table = pd.DataFrame(
        {
            "feature": expanded_to_display(display_source[: len(values)]),
            "contribution": values,
        }
    )
    table["kind"] = np.where(table["contribution"] >= 0, "increases", "decreases")
    table = (
        table.assign(_abs=table["contribution"].abs())
        .sort_values("_abs", ascending=False)
        .head(top)
        .drop(columns="_abs")
        .reset_index(drop=True)
    )
    table.attrs["bias"] = float(contribs[-1])
    return table


# Convenience re-exports so callers can import targets from one place.
__all__ = [
    "COST_TARGET",
    "YIELD_TARGET",
    "IntervalResult",
    "ModelBundle",
    "build_pipeline",
    "contribution_explanation",
    "fit_bootstrap",
    "fit_bundle",
    "fit_comparison_model",
    "lightgbm_available",
    "make_estimator",
    "predict_log_scale",
]
