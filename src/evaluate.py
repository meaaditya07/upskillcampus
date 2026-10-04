"""Metrics, baselines, permutation importance and interval calibration.

Two reporting conventions are used throughout and the distinction matters:

*Original units*
    RMSE in quintals and RMSE in INR. These are what a planner actually cares
    about, but they are dominated by the largest observations.

*Log scale*
    The models are fit on ``log1p(target)``, so these are the metrics that
    describe the objective actually being optimised. Both are reported; neither
    is hidden.

:func:`regression_metrics` deliberately skips MAPE where the actual value is
zero or near zero, because MAPE is undefined there and silently returning 0
flatters the model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)


def _safe(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype="float64")


def regression_metrics(
    y_true, y_pred, target: str = ""
) -> dict[str, float]:
    """RMSE / MAE / R2 / MAPE / sMAPE for one array pair."""
    y_true = _safe(y_true)
    y_pred = _safe(y_pred)
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[finite], y_pred[finite]

    if y_true.size == 0:
        return {}

    rmse = float(root_mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))

    # R2 is undefined when the actuals have no variance.
    r2 = float(r2_score(y_true, y_pred)) if np.std(y_true) > 0 else float("nan")

    # MAPE is only meaningful away from zero.
    threshold = 1e-6 * max(float(np.mean(np.abs(y_true))), 1.0)
    mask = np.abs(y_true) > threshold
    mape = (
        float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)
        if mask.any()
        else float("nan")
    )

    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    smask = denom > threshold
    smape = (
        float(np.mean(np.abs(y_true[smask] - y_pred[smask]) / denom[smask]) * 100.0)
        if smask.any()
        else float("nan")
    )

    prefix = f"{target}_" if target else ""
    return {
        f"{prefix}rmse": rmse,
        f"{prefix}mae": mae,
        f"{prefix}r2": r2,
        f"{prefix}mape": mape,
        f"{prefix}smape": smape,
        f"{prefix}n": int(y_true.size),
        f"{prefix}mean_actual": float(np.mean(y_true)),
        f"{prefix}mean_predicted": float(np.mean(y_pred)),
        f"{prefix}bias": float(np.mean(y_pred) - np.mean(y_true)),
    }


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------
class GlobalMeanBaseline:
    """Predicts the training mean. The bar nothing should fail to clear."""

    name = "Global mean"
    supports_groups = False

    def fit(self, frame: pd.DataFrame, y: pd.Series) -> GlobalMeanBaseline:
        self.value_ = float(pd.to_numeric(y, errors="coerce").mean())
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.full(len(frame), self.value_)


class GroupMeanBaseline:
    """Predicts the training mean of the group (e.g. per-crop).

    A far stronger baseline than the global mean, and the one that actually
    demonstrates the model has learned more than 'crops differ'.
    """

    supports_groups = True

    def __init__(self, group: str = "crop") -> None:
        self.group = group
        self.name = f"Per-{group} mean"

    def fit(self, frame: pd.DataFrame, y: pd.Series) -> GroupMeanBaseline:
        values = pd.to_numeric(y, errors="coerce")
        self.group_means_ = (
            pd.DataFrame({self.group: frame[self.group].to_numpy(),
                          "y": values.to_numpy()})
            .groupby(self.group, observed=True)["y"]
            .mean()
        )
        self.global_mean_ = float(values.mean())
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        lookup = frame[self.group].map(self.group_means_)
        return lookup.fillna(self.global_mean_).to_numpy(dtype="float64")


def evaluate_baselines(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    groups: tuple[str, ...] = ("crop", "state", "variety"),
) -> list[dict]:
    """Score every baseline on the same test split as the real models."""
    y_train = pd.to_numeric(train[target], errors="coerce")
    y_test = pd.to_numeric(test[target], errors="coerce").to_numpy(dtype="float64")

    results: list[dict] = []
    for baseline in [GlobalMeanBaseline()] + [
        GroupMeanBaseline(g) for g in groups if g in train.columns
    ]:
        baseline.fit(train, y_train)
        predicted = np.clip(baseline.predict(test), 0.0, None)
        metrics = regression_metrics(y_test, predicted, target=target)
        results.append({"model": baseline.name, **metrics})
    return results


def baseline_table(results: list[dict], model_metrics: dict, target: str) -> pd.DataFrame:
    """Baselines plus the trained model, ready for display."""
    rows = list(results)
    rows.append(
        {
            "model": "XGBoost (this project)",
            **{k: v for k, v in model_metrics.items() if k.startswith(target)},
        }
    )
    frame = pd.DataFrame(rows)
    keep = ["model", f"{target}_rmse", f"{target}_mae", f"{target}_r2", f"{target}_mape"]
    keep = [c for c in keep if c in frame.columns]
    frame = frame.loc[:, keep].rename(
        columns={
            f"{target}_rmse": "RMSE",
            f"{target}_mae": "MAE",
            f"{target}_r2": "R2",
            f"{target}_mape": "MAPE %",
        }
    )
    if "R2" in frame.columns:
        frame = frame.sort_values("R2", ascending=False, na_position="last")
    return frame.reset_index(drop=True)


# --------------------------------------------------------------------------
# Interval calibration
# --------------------------------------------------------------------------
def interval_coverage(
    y_true, lower, upper
) -> dict[str, float]:
    """Empirical coverage of a prediction interval."""
    y_true = _safe(y_true)
    lower = _safe(lower)
    upper = _safe(upper)
    inside = (y_true >= lower) & (y_true <= upper)
    width = upper - lower
    return {
        "coverage": float(inside.mean() * 100.0),
        "mean_width": float(np.mean(width)),
        "median_width": float(np.median(width)),
        "n": int(y_true.size),
    }


def calibration_table(
    y_true, lower, upper, n_bins: int = 10
) -> pd.DataFrame:
    """Coverage by predicted-value decile, for the calibration chart.

    Well-calibrated intervals show ~80% coverage in every bin. Coverage that
    collapses in the upper bins means the model is over-confident exactly where
    the stakes are highest.
    """
    y_true = _safe(y_true)
    point = (_safe(lower) + _safe(upper)) / 2.0
    edges = np.quantile(point, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if edges.size < 3:
        return pd.DataFrame(
            columns=["bin", "n", "predicted_mid", "actual_mean", "coverage"]
        )

    bin_index = np.clip(np.digitize(point, edges[1:-1]), 0, edges.size - 2)
    rows = []
    for b in range(edges.size - 1):
        mask = bin_index == b
        if not mask.any():
            continue
        rows.append(
            {
                "bin": f"{edges[b]:,.0f}-{edges[b + 1]:,.0f}",
                "n": int(mask.sum()),
                "predicted_mid": float(np.mean(point[mask])),
                "actual_mean": float(np.mean(y_true[mask])),
                "coverage": float(
                    ((y_true[mask] >= lower[mask]) & (y_true[mask] <= upper[mask]))
                    .mean()
                    * 100.0
                ),
                "mean_width": float(np.mean(upper[mask] - lower[mask])),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Permutation importance
# --------------------------------------------------------------------------
def permutation_importance(
    pipeline,
    X,
    y_true,
    features: list[str],
    n_repeats: int = 5,
    n_rows: int = 1500,
    random_state: int = 0,
) -> pd.DataFrame:
    """Importance measured by degradation in R2 when a feature is shuffled.

    Gain-based importance is a property of the training procedure; permutation
    importance is a property of the fitted model's behaviour on unseen data,
    which is the thing we actually care about reporting.

    Scored on a random subsample: each repeat refits nothing but re-predicts,
    and 1,500 rows is ample for a stable R2 while keeping 5 repeats x ~17
    features interactive.
    """
    from sklearn.metrics import r2_score as _r2

    X = pd.DataFrame(X).reset_index(drop=True)
    y = np.asarray(y_true, dtype="float64")
    rng = np.random.default_rng(random_state)

    if len(X) > n_rows:
        idx = rng.choice(len(X), size=n_rows, replace=False)
        idx = np.sort(idx)
        X = X.iloc[idx].reset_index(drop=True)
        y = y[idx]

    baseline = _r2(y, np.asarray(pipeline.predict(X), dtype="float64"))
    rows = []
    for feature in features:
        if feature not in X.columns:
            continue
        drops = []
        for _ in range(n_repeats):
            shuffled = X.copy()
            shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
            score = _r2(
                y, np.asarray(pipeline.predict(shuffled), dtype="float64")
            )
            drops.append(baseline - score)
        rows.append(
            {
                "feature": feature,
                "r2_drop_mean": float(np.mean(drops)),
                "r2_drop_std": float(np.std(drops)),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("r2_drop_mean", ascending=False)
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------
# Residual diagnostics
# --------------------------------------------------------------------------
def residual_frame(y_true, y_pred) -> pd.DataFrame:
    y_true = _safe(y_true)
    y_pred = _safe(y_pred)
    residual = y_true - y_pred
    percentage = np.where(
        np.abs(y_true) > 1e-9, residual / np.abs(y_true) * 100.0, np.nan
    )
    return pd.DataFrame(
        {
            "actual": y_true,
            "predicted": y_pred,
            "residual": residual,
            "abs_residual": np.abs(residual),
            "pct_error": percentage,
        }
    )


def cross_validate(
    pipeline_factory,
    frame: pd.DataFrame,
    target: str,
    features: list[str],
    n_splits: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """K-fold CV on log scale, reported back in original units."""
    from sklearn.model_selection import KFold

    X = frame.loc[:, features]
    y = pd.to_numeric(frame[target], errors="coerce")
    y = y.fillna(y.median()).to_numpy(dtype="float64")

    rows = []
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X), start=1):
        pipeline = pipeline_factory()
        pipeline.fit(X.iloc[train_idx], y[train_idx])
        predicted = np.clip(
            np.asarray(pipeline.predict(X.iloc[test_idx]), dtype="float64"), 0, None
        )
        metrics = regression_metrics(y[test_idx], predicted, target=target)
        rows.append({"fold": fold, **metrics})
    return pd.DataFrame(rows)


def summarise_folds(folds: pd.DataFrame, target: str) -> pd.DataFrame:
    """Mean +/- standard deviation across folds, for a compact display."""
    metrics = [
        c for c in folds.columns
        if c.startswith(target) and c.endswith(("_rmse", "_mae", "_r2"))
    ]
    rows = [
        {
            "metric": m.replace(f"{target}_", "").upper(),
            "mean": float(folds[m].mean()),
            "std": float(folds[m].std()),
        }
        for m in metrics
    ]
    return pd.DataFrame(rows)
