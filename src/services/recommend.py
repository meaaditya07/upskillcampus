"""Zone and state recommendations.

Ranking is computed from *observed* data rather than model predictions, on
purpose: a recommendation is a claim about where a crop does reliably well, and
backing it with many seasons of actual records is more defensible than backing
it with one model's extrapolation. The trained models are still what the
predictor page uses.

Three scores are combined per the weights in
:data:`config.settings.DEFAULT_SCORE_WEIGHTS`:

``efficiency``
    median quintals per hectare -- the headline agronomic number
``cost``
    median cost per hectare, inverted so lower is better
``reliability``
    inverse coefficient of variation of yield, so a place that swings wildly
    between seasons is penalised

Every ranking is guarded by a minimum sample size. A recommendation backed by
three observations is worse than no recommendation, so thin groups are returned
with ``confidence='low'`` and a reason rather than being silently ranked.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from config.settings import (
    CROP_SEASON_AFFINITY,
    DEFAULT_SCORE_WEIGHTS,
    INDICATIVE_PRICE_PER_QUINTAL,
    LOW_CONFIDENCE_N,
    QUINTAL_PER_TONNE,
    indicative_price,
)
from src.services import registry

#: Multiplier applied to each year back from the most recent, so recent seasons
#: dominate the ranking without older ones vanishing.
RECENCY_HALFLIFE = 6.0

MIN_YEARS_OBSERVED = 3


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted median, falling back to a plain median if weights degenerate."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values)
    values, weights = values[mask], weights[mask]
    if values.size == 0:
        return float("nan")
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        weights = np.ones_like(values)
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cutoff = 0.5 * weights.sum()
    index = int(np.searchsorted(np.cumsum(weights), cutoff))
    return float(values[min(index, values.size - 1)])


def robust_cv(values: np.ndarray) -> float:
    """Median-absolute-deviation coefficient of variation.

    A plain std/mean is useless here: a handful of 50x entry errors that survive
    the dataset-level outlier flag will push sigma arbitrarily high and make a
    reliable state look wildly erratic. MAD has a 50% breakdown point, so a few
    such rows cannot move it. The 1.4826 factor rescales MAD to be a consistent
    estimator of sigma for normally distributed data.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("nan")
    median = float(np.median(values))
    if abs(median) < 1e-9:
        return float("nan")
    mad = float(np.median(np.abs(values - median)))
    return 1.4826 * mad / abs(median)


def _recency_weights(years: pd.Series) -> np.ndarray:
    """Exponentially decaying weights, newest season weighted 1.0."""
    years = pd.to_numeric(years, errors="coerce")
    newest = np.nanmax(years.to_numpy(dtype=float)) if years.notna().any() else 0.0
    age = np.clip(newest - years.to_numpy(dtype=float), 0.0, None)
    return np.power(0.5, age / RECENCY_HALFLIFE)


def _normalise(series: pd.Series, higher_is_better: bool) -> pd.Series:
    """Min-max to [0, 1]; a flat column becomes 1.0 rather than 0/0."""
    numeric = pd.to_numeric(series, errors="coerce")
    low, high = numeric.min(), numeric.max()
    if not np.isfinite(low) or not np.isfinite(high) or high - low < 1e-12:
        return pd.Series(np.ones(len(numeric)), index=series.index)
    scaled = (numeric - low) / (high - low)
    return scaled if higher_is_better else 1.0 - scaled


@dataclass
class Recommendation:
    """A ranked table plus the provenance of the ranking."""

    table: pd.DataFrame
    scope: str
    weights: dict[str, float]
    n_observations: int

    def top(self, n: int = 10) -> pd.DataFrame:
        return self.table.head(n)

    def as_records(self, n: int = 10) -> list[dict[str, Any]]:
        return self.table.head(n).to_dict("records")


def _score_groups(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Aggregate to groups and compute the three component scores."""
    if frame.empty:
        return pd.DataFrame()

    frame = frame.copy()
    frame["_w"] = _recency_weights(frame["year"])
    frame["efficiency"] = frame["production_efficiency"]
    frame["cost_per_ha"] = frame["cost_per_unit"]

    rows = []
    for key, group in frame.groupby(group_cols, observed=True, sort=False):
        weights = group["_w"].to_numpy(dtype=float)
        efficiency = _weighted_median(
            group["efficiency"].to_numpy(dtype=float), weights
        )
        cost = _weighted_median(group["cost_per_ha"].to_numpy(dtype=float), weights)
        yields = group["efficiency"].to_numpy(dtype=float)
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, key_tuple, strict=True))
        row.update(
            n_observations=len(group),
            n_years=int(pd.to_numeric(group["year"], errors="coerce").nunique()),
            median_efficiency=efficiency,
            median_cost_per_ha=cost,
            yield_cv=robust_cv(yields),
            mean_production_quintals=float(
                np.nanmean(group["production_quintals"].to_numpy(dtype=float))
            )
            if len(group)
            else float("nan"),
        )
        rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    table["score_efficiency"] = _normalise(table["median_efficiency"], True)
    table["score_cost"] = _normalise(table["median_cost_per_ha"], False)
    table["score_reliability"] = (
        _normalise(table["yield_cv"], False)
        if table["yield_cv"].notna().any()
        else 1.0
    )

    weights = DEFAULT_SCORE_WEIGHTS
    table["score"] = (
        weights["efficiency"] * table["score_efficiency"]
        + weights["cost"] * table["score_cost"]
        + weights["reliability"] * table["score_reliability"]
    )
    table["confidence"] = np.where(
        (table["n_observations"] < LOW_CONFIDENCE_N)
        | (table["n_years"] < MIN_YEARS_OBSERVED),
        "low",
        "ok",
    )

    # Thin groups sink to the bottom instead of topping the list on a fluke.
    table["score"] = table["score"] - np.where(
        table["confidence"] == "low", 0.5, 0.0
    )
    table = table.sort_values("score", ascending=False).reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return table


def _prepare(frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """Usable rows for ranking: prepared, outlier-flagged, finite ratios."""
    if frame is None:
        frame = registry.load_dataset()
    if "is_outlier" in frame:
        frame = frame.loc[~frame["is_outlier"].fillna(False).astype(bool)]
    usable = frame.copy()
    for column in ("production_efficiency", "cost_per_unit"):
        if column in usable:
            usable = usable[np.isfinite(pd.to_numeric(usable[column], errors="coerce"))]
    return usable


def recommend_for_crop(
    crop: str,
    *,
    frame: pd.DataFrame | None = None,
    state: str | None = None,
    limit: int = 10,
) -> Recommendation:
    """Best states (or zones within a state) for growing ``crop``.

    Passing ``state`` switches the ranking from state level to zone level,
    which is the useful question once a farmer has already picked a state.
    """
    data = _prepare(frame)
    subset = data.loc[data["crop"] == crop]
    if subset.empty:
        return Recommendation(pd.DataFrame(), "crop", DEFAULT_SCORE_WEIGHTS, 0)

    scope = "state"
    cols = ["state"]
    if state is not None:
        subset = subset.loc[subset["state"] == state]
        scope = "zone"
        cols = ["recommended_zone"]

    table = _score_groups(subset, cols)
    if not table.empty:
        table = table.head(limit)
        table["crop"] = crop
    return Recommendation(table, scope, DEFAULT_SCORE_WEIGHTS, len(subset))


def recommend_state_crop_matrix(
    *,
    frame: pd.DataFrame | None = None,
    crops: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Every (crop, state) pair scored, for the recommendation heatmap."""
    data = _prepare(frame)
    wanted = set(crops) if crops else set(data["crop"].unique())
    subset = data.loc[data["crop"].isin(wanted)]
    table = _score_groups(subset, ["crop", "state"])
    if not table.empty:
        table = table.sort_values(["crop", "rank"]).reset_index(drop=True)
    return table


def recommend_seasons(
    state: str, crop: str, *, frame: pd.DataFrame | None = None
) -> Recommendation:
    """Which season to plant, given a state and crop."""
    data = _prepare(frame)
    subset = data.loc[(data["state"] == state) & (data["crop"] == crop)]
    if subset.empty:
        allowed = CROP_SEASON_AFFINITY.get(crop, [])
        table = pd.DataFrame({"season": list(allowed), "n_observations": 0})
        if not table.empty:
            table.insert(0, "rank", np.arange(1, len(table) + 1))
        return Recommendation(table, "season", DEFAULT_SCORE_WEIGHTS, 0)
    table = _score_groups(subset, ["season_name"])
    return Recommendation(table, "season", DEFAULT_SCORE_WEIGHTS, len(subset))


def recommend_varieties(
    state: str,
    crop: str,
    *,
    season: str | None = None,
    frame: pd.DataFrame | None = None,
    limit: int = 8,
) -> Recommendation:
    """Which cultivar, within a state (and optionally a season)."""
    data = _prepare(frame)
    subset = data.loc[(data["state"] == state) & (data["crop"] == crop)]
    if season:
        subset = subset.loc[subset["season_name"] == season]
    if subset.empty:
        return Recommendation(pd.DataFrame(), "variety", DEFAULT_SCORE_WEIGHTS, 0)
    table = _score_groups(subset, ["variety"]).head(limit)
    return Recommendation(table, "variety", DEFAULT_SCORE_WEIGHTS, len(subset))


def rank_with_profit(
    crop: str, *, frame: pd.DataFrame | None = None, limit: int = 10
) -> Recommendation:
    """State ranking for a crop, augmented with indicative margin per hectare.

    Efficiency alone can crown a place that grows well but cannot be profitably
    marketed. Adding an indicative margin lets the table say which is which.
    """
    result = recommend_for_crop(crop, frame=frame, limit=limit)
    table = result.table
    if table.empty:
        return result

    price, _ = indicative_price(crop)
    revenue = table["median_efficiency"] * price
    table = table.assign(
        indicative_price_per_quintal=price,
        indicative_revenue_per_ha=revenue,
        indicative_profit_per_ha=revenue - table["median_cost_per_ha"],
        indicative_margin_pct=np.where(
            revenue > 0,
            (revenue - table["median_cost_per_ha"]) / revenue * 100.0,
            np.nan,
        ),
        indicative_tonnes_per_ha=table["median_efficiency"] / QUINTAL_PER_TONNE,
    )
    return Recommendation(table, result.scope, result.weights, result.n_observations)


def price_table() -> pd.DataFrame:
    """The indicative price catalogue as a frame, for display."""
    rows = [
        {"crop": crop, "indicative_price_per_quintal": price}
        for crop, price in sorted(INDICATIVE_PRICE_PER_QUINTAL.items())
    ]
    return pd.DataFrame(rows)
