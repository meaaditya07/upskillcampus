"""Scenario prediction: turn user inputs into yield and cost forecasts.

This is the piece most likely to be called incorrectly, so it owns the two
invariants that matter:

1. ``engineer_features`` must run before any bundle sees the frame. The
   pipeline's ``ColumnTransformer`` selects ``log_quantity`` / ``sqrt_quantity``
   / ``year_index``, which only exist after engineering, so skipping it fails
   deep inside sklearn with an opaque column error.
2. Only columns the model was trained on may be supplied. Extra columns are
   harmless (the transformer drops them), but *missing* ones are not.

Streamlit-free by design.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from config.settings import (
    CROP_SEASON_AFFINITY,
    QUINTAL_PER_TONNE,
    SEASON_DURATIONS,
    SEASON_NAMES,
    YEAR_MAX,
    YEAR_MIN,
)
from src.preprocessing import engineer_features
from src.schema import (
    canonical_crop,
    canonical_state,
    canonical_zone,
    classify_variety,
    crop_group,
    parse_season,
    parse_zone,
)
from src.services import registry

YIELD_TARGET = "production_quintals"
COST_TARGET = "cost"


@dataclass
class Scenario:
    """One farmer's plot and season.

    Defaults are the dataset medians, so a bare ``Scenario()`` is already a
    plausible mid-country field rather than a set of arbitrary round numbers.
    """

    state: str = "Uttar Pradesh"
    crop: str = "Rice"
    variety: str = "Pusa Basmati 1"
    recommended_zone: str = "Uttar Pradesh"
    season: str = "Kharif"
    # Left as None so it is *derived* from the season by parse_season, exactly
    # as training did. Hard-coding "Medium" here would pair Kharif with Medium
    # and Rabi with Medium, and a season/duration pair the model never saw in
    # training is silently zeroed rather than flagged.
    season_duration: str | None = None
    area_ha: float = 1.0
    year: int = 2014

    def row(self) -> dict[str, Any]:
        """A one-row raw frame, pre-feature-engineering.

        Every categorical goes through the *same* canonicalisation the loader
        applies. This is not cosmetic: ``state``, ``crop``, ``season_name`` and
        ``recommended_zone`` are one-hot inputs, so a stray leading space or a
        lowercase "wheat" would otherwise be a level the model never saw in
        training. ``handle_unknown="ignore"`` would then quietly zero those
        columns and return a confident answer computed from the wrong inputs.

        ``crop_group``, ``variety_class`` and ``zone_scope`` are derived with
        the loader's helpers for the same reason -- a private reimplementation
        here would emit training-unknown levels.
        """
        state = canonical_state(self.state)
        crop = canonical_crop(self.crop)
        # parse_zone returns (scope, name) -- same order the loader consumes.
        zone_scope, _ = parse_zone(self.recommended_zone)
        zone = canonical_zone(self.recommended_zone)
        season, season_duration = parse_season(self.season)
        return {
            "state": state,
            "crop": crop,
            "variety": self.variety,
            "recommended_zone": zone,
            "season_name": season,
            "season_duration": self.season_duration or season_duration,
            "quantity": float(self.area_ha),
            "year": float(self.year),
            "unit": "Quintals",
            "zone_scope": zone_scope,
            "crop_group": crop_group(crop),
            "variety_class": classify_variety(self.variety),
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def scenarios_to_frame(scenarios: Iterable[Scenario] | pd.DataFrame) -> pd.DataFrame:
    """Build the raw model-input frame for one or many scenarios."""
    if isinstance(scenarios, pd.DataFrame):
        return scenarios.copy()
    rows = [s.row() for s in scenarios]
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _check_columns(bundle, frame: pd.DataFrame) -> None:
    required = list(bundle.features)
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(
            f"{bundle.name} needs columns that are absent: {missing}. "
            "Build the frame with scenarios_to_frame()."
        )


@dataclass
class Forecast:
    """Point forecasts with calibrated intervals, per scenario."""

    rows: pd.DataFrame
    yield_point: np.ndarray
    yield_lower: np.ndarray
    yield_upper: np.ndarray
    cost_point: np.ndarray
    cost_lower: np.ndarray
    cost_upper: np.ndarray

    def __len__(self) -> int:
        return len(self.rows)

    def first(self) -> dict[str, float]:
        """Single-scenario summary, as the predictor page displays it."""
        return {
            "yield_quintals": float(self.yield_point[0]),
            "yield_lower": float(self.yield_lower[0]),
            "yield_upper": float(self.yield_upper[0]),
            "yield_tonnes": float(self.yield_point[0]) / QUINTAL_PER_TONNE,
            "cost": float(self.cost_point[0]),
            "cost_lower": float(self.cost_lower[0]),
            "cost_upper": float(self.cost_upper[0]),
            "cost_per_hectare": float(self.cost_point[0]),
        }

    def per_hectare(self) -> np.ndarray:
        area = self.rows["quantity"].to_numpy(dtype=float)
        return np.divide(
            self.yield_point, np.where(area > 0, area, np.nan)
        )

    def interval_width(self) -> np.ndarray:
        return self.yield_upper - self.yield_lower


def forecast(
    scenarios: Iterable[Scenario] | pd.DataFrame,
    *,
    bundles: dict[str, Any] | None = None,
) -> Forecast:
    """Predict production and cost for one or more scenarios.

    Both targets are forecast from the *same* input frame, which is what makes
    the profitability maths on the predictor page coherent: one area, one
    season, two consistent numbers.
    """
    if bundles is None:
        bundles = registry.load_bundles()

    raw = scenarios_to_frame(scenarios)
    if raw.empty:
        raise ValueError("no scenarios supplied")

    frame = engineer_features(raw)

    for target in (YIELD_TARGET, COST_TARGET):
        _check_columns(bundles[target], frame)

    y_bundle = bundles[YIELD_TARGET]
    c_bundle = bundles[COST_TARGET]

    y_int = y_bundle.predict_with_interval(frame)
    c_int = c_bundle.predict_with_interval(frame)

    out = raw.copy()
    return Forecast(
        rows=out,
        yield_point=y_int.point,
        yield_lower=y_int.lower,
        yield_upper=y_int.upper,
        cost_point=c_int.point,
        cost_lower=c_int.lower,
        cost_upper=c_int.upper,
    )


def predict_one(scenario: Scenario, **kwargs) -> Forecast:
    return forecast([scenario], **kwargs)


def explain(scenario: Scenario, *, target: str = YIELD_TARGET, top: int = 10):
    """Per-feature contributions for one scenario."""
    from src.models import contribution_explanation

    bundle = registry.load_bundle(target)
    frame = engineer_features(scenarios_to_frame([scenario]))
    _check_columns(bundle, frame)
    return contribution_explanation(bundle, frame, top=top)


# --------------------------------------------------------------------------
# Reference data for the input controls
# --------------------------------------------------------------------------
def known_states() -> list[str]:
    from src.schema import known_states as _known

    return _known()


def crops() -> list[str]:
    from src.schema import CROP_ALIASES

    return sorted(set(CROP_ALIASES.values()))


def varieties_for(crop: str) -> list[str]:
    """Varieties actually observed for a crop, most common first.

    Read from the dataset rather than hard-coded, so the dropdown can never
    offer a combination the model has no evidence for.
    """
    try:
        frame = registry.load_dataset()
    except Exception:
        return ["Unknown"]
    subset = frame.loc[frame["crop"] == crop, "variety"]
    if subset.empty:
        return ["Unknown"]
    counts = subset.value_counts()
    return list(counts.index)


def zones_for(state: str) -> list[str]:
    try:
        frame = registry.load_dataset()
    except Exception:
        return [state]
    subset = frame.loc[frame["state"] == state, "recommended_zone"].dropna()
    if subset.empty:
        return [state]
    return list(subset.value_counts().index)


def seasons_for_crop(crop: str) -> list[str]:
    """Seasons a crop is actually grown in, per the configured affinity table."""
    allowed = CROP_SEASON_AFFINITY.get(crop)
    if not allowed:
        return list(SEASON_NAMES)
    return [s for s in SEASON_NAMES if s in allowed]


def season_durations() -> list[str]:
    return list(SEASON_DURATIONS)


def year_range() -> tuple[int, int]:
    return (YEAR_MIN, YEAR_MAX)


def scenario_from_row(row: dict[str, Any]) -> Scenario:
    """Build a Scenario from a flat dict, ignoring anything unrecognised."""
    fields = set(Scenario.__dataclass_fields__)
    kwargs: dict[str, Any] = {}
    for key, value in row.items():
        normalised = {
            "season_name": "season",
            "area_ha": "area_ha",
            "quantity": "area_ha",
        }.get(key, key)
        if normalised in fields:
            kwargs[normalised] = value
    kwargs.setdefault("area_ha", row.get("quantity", 1.0))
    return Scenario(**kwargs)
