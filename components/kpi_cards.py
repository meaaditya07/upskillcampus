"""KPI tiles and small formatted-number helpers.

Numbers a farmer acts on are shown twice -- once rounded for reading and once
exact in the tooltip -- because "1.2k quintals" and "1,247 quintals" support
very different decisions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import streamlit as st

from components import theme

TILES_PER_ROW = 4


def compact(value: float, unit: str = "") -> str:
    """1234567 -> '1.23M'."""
    number = float(value)
    for limit, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(number) >= limit:
            text = f"{number / limit:.2f}".rstrip("0").rstrip(".")
            return f"{text}{suffix}{unit}"
    return f"{number:,.0f}{unit}"


def quintals(value: float) -> str:
    """Quintals, spelled out with the Indian conversion alongside."""
    from config.settings import QUINTAL_PER_TONNE

    return f"{value:,.0f} q ({value / QUINTAL_PER_TONNE:,.1f} t)"


def rupees(value: float, precise: bool = False) -> str:
    """Rupees, in Indian digit grouping (12,34,567)."""
    if not precise:
        return f"₹{compact(value)}"
    negative = value < 0
    digits = f"{abs(round(value)):,}"
    # Regroup the last three digits, then every pair.
    head, _, tail = digits.rpartition(",")
    if tail:
        digits = f"{head},{tail}"
    parts = digits.split(",")
    if len(parts) > 2:
        grouped = ",".join(parts[:-2]) + "," + ",".join(
            f"{int(p):02d}" for p in parts[-2:]
        )
        digits = grouped
    return f"{'-' if negative else ''}₹{digits}"


def hectares(value: float) -> str:
    """Hectares, grouped."""
    return f"{value:,.0f} ha"


def percent(value: float, digits: int = 1) -> str:
    """A percentage to one decimal by default."""
    return f"{value:,.{digits}f}%"


def row(tiles: Sequence[tuple[str, Any, str]], per_row: int = TILES_PER_ROW) -> None:
    """Render tiles in a responsive grid.

    Each tile is ``(label, value, note)``.
    """
    tiles = list(tiles)
    for start in range(0, len(tiles), per_row):
        chunk = tiles[start: start + per_row]
        columns = st.columns(len(chunk))
        for column, (label, value, note) in zip(columns, chunk, strict=True):
            with column:
                theme.card(label, value, note)


def kpi_strip(frame, metrics: Mapping[str, Any] | None = None) -> None:
    """The headline tiles on the overview page."""
    from src.services import insights

    values = dict(metrics or insights.headline_kpis(frame))
    if not values:
        return
    row(
        [
            ("Observations", f"{values.get('observations', 0):,}", "clean rows"),
            ("States / UTs", f"{values.get('states', 0):,}",
             f"{values.get('crops', 0):,} crops"),
            (
                "Total production",
                quintals(values.get("total_production_quintals", 0.0)).split(" (")[0],
                f"{values.get('total_production_tonnes', 0.0):,.0f} tonnes",
            ),
            (
                "Area cultivated",
                hectares(values.get("total_area_ha", 0.0)),
                rupees(values.get("total_cost_inr", 0.0)) + " spent",
            ),
        ]
    )


def model_tiles(metrics: Mapping[str, Any]) -> None:
    """Accuracy tiles from metrics.json, so the claim is never hard-coded."""
    from src.preprocessing import COST_TARGET, YIELD_TARGET

    models = metrics.get("models", {})
    produced = models.get(YIELD_TARGET, {}).get("metrics", {})
    cost = models.get(COST_TARGET, {}).get("metrics", {})
    coverage = (
        models.get(YIELD_TARGET, {}).get("interval", {})
        .get("test_coverage", {})
        .get("coverage")
    )
    tiles = []
    for label, block, target in (
        ("Production R²", produced, YIELD_TARGET),
        ("Cost R²", cost, COST_TARGET),
    ):
        r2 = block.get(f"{target}_r2")
        rmse = block.get(f"{target}_rmse")
        if r2 is None:
            continue
        tiles.append(
            (
                label,
                f"{float(r2):.3f}",
                f"RMSE {compact(float(rmse))}" if rmse is not None else "",
            )
        )
    if coverage is not None:
        tiles.append(
            ("Interval coverage", percent(float(coverage)), "nominal 80%")
        )
    if tiles:
        row(tiles)
