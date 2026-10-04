"""Aggregations for the overview / EDA page.

Each function returns a small tidy frame that a chart can consume directly, so
the page stays declarative and the numbers stay testable outside Streamlit.

Every aggregation is computed on the outlier-flagged-clean subset: a single
50x production entry error otherwise dominates a national total.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from config.settings import QUINTAL_PER_TONNE, YEAR_MIN
from src.services import registry

CURRENCY = "INR"


def _frame() -> pd.DataFrame:
    """Prepared data with entry-error rows removed.

    ``registry.load_dataset`` returns the frame as training prepared it, so the
    ``is_outlier`` flag is present; the guard is only there so the aggregations
    still work if a caller passes a raw frame in.
    """
    frame = registry.load_dataset()
    if "is_outlier" in frame:
        frame = frame.loc[~frame["is_outlier"].fillna(False).astype(bool)]
    return frame.copy()


def headline_kpis(frame: pd.DataFrame | None = None) -> dict[str, Any]:
    """The numbers in the top KPI strip."""
    data = _frame() if frame is None else frame
    if data.empty:
        return {}

    total_production = float(data["production_quintals"].sum())
    total_area = float(data["quantity"].sum())
    total_cost = float(data["cost"].sum())

    return {
        "observations": len(data),
        "states": int(data["state"].nunique()),
        "crops": int(data["crop"].nunique()),
        "zones": int(data["recommended_zone"].nunique()),
        "year_min": int(data["year"].min()),
        "year_max": int(data["year"].max()),
        "total_production_quintals": total_production,
        "total_production_tonnes": total_production / QUINTAL_PER_TONNE,
        "total_area_ha": total_area,
        "total_cost_inr": total_cost,
        "mean_efficiency_q_per_ha": float(data["production_efficiency"].mean()),
        "median_efficiency_q_per_ha": float(data["production_efficiency"].median()),
        "mean_cost_per_ha": float(data["cost_per_unit"].mean()),
        "median_cost_per_ha": float(data["cost_per_unit"].median()),
        "total_cost_per_ha": total_cost / total_area if total_area else float("nan"),
        "yield_per_hectare_tonnes": (
            data["production_efficiency"].mean() / QUINTAL_PER_TONNE
        ),
    }


def production_by_year(frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """National production, area and cost totals per season."""
    data = _frame() if frame is None else frame
    grouped = data.groupby("year", as_index=False).agg(
        production_quintals=("production_quintals", "sum"),
        area_ha=("quantity", "sum"),
        cost_inr=("cost", "sum"),
        observations=("production_quintals", "size"),
    )
    grouped = grouped.sort_values("year").reset_index(drop=True)
    grouped["production_tonnes"] = grouped["production_quintals"] / QUINTAL_PER_TONNE
    grouped["yield_q_per_ha"] = (
        grouped["production_quintals"] / grouped["area_ha"]
    )
    grouped["cost_per_ha"] = grouped["cost_inr"] / grouped["area_ha"]
    grouped["year_index"] = grouped["year"] - YEAR_MIN
    return grouped


def by_dimension(
    frame: pd.DataFrame | None = None,
    dimension: str = "state",
    *,
    limit: int | None = 20,
) -> pd.DataFrame:
    """Aggregate production, area, cost and efficiency by any column."""
    data = _frame() if frame is None else frame
    if dimension not in data.columns:
        raise KeyError(f"{dimension!r} is not a column in the dataset")

    grouped = data.groupby(dimension, as_index=False).agg(
        production_quintals=("production_quintals", "sum"),
        area_ha=("quantity", "sum"),
        cost_inr=("cost", "sum"),
        observations=("production_quintals", "size"),
        mean_efficiency=("production_efficiency", "mean"),
        median_efficiency=("production_efficiency", "median"),
        mean_cost_per_ha=("cost_per_unit", "mean"),
    )
    grouped["production_tonnes"] = (
        grouped["production_quintals"] / QUINTAL_PER_TONNE
    )
    grouped["cost_per_ha"] = grouped["cost_inr"] / grouped["area_ha"]
    grouped = grouped.sort_values("production_quintals", ascending=False)
    return grouped.head(limit).reset_index(drop=True) if limit else grouped


def top_states_by(
    metric: str = "production_quintals",
    *,
    frame: pd.DataFrame | None = None,
    limit: int = 15,
    ascending: bool = False,
) -> pd.DataFrame:
    data = by_dimension(frame, "state", limit=None)
    if metric not in data.columns:
        raise KeyError(f"{metric!r} is not an aggregated metric")
    return data.sort_values(metric, ascending=ascending).head(limit).reset_index(drop=True)


def crop_mix(frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """Share of production, area and cost attributable to each crop."""
    data = _frame() if frame is None else frame
    grouped = data.groupby("crop", as_index=False).agg(
        production_quintals=("production_quintals", "sum"),
        area_ha=("quantity", "sum"),
        cost_inr=("cost", "sum"),
        observations=("production_quintals", "size"),
        mean_efficiency=("production_efficiency", "mean"),
        mean_cost_per_ha=("cost_per_unit", "mean"),
    )
    grouped["production_tonnes"] = (
        grouped["production_quintals"] / QUINTAL_PER_TONNE
    )
    grouped["production_share_pct"] = (
        100.0 * grouped["production_quintals"] / grouped["production_quintals"].sum()
    )
    grouped["area_share_pct"] = 100.0 * grouped["area_ha"] / grouped["area_ha"].sum()
    grouped["cost_share_pct"] = 100.0 * grouped["cost_inr"] / grouped["cost_inr"].sum()
    return grouped.sort_values("production_quintals", ascending=False).reset_index(drop=True)


def season_distribution(frame: pd.DataFrame | None = None) -> pd.DataFrame:
    data = _frame() if frame is None else frame
    grouped = data.groupby("season_name", as_index=False).agg(
        production_quintals=("production_quintals", "sum"),
        area_ha=("quantity", "sum"),
        cost_inr=("cost", "sum"),
        observations=("production_quintals", "size"),
        mean_efficiency=("production_efficiency", "mean"),
    )
    grouped["share_pct"] = (
        100.0 * grouped["observations"] / grouped["observations"].sum()
    )
    return grouped.sort_values("observations", ascending=False).reset_index(drop=True)


def state_crop_matrix(
    *,
    frame: pd.DataFrame | None = None,
    value: str = "median_efficiency",
) -> pd.DataFrame:
    """State x crop pivot of efficiency or cost, for the heatmap."""
    data = _frame() if frame is None else frame
    column = {
        "median_efficiency": "production_efficiency",
        "mean_efficiency": "production_efficiency",
        "median_cost": "cost_per_unit",
        "mean_cost": "cost_per_unit",
    }[value]
    pivot = data.pivot_table(
        index="state", columns="crop", values=column, aggfunc="median"
    )
    return pivot.sort_index()


def efficiency_vs_cost(
    *, frame: pd.DataFrame | None = None, dimension: str = "state"
) -> pd.DataFrame:
    """Scatter inputs: yield against cost, aggregated by ``dimension``.

    The four-quadrant reading is the point of this chart -- high yield and low
    cost is the quadrant worth moving towards.
    """
    data = _frame() if frame is None else frame
    grouped = data.groupby(dimension, as_index=False).agg(
        median_efficiency=("production_efficiency", "median"),
        median_cost_per_ha=("cost_per_unit", "median"),
        observations=("production_quintals", "size"),
    )
    grouped = grouped[np.isfinite(grouped["median_efficiency"])]
    grouped = grouped[np.isfinite(grouped["median_cost_per_ha"])]

    if grouped.empty:
        return grouped

    median_efficiency = grouped["median_efficiency"].median()
    median_cost = grouped["median_cost_per_ha"].median()
    grouped["quadrant"] = np.where(
        grouped["median_efficiency"] >= median_efficiency,
        np.where(
            grouped["median_cost_per_ha"] <= median_cost,
            "High yield / low cost",
            "High yield / high cost",
        ),
        np.where(
            grouped["median_cost_per_ha"] <= median_cost,
            "Low yield / low cost",
            "Low yield / high cost",
        ),
    )
    grouped.insert(0, "name", grouped[dimension])
    return grouped.sort_values("median_efficiency", ascending=False).reset_index(drop=True)


def correlation_matrix(
    *, frame: pd.DataFrame | None = None, columns: list[str] | None = None
) -> pd.DataFrame:
    """Correlation among the numeric columns a reader might correlate."""
    data = _frame() if frame is None else frame
    if columns is None:
        columns = [
            "quantity", "production_quintals", "production_efficiency",
            "cost", "cost_per_unit", "year",
        ]
    available = [c for c in columns if c in data.columns]
    numeric = data[available].apply(pd.to_numeric, errors="coerce")
    return numeric.corr(method="spearman").round(4)


def data_quality_summary() -> dict[str, Any]:
    """Loader provenance and cleaning counts, for the transparency panel."""
    status = registry.training_status()
    quality = status.get("data_quality") or {}
    provenance = status.get("provenance") or {}
    return {
        "synthetic": status.get("is_synthetic"),
        "rows_in": quality.get("rows_in"),
        "rows_out": quality.get("rows_out"),
        "rows_dropped": quality.get("rows_dropped_total"),
        "dropped_reasons": quality.get("dropped_rows", {}),
        "normalised_units": quality.get("normalised_units", {}),
        "notes": quality.get("notes", []),
        "source_file": provenance.get("path"),
        "sha256": provenance.get("sha256"),
    }


def distribution_summary(
    column: str = "production_efficiency", *, frame: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Quantile summary of a numeric column, for the skewness callout."""
    data = _frame() if frame is None else frame
    series = pd.to_numeric(data[column], errors="coerce").dropna()
    if series.empty:
        return pd.DataFrame()
    quantiles = series.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    rows = [{"statistic": "mean", "value": float(series.mean())},
            {"statistic": "std", "value": float(series.std())},
            {"statistic": "skew", "value": float(series.skew())}]
    for quantile, value in quantiles.items():
        rows.append({"statistic": f"p{quantile * 100:g}", "value": float(value)})
    return pd.DataFrame(rows)


def outlier_report(
    *, frame: pd.DataFrame | None = None, top: int = 10
) -> pd.DataFrame:
    """Rows the robust within-crop z-score flagged as likely entry errors."""
    data = registry.load_dataset()
    if "is_outlier" not in data.columns:
        return pd.DataFrame()
    flagged = data.loc[data["is_outlier"]].copy()
    if flagged.empty:
        return flagged
    columns = [
        c for c in ("Crop", "state", "year", "Quantity", "production",
                    "production_efficiency", "cost", "cost_per_unit")
        if c in flagged.columns
    ]
    return flagged.sort_values("production_efficiency", ascending=False)[
        columns
    ].head(top).reset_index(drop=True)
