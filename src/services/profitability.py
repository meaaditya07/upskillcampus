"""Profitability simulation: what a forecast is worth in rupees.

Keeps every assumption explicit and visible, because the honest answer to "what
will I earn" depends far more on the price assumption than on the yield model.
The price catalogue is indicative, not an official MSP series, so results are
labelled as estimates everywhere and the vintage adjustment is reported
alongside the number.

Break-even analysis is included because for a farmer the useful question is
often not "what is the profit" but "how wrong can the forecast be before I lose
money".

Streamlit-free by design.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config.settings import (
    INDICATIVE_PRICE_PER_QUINTAL,
    PRICE_ASSUMPTION_NOTE,
    QUINTAL_PER_TONNE,
    YEAR_MAX,
    YEAR_MIN,
    indicative_price,
)

#: A forecast's interval is treated as fatal only once profit goes negative.
MARGIN_OF_SAFETY_FLOOR = 0.0


@dataclass
class ProfitResult:
    """Economics of one or more forecasts."""

    frame: pd.DataFrame
    price_per_quintal: np.ndarray
    revenue: np.ndarray
    cost: np.ndarray
    profit: np.ndarray
    margin_pct: np.ndarray
    break_even_price: np.ndarray
    break_even_yield: np.ndarray
    revenue_low: np.ndarray
    revenue_high: np.ndarray
    profit_low: np.ndarray
    profit_high: np.ndarray
    breakeven_risk: np.ndarray

    def __len__(self) -> int:
        return len(self.frame)

    def first(self) -> dict[str, float]:
        return {
            "price_per_quintal": float(self.price_per_quintal[0]),
            "revenue": float(self.revenue[0]),
            "cost": float(self.cost[0]),
            "profit": float(self.profit[0]),
            "margin_pct": float(self.margin_pct[0]),
            "break_even_price": float(self.break_even_price[0]),
            "break_even_yield": float(self.break_even_yield[0]),
            "profit_low": float(self.profit_low[0]),
            "profit_high": float(self.profit_high[0]),
            "breakeven_risk": float(self.breakeven_risk[0]),
        }


def _safe_div(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    denom = np.where(np.abs(denominator) < 1e-9, np.nan, denominator)
    return np.divide(numerator, denom, out=np.full_like(numerator, np.nan, dtype=float),
                     where=~np.isnan(denom))


def assess(
    forecast,
    *,
    price_override: float | None = None,
    cost_override: float | None = None,
    price_shift_pct: float = 0.0,
) -> ProfitResult:
    """Turn a :class:`~src.services.predictor.Forecast` into rupees.

    Parameters
    ----------
    price_override:
        Absolute INR/quintal, bypassing the indicative catalogue. Lets the user
        price a scenario at their own local rate.
    price_shift_pct:
        Percentage adjustment applied on top of the catalogue price, for
        "what if the market is 10% weaker" style sensitivity.
    cost_override:
        Absolute INR total, for a farmer who knows their own input bill.
    """
    rows = forecast.rows.copy()
    crop = rows["crop"].astype(str)
    years = pd.to_numeric(rows.get("year", YEAR_MAX), errors="coerce").fillna(YEAR_MAX)

    base_prices = np.array(
        [indicative_price(c, y)[0] for c, y in zip(crop, years, strict=True)],
        dtype=float,
    )
    prices = base_prices * (1.0 + float(price_shift_pct) / 100.0)
    if price_override is not None:
        prices = np.full(len(rows), float(price_override), dtype=float)

    costs = np.asarray(forecast.cost_point, dtype=float)
    if cost_override is not None:
        costs = np.full(len(rows), float(cost_override), dtype=float)

    yields = np.asarray(forecast.yield_point, dtype=float)
    y_low = np.asarray(forecast.yield_lower, dtype=float)
    y_high = np.asarray(forecast.yield_upper, dtype=float)

    revenue = yields * prices
    profit = revenue - costs

    with np.errstate(divide="ignore", invalid="ignore"):
        margin = np.where(revenue > 0, profit / revenue * 100.0, np.nan)

        # The price at which the *point* forecast exactly covers its costs.
        break_even_price = _safe_div(costs, yields)
        # The yield at which the point price exactly covers costs.
        break_even_yield = _safe_div(costs, prices)

    profit_low = y_low * prices - costs
    profit_high = y_high * prices - costs

    # Probability of ending up below break-even, implied by the interval.
    # Treating the yield range as uniform gives P(Y < threshold) =
    # (threshold - lower) / (upper - lower), clipped to [0, 1]. It turns the
    # interval into an actionable risk flag instead of a decorative band.
    span = np.where(y_high > y_low, y_high - y_low, np.nan)
    loss_threshold = costs / np.where(prices > 0, prices, np.nan)
    with np.errstate(invalid="ignore"):
        risk = np.clip(
            np.divide(
                loss_threshold - y_low,
                span,
                out=np.full_like(span, np.nan, dtype=float),
                where=~np.isnan(span),
            ),
            0.0,
            1.0,
        )

    rows = rows.assign(
        price_per_quintal=prices,
        indicative_price_per_quintal=base_prices,
        revenue=revenue,
        total_cost=costs,
        profit=profit,
        margin_pct=margin,
        break_even_price=break_even_price,
        break_even_yield=break_even_yield,
        profit_low=profit_low,
        profit_high=profit_high,
        breakeven_risk=risk,
        tonnes=yields / QUINTAL_PER_TONNE,
    )

    return ProfitResult(
        frame=rows,
        price_per_quintal=prices,
        revenue=revenue,
        cost=costs,
        profit=profit,
        margin_pct=margin,
        break_even_price=break_even_price,
        break_even_yield=break_even_yield,
        revenue_low=y_low * prices,
        revenue_high=y_high * prices,
        profit_low=profit_low,
        profit_high=profit_high,
        breakeven_risk=risk,
    )


def price_sensitivity(
    forecast,
    *,
    shifts: tuple[float, ...] = (-20.0, -10.0, 0.0, 10.0, 20.0),
) -> pd.DataFrame:
    """Profit across a grid of price shifts, for a tornado-style chart."""
    rows = []
    for shift in shifts:
        result = assess(forecast, price_shift_pct=shift)
        for index in range(len(result)):
            rows.append(
                {
                    "scenario": index,
                    "shift_pct": shift,
                    "price_per_quintal": float(result.price_per_quintal[index]),
                    "profit": float(result.profit[index]),
                    "revenue": float(result.revenue[index]),
                    "margin_pct": float(result.margin_pct[index]),
                }
            )
    return pd.DataFrame(rows)


def area_sensitivity(
    forecast_factory,
    *,
    areas=(0.5, 1.0, 2.0, 5.0),
) -> pd.DataFrame:
    """Profit across cultivated-area scenarios.

    ``forecast_factory`` is a callable taking ``area_ha`` and returning a
    Forecast, so the caller controls which other scenario fields stay fixed.
    Economies of scale in cultivation cost are the whole point of the sweep.
    """
    rows = []
    for area in areas:
        result = assess(forecast_factory(area))
        for index in range(len(result)):
            rows.append(
                {
                    "area_ha": area,
                    "profit": float(result.profit[index]),
                    "revenue": float(result.revenue[index]),
                    "total_cost": float(result.cost[index]),
                    "cost_per_hectare": float(result.cost[index]) / area,
                    "margin_pct": float(result.margin_pct[index]),
                }
            )
    return pd.DataFrame(rows)


def assumptions() -> dict:
    """The price assumptions, for display next to any profit figure."""
    return {
        "note": PRICE_ASSUMPTION_NOTE,
        "vintage_window": (YEAR_MIN, YEAR_MAX),
        "unit": "INR per quintal",
        "catalogue": dict(sorted(INDICATIVE_PRICE_PER_QUINTAL.items())),
    }
