"""Yield & cost forecast for a single field, with profitability."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components import charts, kpi_cards, sidebar, theme
from config.settings import PRICE_ASSUMPTION_NOTE, QUINTAL_PER_TONNE
from src.services import predictor, profitability, registry


def render() -> None:
    sidebar.sidebar("Yield & cost forecast")

    st.markdown("## Yield & cost forecast")
    st.caption(
        "Describe one field. The production and cost models are fitted on the "
        "same inputs, so the revenue and profit below describe that same field "
        "rather than two unrelated numbers."
    )

    if not registry.artifacts_ready():
        st.error("Model artefacts are not built. Run `python scripts/bootstrap.py`.")
        return

    bundles = registry.load_bundles()

    left, right = st.columns([1, 1], gap="large")
    with left:
        theme.section("Field")
        states = predictor.known_states()
        state = st.selectbox(
            "State", states,
            index=states.index("Punjab") if "Punjab" in states else 0,
        )
    with right:
        theme.section("Season and area")
        low, high = predictor.year_range()
        year = st.slider("Year", int(low), int(high), int(high))
        area = st.number_input(
            "Cultivated area (hectares)", min_value=0.01, max_value=500.0,
            value=2.0, step=0.25,
        )

    crop = st.selectbox("Crop", predictor.crops())
    zones = predictor.zones_for(state) or [f"State - {state}"]
    zone = st.selectbox("Recommended zone", zones)
    variety = st.selectbox("Variety", predictor.varieties_for(crop))
    seasons = predictor.seasons_for_crop(crop) or ["Kharif"]
    season = st.selectbox("Season", seasons)

    scenario = predictor.Scenario(
        state=state, crop=crop, variety=variety, recommended_zone=zone,
        season=season, area_ha=area, year=year,
    )

    st.divider()
    forecast = predictor.forecast([scenario], bundles=bundles)
    numbers = forecast.first()
    profit = profitability.assess(forecast)
    economics = profit.first()
    per_ha = float(forecast.per_hectare()[0])

    kpi_cards.row(
        [
            (
                "Production",
                f"{numbers['yield_quintals']:,.0f} q",
                (
                    f"{numbers['yield_tonnes']:,.1f} t "
                    f"({per_ha:,.1f} q/ha)"
                ),
            ),
            (
                "80% range",
                f"{numbers['yield_lower']:,.0f} - {numbers['yield_upper']:,.0f} q",
                "calibrated on held-out data",
            ),
            (
                "Cultivation cost",
                kpi_cards.rupees(numbers["cost"]),
                (
                    f"range {kpi_cards.rupees(numbers['cost_lower'])} - "
                    f"{kpi_cards.rupees(numbers['cost_upper'])}"
                ),
            ),
            (
                "Net profit",
                kpi_cards.rupees(economics["profit"]),
                f"margin {economics['margin_pct']:.1f}%",
            ),
        ]
    )

    st.divider()
    left, right = st.columns([3, 2])
    with left:
        theme.section("Production forecast")
        st.plotly_chart(
            charts.forecast_interval(
                pd.DataFrame(
                    {
                        "Scenario": [f"{crop} - {zone}"],
                        "point": [numbers["yield_quintals"]],
                        "lower": [numbers["yield_lower"]],
                        "upper": [numbers["yield_upper"]],
                    }
                ),
                "Scenario",
                value_label="Production (quintals)",
            ),
            width="stretch",
        )
    with right:
        theme.section("Economics")
        st.plotly_chart(
            charts.profit_waterfall(economics), width="stretch"
        )

    theme.note(
        f"Break-even price ₹{economics['break_even_price']:,.0f} per quintal "
        f"against an assumed ₹{economics['price_per_quintal']:,.0f}. "
        f"Break-even yield {economics['break_even_yield']:,.0f} quintals on "
        f"{area:,.2f} ha."
    )
    theme.note(PRICE_ASSUMPTION_NOTE, kind="warning")

    st.divider()
    left, right = st.columns(2)
    with left:
        theme.section("Price sensitivity")
        sweep = profitability.price_sensitivity(forecast)
        st.plotly_chart(
            charts.sensitivity_line(sweep, "price_per_quintal", "profit",
                                    "Profit against price per quintal"),
            width="stretch",
        )
        st.dataframe(sweep, width="stretch", hide_index=True)
    with right:
        theme.section("Area sensitivity")
        area_sweep = profitability.area_sensitivity(
            lambda ha: predictor.forecast(
                [predictor.Scenario(
                    state=state, crop=crop, variety=variety,
                    recommended_zone=zone, season=season,
                    area_ha=ha, year=year,
                )],
                bundles=bundles,
            )
        )
        st.plotly_chart(
            charts.sensitivity_line(area_sweep, "area_ha", "profit",
                                    "Profit against cultivated area"),
            width="stretch",
        )
        st.dataframe(area_sweep, width="stretch", hide_index=True)

    st.divider()
    theme.section("What drove this prediction")
    st.caption(
        "Additive contributions from the boosted trees. Green pushed the "
        "forecast up, red pushed it down; the sum plus the model's base value "
        "is the point estimate."
    )
    contributions = predictor.explain(scenario, top=12)
    if contributions.empty:
        theme.note(
            f"Explanations unavailable ({contributions.attrs.get('error', 'unknown')}).",
            kind="warning",
        )
    else:
        st.plotly_chart(
            charts.contribution_waterfall(contributions), width="stretch"
        )
        st.dataframe(contributions, width="stretch", hide_index=True)

    st.divider()
    with st.expander("How this number is produced"):
        st.markdown(
            f"""
- **Production** is the point estimate from an XGBoost regressor fitted on
  `log1p(production_quintals)`; the displayed range comes from two
  `reg:quantileerror` models at the 10th and 90th percentiles, widened by a
  factor calibrated on a held-out calibration split.
- **Cost** is a second, independent model on the same inputs. The two models
  never see `production` or each other's target, so the yield and cost figures
  are not jointly biased.
- **Area is linear.** Cultivation cost scales with hectares and the model's
  per-hectare cost is applied to your area, so doubling the hectares roughly
  doubles the cost.
- **Quintals, not kilograms.** One quintal is 50 kg; 20 quintals make a tonne.
  Production is reported in quintals and tonnes, never mixed.
- **One Indian quintal = {QUINTAL_PER_TONNE:.0f} per tonne** by definition,
  not by assumption about the data.
"""
        )
