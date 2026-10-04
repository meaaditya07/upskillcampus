"""Recommendations: where and what to plant, and how confident we are."""

from __future__ import annotations

import streamlit as st

from components import charts, kpi_cards, sidebar, theme
from config.settings import DEFAULT_SCORE_WEIGHTS, PRICE_ASSUMPTION_NOTE
from src.services import predictor, recommend, registry

SCORE_COLUMNS = [
    "rank", "score", "confidence", "n_observations", "n_years",
    "median_efficiency", "median_cost_per_ha", "yield_cv",
]


def _confidence_badge(value: str) -> str:
    return {"ok": "well evidenced", "low": "thin evidence"}.get(str(value), str(value))


def render() -> None:
    filters = sidebar.sidebar("Recommendations")

    st.markdown("## Recommendations")
    st.caption(
        "Rankings are built from recency-weighted medians within each group, "
        "then scored on efficiency, cost and reliability. Groups with too few "
        "observations are demoted rather than hidden."
    )

    dataset = registry.load_dataset()
    frame = sidebar.apply_filters(dataset, filters)

    left, right = st.columns([1, 1], gap="large")
    with left:
        theme.section("What are you growing?")
        crops = predictor.crops()
        crop = st.selectbox(
            "Crop", crops, index=crops.index("Wheat") if "Wheat" in crops else 0
        )
    with right:
        theme.section("Where?")
        states = predictor.known_states()
        scope_state = st.selectbox(
            "Rank within a state", ["All states", *states],
            index=states.index("Punjab") + 1 if "Punjab" in states else 0,
            help="Choosing a state ranks its zones instead of the whole country.",
        )
        limit = st.slider("Show top N", 3, 25, 10)

    state_arg = None if scope_state == "All states" else scope_state
    result = recommend.recommend_for_crop(
        crop, frame=frame, state=state_arg, limit=limit
    )

    if result.table.empty:
        st.warning(
            f"No observations for {crop}"
            + (f" in {scope_state}" if state_arg else "")
            + " under the current filters."
        )
        return

    theme.note(
        f"Ranking **{result.scope}** level from "
        f"{result.n_observations:,} usable observations. "
        f"Weights: {DEFAULT_SCORE_WEIGHTS['efficiency']:.0%} efficiency, "
        f"{DEFAULT_SCORE_WEIGHTS['cost']:.0%} cost, "
        f"{DEFAULT_SCORE_WEIGHTS['reliability']:.0%} reliability.",
        kind="ok",
    )

    best = result.table.iloc[0]
    label_column = "state" if result.scope == "state" else "recommended_zone"
    kpi_cards.row(
        [
            ("Best option", str(best[label_column]),
             f"score {best['score']:.3f}"),
            ("Median output", f"{best['median_efficiency']:,.1f} q/ha",
             f"from {int(best['n_observations']):,} observations"),
            ("Median cost", kpi_cards.rupees(float(best["median_cost_per_ha"])),
             "per hectare"),
            ("Reliability", _confidence_badge(best["confidence"]),
             f"yield CV {float(best['yield_cv']):.2f}"
             if pd_notna(best.get("yield_cv")) else ""),
        ]
    )

    st.divider()
    theme.section("Ranking")
    label = "State / UT" if result.scope == "state" else "Zone"
    st.plotly_chart(
        charts.horizontal_bars(
            result.table, label_column, "score",
            f"{crop}: composite score by {result.scope}", height=max(360, 26 * len(result.table)),
        ),
        width="stretch",
    )

    shown = [c for c in SCORE_COLUMNS if c in result.table.columns]
    table = result.table[[label_column] + [c for c in shown if c != "rank"]].rename(
        columns={label_column: label}
    )
    st.dataframe(
        table, width="stretch", hide_index=True,
        column_config={
            "score": st.column_config.ProgressColumn(
                "Score", min_value=0.0, max_value=1.0, format="%.3f"
            ),
            "median_efficiency": st.column_config.NumberColumn("q/ha", format="%.1f"),
            "median_cost_per_ha": st.column_config.NumberColumn("Cost/ha", format="%.0f"),
            "yield_cv": st.column_config.NumberColumn("Yield CV", format="%.3f"),
        },
    )

    st.divider()
    left, right = st.columns(2)
    with left:
        theme.section("Efficiency against cost")
        st.caption("The upper-left corner is cheap and productive.")
        columns = ["median_cost_per_ha", "median_efficiency", label_column, "n_observations"]
        st.plotly_chart(
            charts.horizontal_bars(
                result.table[[c for c in columns if c in result.table.columns]],
                label_column, "median_cost_per_ha",
                "Median cost per hectare", height=max(360, 26 * len(result.table)),
            ),
            width="stretch",
        )
    with right:
        theme.section("Evidence behind each rank")
        st.caption(
            "A high score on thin evidence is a coincidence, not a finding. "
            "Thin groups are pushed down the list automatically."
        )
        st.plotly_chart(
            charts.horizontal_bars(
                result.table, label_column, "n_observations",
                "Observations per option", height=max(360, 26 * len(result.table)),
            ),
            width="stretch",
        )

    if state_arg:
        st.divider()
        theme.section(f"Within {state_arg}")
        inner_left, inner_right = st.columns(2)
        with inner_left:
            seasons = recommend.recommend_seasons(state_arg, crop, frame=frame)
            st.markdown("**Which season**")
            if not seasons.table.empty:
                st.dataframe(
                    seasons.table, width="stretch", hide_index=True
                )
            else:
                st.caption("No season evidence for this combination.")
        with inner_right:
            varieties = recommend.recommend_varieties(state_arg, crop, frame=frame)
            st.markdown("**Which variety**")
            if not varieties.table.empty:
                st.dataframe(
                    varieties.table, width="stretch", hide_index=True
                )
            else:
                st.caption("No variety evidence for this combination.")

    st.divider()
    with st.expander("How the score is built"):
        st.markdown(
            f"""
Each option is reduced to three numbers before scoring:

1. **Median efficiency** in quintals per hectare, weighted so recent years
   count for more than old ones. A median rather than a mean, so one bad
   season cannot dominate.
2. **Median cost per hectare**, same weighting, scored so lower is better.
3. **Reliability**, from the robust coefficient of variation of that option's
   own yields. A crop that swings wildly is penalised even when its average
   looks good.

These are min-max normalised across the options being compared and combined
as **{DEFAULT_SCORE_WEIGHTS['efficiency']:.0%} efficiency,
{DEFAULT_SCORE_WEIGHTS['cost']:.0%} cost,
{DEFAULT_SCORE_WEIGHTS['reliability']:.0%} reliability**.

Options with fewer than 30 observations or fewer than 3 distinct years are
marked **low** confidence and lose 0.5 of score, so a single lucky season
cannot top the list.

Normalisation is relative to the options on screen: this ranks the candidates
against each other, it does not claim an absolute score.
"""
        )

    st.divider()
    theme.section("What it would earn")
    priced = recommend.rank_with_profit(crop, frame=frame, limit=limit).table
    if not priced.empty:
        st.dataframe(priced, width="stretch", hide_index=True)
        theme.note(PRICE_ASSUMPTION_NOTE, kind="warning")


def pd_notna(value) -> bool:
    try:
        import pandas as pd

        return bool(pd.notna(value))
    except Exception:
        return False
