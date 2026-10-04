"""Overview: what the dataset says before any modelling."""

from __future__ import annotations

import streamlit as st

from components import charts, kpi_cards, sidebar, theme
from src.services import insights, registry


def render() -> None:
    filters = sidebar.sidebar("Overview")

    st.markdown("## Overview")
    st.caption(
        "National crop production, area and cost across states, crops and "
        "seasons. Every figure below excludes rows the pipeline flagged as "
        "likely entry errors."
    )

    dataset = registry.load_dataset()
    frame = sidebar.apply_filters(dataset, filters)

    if frame.empty:
        st.warning("No rows match the current filters.")
        return

    kpi_cards.kpi_strip(frame)
    st.divider()

    left, right = st.columns([3, 2])
    with left:
        theme.section("Production over time")
        metric = st.selectbox(
            "Metric", ["production_quintals", "quantity", "cost"],
            format_func=lambda c: {
                "production_quintals": "Total production (quintals)",
                "quantity": "Area cultivated (hectares)",
                "cost": "Total cost (INR)",
            }[c],
            label_visibility="collapsed",
        )
        st.plotly_chart(
            charts.production_trend(frame, metric), width="stretch"
        )
    with right:
        theme.section("Crop mix")
        st.plotly_chart(charts.crop_mix(insights.crop_mix(frame)),
                        width="stretch")

    st.divider()

    geojson, note = registry.load_geojson()
    if geojson is None:
        theme.note(f"Basemap unavailable: {note}")
    else:
        value = st.radio(
            "Shade states by",
            ["production_quintals", "median_efficiency", "mean_cost_per_ha"],
            horizontal=True,
            format_func=lambda c: {
                "production_quintals": "Total production",
                "median_efficiency": "Median efficiency (q/ha)",
                "mean_cost_per_ha": "Mean cost per hectare",
            }[c],
        )
        aggregated = insights.by_dimension(frame, "state", limit=None)
        st.plotly_chart(
            charts.state_choropleth(aggregated, geojson, value),
            width="stretch",
        )
        coverage = _coverage(aggregated, geojson)
        theme.note(
            f"{coverage['n_matched']}/{coverage['n_data_states']} dataset states "
            f"matched to the basemap ({coverage['coverage_pct']:.0f}% coverage)."
            + (f" Unmatched: {', '.join(coverage['unmatched'])}"
               if coverage["unmatched"] else ""),
            kind="ok" if not coverage["unmatched"] else "warning",
        )

    st.divider()

    left, right = st.columns(2)
    with left:
        theme.section("Leading states")
        st.plotly_chart(
            charts.horizontal_bars(
                insights.top_states_by("production_quintals", frame=frame, limit=15),
                "state", "production_quintals", "Total production (quintals)",
            ),
            width="stretch",
        )
    with right:
        theme.section("Season split")
        st.plotly_chart(charts.season_split(insights.season_distribution(frame)),
                        width="stretch")

    st.divider()
    theme.section("The core trade-off")
    st.caption(
        "Cost per hectare against output per hectare. The efficient, cheap "
        "upper-left corner is where a recommendation engine should point."
    )
    st.plotly_chart(charts.scatter_efficiency_cost(frame), width="stretch")

    st.divider()
    with st.expander("Data tables"):
        tab1, tab2, tab3 = st.tabs(["By state", "By crop", "Season x crop"])
        with tab1:
            st.dataframe(insights.by_dimension(frame, "state", limit=None),
                         width="stretch")
        with tab2:
            st.dataframe(insights.crop_mix(frame), width="stretch")
        with tab3:
            st.dataframe(insights.state_crop_matrix(frame=frame), width="stretch")


def _coverage(aggregated, geojson) -> dict:
    from src import geo

    return geo.coverage(aggregated, geojson)
