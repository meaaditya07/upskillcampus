"""Chart builders.

Every function returns a Plotly figure with the theme already applied, so the
pages never hand-roll colours.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from components import theme
from config.settings import ACCENT

EARTH = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min/plotly-min.js"


def _fin(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(**theme.plotly_layout(height=height))
    fig.update_traces(hoverlabel={"font": {"size": 12}})
    return fig


def _empty(message: str, height: int = 260) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message, showarrow=False, xref="paper", yref="paper",
        x=0.5, y=0.5, font={"color": theme.active_palette()["muted"], "size": 14},
    )
    fig.update_layout(**theme.plotly_layout(height=height))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


# --------------------------------------------------------------------------
# Trend
# --------------------------------------------------------------------------
def _as_frame(data) -> pd.DataFrame:
    """Accept a DataFrame, a list of records from ``metrics.json``, or None.

    ``metrics.json`` stores tabular blocks as lists of dicts, so chart helpers
    that are fed straight from it need to normalise before touching ``.empty``.
    """
    if data is None:
        return pd.DataFrame()
    if isinstance(data, pd.DataFrame):
        return data
    return pd.DataFrame(list(data))


def production_trend(frame: pd.DataFrame, metric: str = "production_quintals") -> go.Figure:
    """National production (or area/cost) by year."""
    if frame is None or frame.empty:
        return _empty("No data available")
    series = frame.groupby("year", as_index=False)[metric].sum().sort_values("year")
    fig = px.area(series, x="year", y=metric, color_discrete_sequence=[ACCENT])
    fig.update_traces(line={"width": 2.5})
    return _fin(fig, 340)


def yearly_with_range(frame: pd.DataFrame, group: str) -> go.Figure:
    """Median efficiency per group with an interquartile band."""
    if frame is None or frame.empty:
        return _empty("No data available")
    groups = list(frame[group].dropna().unique())
    colours = theme.categorical_scale(max(len(groups), 1))
    fig = go.Figure()
    for index, key in enumerate(groups):
        colour = colours[index % len(colours)]
        grouped = frame.loc[frame[group] == key].groupby("year")[
            "production_efficiency"
        ]
        med, q1, q3 = grouped.median(), grouped.quantile(0.25), grouped.quantile(0.75)
        # Band first (lower trace then upper with tonexty) so it sits under the
        # median line rather than over it.
        fig.add_trace(
            go.Scatter(x=med.index, y=q1, mode="lines", line={"width": 0},
                       showlegend=False, hoverinfo="skip", legendgroup=str(key))
        )
        fig.add_trace(
            go.Scatter(x=med.index, y=q3, mode="lines", line={"width": 0},
                       fill="tonexty", fillcolor=_alpha(colour, 0.16),
                       showlegend=False, hoverinfo="skip", legendgroup=str(key))
        )
        fig.add_trace(
            go.Scatter(x=med.index, y=med, mode="lines", name=str(key),
                       line={"color": colour, "width": 2}, legendgroup=str(key))
        )
    fig.update_layout(
        xaxis_title="Year", yaxis_title="Median quintals per hectare"
    )
    return _fin(fig, 380)


def _alpha(colour: str, alpha: float) -> str:
    colour = colour.lstrip("#")
    r, g, b = (int(colour[i: i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------
def horizontal_bars(
    frame: pd.DataFrame, label: str, value: str, title: str = "",
    height: int = 420,
) -> go.Figure:
    """Ranked horizontal bars, shaded by the value itself."""
    if frame is None or frame.empty:
        return _empty("No data available", height)
    fig = px.bar(
        frame.sort_values(value, ascending=True),
        x=value, y=label, orientation="h",
        color=value, color_continuous_scale=theme.sequential_scale(),
        labels={label: "", value: ""},
        title=title,
    )
    fig.update_layout(coloraxis_showscale=False)
    return _fin(fig, height)


def crop_mix(frame: pd.DataFrame) -> go.Figure:
    """Share of production per crop."""
    if frame is None or frame.empty:
        return _empty("No data available")
    fig = px.pie(
        frame, names="crop", values="production_quintals",
        color_discrete_sequence=theme.categorical_scale(len(frame)),
        hole=0.45,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return _fin(fig, 380)


def season_split(frame: pd.DataFrame) -> go.Figure:
    """Total production by season."""
    if frame is None or frame.empty:
        return _empty("No data available")
    fig = px.bar(
        frame, x="season_name", y="production_quintals",
        color="season_name",
        color_discrete_sequence=theme.categorical_scale(len(frame)),
        labels={"season_name": "", "production_quintals": "Quintals"},
    )
    fig.update_layout(showlegend=False)
    return _fin(fig, 300)


# --------------------------------------------------------------------------
# Geography
# --------------------------------------------------------------------------
def state_choropleth(
    frame: pd.DataFrame, geojson: dict, value: str = "production_quintals",
    title: str = "",
) -> go.Figure:
    """Per-state metric. Values are keyed by canonical state name."""
    if frame is None or frame.empty or geojson is None:
        return _empty("Basemap unavailable", 460)
    fig = px.choropleth(
        frame, geojson=geojson, locations="state", color=value,
        color_continuous_scale=theme.sequential_scale(),
        labels={value: ""},
        title=title,
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin={"l": 0, "r": 0, "t": 40, "b": 0})
    return _fin(fig, 470)


# --------------------------------------------------------------------------
# Model output
# --------------------------------------------------------------------------
def forecast_interval(
    frame: pd.DataFrame, label: str,
    point: str = "point", lower: str = "lower", upper: str = "upper",
    value_label: str = "",
    height: int = 240,
) -> go.Figure:
    """Point forecast with its calibrated interval, one trace per scenario."""
    if frame is None or frame.empty:
        return _empty("No forecast yet", height)
    fig = go.Figure()
    for index, (_, row) in enumerate(frame.iterrows()):
        colour = theme.categorical_scale(len(frame))[index % 64]
        fig.add_trace(
            go.Scatter(
                x=[row[point], row[point]], y=[row[lower], row[upper]],
                mode="lines", line={"color": colour, "width": 8},
                opacity=0.35, showlegend=False, hoverinfo="skip",
                name=str(row.get(label, index)),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[row[point]], y=[row[point]], mode="markers",
                marker={"color": colour, "size": 11, "line": {"width": 2,
                                                              "color": theme.active_palette()["paper"]}},
                name=str(row.get(label, index)),
                hovertemplate=(
                    f"{value_label}: %{{x:,.1f}}<br>"
                    f"Range: %{{customdata[0]:,.1f}} - %{{customdata[1]:,.1f}}"
                    "<extra></extra>"
                ),
                customdata=[[row[lower], row[upper]]],
            )
        )
    fig.update_layout(xaxis_title=value_label, showlegend=len(frame) <= 12)
    return _fin(fig, height)


def actual_vs_predicted(
    cache: pd.DataFrame, target: str, log: bool = True, height: int = 380
) -> go.Figure:
    """Held-out predictions against reality, on the log scale if it fits better."""
    if cache is None or cache.empty or f"{target}_pred" not in cache.columns:
        return _empty("No held-out predictions cached", height)
    cols = [f"{target}_actual", f"{target}_pred"]
    cols = [c for c in cols if c in cache.columns]
    sample = cache[cols].dropna()
    if sample.empty:
        return _empty("No held-out predictions cached", height)
    fig = px.scatter(
        sample, x=f"{target}_actual", y=f"{target}_pred",
        labels={f"{target}_actual": "Actual", f"{target}_pred": "Predicted"},
        opacity=0.35,
    )
    lo = float(min(sample[cols].min()))
    hi = float(max(sample[cols].max()))
    fig.add_trace(
        go.Scatter(
            x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect prediction",
            line={"dash": "dash", "color": theme.active_palette()["muted"]},
        )
    )
    if log and lo > 0:
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")
    return _fin(fig, height)


def residual_histogram(cache: pd.DataFrame, target: str, height: int = 300) -> go.Figure:
    """Held-out residuals, with zero marked."""
    column = f"{target}_residual"
    if cache is None or cache.empty or column not in cache.columns:
        return _empty("No residuals cached", height)
    series = cache[column].dropna()
    fig = px.histogram(
        series, nbins=60, labels={column: "Residual (actual - predicted)"},
    )
    fig.add_vline(x=0.0, line={"dash": "dash",
                                "color": theme.active_palette()["muted"]})
    return _fin(fig, height)


def calibration_curve(cache: pd.DataFrame, target: str, height: int = 340) -> go.Figure:
    """Empirical coverage by nominal level."""
    if cache is None or cache.empty:
        return _empty("No calibration data", height)
    lower = f"{target}_lower"
    upper = f"{target}_upper"
    actual = f"{target}_actual"
    if not all(c in cache.columns for c in (lower, upper, actual)):
        return _empty("No calibration data", height)
    sample = cache[[lower, upper, actual]].dropna()
    if sample.empty:
        return _empty("No calibration data", height)
    inside = (sample[actual] >= sample[lower]) & (sample[actual] <= sample[upper])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0.8], y=[float(inside.mean()) * 100], mode="markers+text",
            marker={"size": 16, "color": "#f59e0b"},
            text=[f"{inside.mean() * 100:.1f}%"],
            textposition="bottom center",
            name="Empirical coverage",
            hovertemplate="%{y:.1f}% of held-out rows inside<extra></extra>",
        )
    )
    fig.add_hline(y=80.0, line={"dash": "dash",
                                "color": theme.active_palette()["muted"]},
                  annotation_text="nominal 80%")
    fig.update_yaxes(title="Coverage (%)", range=[0, 100])
    fig.update_xaxes(visible=False)
    fig.update_layout(showlegend=False)
    return _fin(fig, height)


def feature_importance(
    importance, top: int = 14, height: int = 400
) -> go.Figure:
    """Which feature blocks the model leaned on, largest first."""
    frame = _as_frame(importance)
    if frame.empty:
        return _empty("No importances recorded", height)
    columns = [c for c in ("feature", "importance") if c in frame.columns]
    subset = frame[columns].copy()
    if "display" in frame.columns:
        subset["display"] = frame["display"]
        subset = subset.sort_values("importance").tail(top)
        label = "display"
    else:
        subset = subset.sort_values("importance").tail(top)
        label = "feature"
    fig = px.bar(
        subset, x="importance", y=label, orientation="h",
        color="importance", color_continuous_scale=theme.sequential_scale(),
    )
    fig.update_layout(coloraxis_showscale=False)
    return _fin(fig, height)


def contribution_waterfall(frame: pd.DataFrame, height: int = 340) -> go.Figure:
    """Additive feature contributions for one forecast."""
    if frame is None or frame.empty:
        return _empty("No explanation available", height)
    subset = frame.head(12).sort_values("contribution")
    colours = [
        "#22c55e" if value >= 0 else "#ef4444" for value in subset["contribution"]
    ]
    fig = go.Figure(
        go.Bar(
            x=subset["contribution"], y=subset["feature"], orientation="h",
            marker={"color": colours},
        )
    )
    fig.add_vline(x=0.0, line={"color": theme.active_palette()["muted"]})
    fig.update_layout(
        xaxis_title="Contribution to the prediction (quintals)",
        yaxis_title="",
    )
    return _fin(fig, height)


def profit_waterfall(result: dict[str, float], height: int = 300) -> go.Figure:
    """Revenue, cost and profit for one scenario."""
    revenue = float(result.get("revenue", 0.0))
    cost = float(result.get("cost", 0.0))
    profit = float(result.get("profit", 0.0))
    fig = go.Figure(
        go.Waterfall(
            orientation="v",
            measure=["relative", "relative", "total"],
            x=["Revenue", "Cost", "Net profit"],
            y=[revenue, -cost, profit],
            increasing={"marker": {"color": "#22c55e"}},
            decreasing={"marker": {"color": "#ef4444"}},
            totals={"marker": {"color": "#f59e0b"}},
        )
    )
    fig.update_layout(yaxis_title="INR", showlegend=False, xaxis_title="")
    return _fin(fig, height)


def sensitivity_line(frame: pd.DataFrame, x: str, y: str, title: str = "",
                     height: int = 300) -> go.Figure:
    """One line chart over a price or area sweep."""
    if frame is None or frame.empty:
        return _empty("No sensitivity data", height)
    fig = px.line(
        frame, x=x, y=y, markers=True,
        labels={x: "", y: "Net profit (INR)"},
    )
    fig.update_traces(line={"width": 2.5})
    fig.update_layout(title=title)
    return _fin(fig, height)


def scatter_efficiency_cost(frame: pd.DataFrame, height: int = 420) -> go.Figure:
    """The core trade-off: efficiency against cost, sized by area."""
    if frame is None or frame.empty:
        return _empty("No data available", height)
    fig = px.scatter(
        frame, x="cost_per_unit", y="production_efficiency",
        color="crop", size="quantity", hover_name="state",
        labels={"cost_per_unit": "Cost per hectare (INR)",
                "production_efficiency": "Quintals per hectare",
                "quantity": "Hectares"},
        color_discrete_sequence=theme.categorical_scale(frame["crop"].nunique()),
    )
    fig.update_traces(marker={"opacity": 0.55, "line": {"width": 0.5}})
    return _fin(fig, height)


def heatmap(matrix: pd.DataFrame, value: str, title: str = "", height: int = 480) -> go.Figure:
    """Crop-by-state score matrix for the recommendation heatmap."""
    if matrix is None or matrix.empty:
        return _empty("No matrix available", height)
    fig = px.imshow(
        matrix, aspect="auto", color_continuous_scale=theme.sequential_scale(),
        labels={"x": "State", "y": "Crop", "color": ""}, title=title,
    )
    return _fin(fig, height)
