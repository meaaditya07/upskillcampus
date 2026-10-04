"""Theme, palette and shared Streamlit styling.

The palette lives in config.settings so the Plotly templates and the raw CSS
cannot drift apart: both read the same dict.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from config.settings import ACCENT, APP_NAME, APP_SUBTITLE, THEMES

_MODE = "dark"


def active_palette() -> dict[str, Any]:
    """The palette for the current mode."""
    return dict(THEMES[_MODE])


def set_mode(mode: str) -> None:
    """Switch between the light and dark palettes."""
    global _MODE  # noqa: PLW0603 - module-level UI state, set once per session
    _MODE = mode if mode in THEMES else "dark"


def mode() -> str:
    """Name of the active palette."""
    return _MODE


def is_dark() -> bool:
    """True when the dark palette is active."""
    return active_palette()["mode"] == "dark"


def toggle_mode() -> None:
    """Flip between the light and dark palettes."""
    set_mode("light" if is_dark() else "dark")


def inject_css() -> None:
    """Page-level CSS. Called once from app.py."""
    p = active_palette()
    st.markdown(
        f"""
        <style>
        :root {{
            --bg: {p['bg']};
            --surface: {p['surface']};
            --surface-alt: {p['surface_alt']};
            --border: {p['border']};
            --text: {p['text']};
            --muted: {p['muted']};
            --accent: {ACCENT};
        }}
        .stApp {{ background: var(--bg); }}
        html, body, [class*="css"] {{ color: var(--text); }}

        .ay-header {{
            display: flex; align-items: baseline; gap: 0.75rem;
            flex-wrap: wrap; margin-bottom: 0.25rem;
        }}
        .ay-title {{
            font-size: 1.6rem; font-weight: 700; letter-spacing: -0.02em;
            color: var(--text); margin: 0;
        }}
        .ay-subtitle {{ color: var(--muted); font-size: 0.95rem; }}

        .ay-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 0.9rem 1rem;
            height: 100%;
        }}
        .ay-card-label {{
            color: var(--muted); font-size: 0.78rem;
            text-transform: uppercase; letter-spacing: 0.06em;
        }}
        .ay-card-value {{
            font-size: 1.55rem; font-weight: 700; margin-top: 0.2rem;
            color: var(--text);
        }}
        .ay-card-note {{ color: var(--muted); font-size: 0.8rem; }}

        .ay-note {{
            border-left: 3px solid var(--accent);
            background: var(--surface-alt);
            padding: 0.6rem 0.85rem; border-radius: 0 8px 8px 0;
            color: var(--muted); font-size: 0.85rem;
        }}
        .ay-warning {{ border-left-color: #ef4444; }}
        .ay-ok {{ border-left-color: #22c55e; }}

        .ay-badge {{
            display: inline-block; padding: 0.15rem 0.55rem;
            border-radius: 999px; font-size: 0.72rem; font-weight: 600;
            border: 1px solid var(--border); background: var(--surface-alt);
            color: var(--muted);
        }}
        .ay-badge-synthetic {{ border-color: #f59e0b; color: #f59e0b; }}

        .ay-footer {{ color: var(--muted); font-size: 0.78rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Plotly
# --------------------------------------------------------------------------
def plotly_layout(**overrides: Any) -> dict[str, Any]:
    """A Plotly layout bound to the active palette."""
    p = active_palette()
    layout = {
        "paper_bgcolor": p["paper"],
        "plot_bgcolor": p["paper"],
        "font": {"color": p["text"], "family": "system-ui, sans-serif"},
        "margin": {"l": 8, "r": 8, "t": 34, "b": 8},
        "hoverlabel": {
            "bgcolor": p["surface_alt"],
            "bordercolor": p["border"],
            "font": {"color": p["text"]},
        },
        "xaxis": {"gridcolor": p["grid"], "zerolinecolor": p["grid"],
                  "linecolor": p["border"], "tickfont": {"color": p["muted"]}},
        "yaxis": {"gridcolor": p["grid"], "zerolinecolor": p["grid"],
                  "linecolor": p["border"], "tickfont": {"color": p["muted"]}},
        "legend": {"bgcolor": "rgba(0,0,0,0)", "font": {"color": p["muted"]}},
    }
    layout.update(overrides)
    return layout


def sequential_scale() -> list[str]:
    """The palette's ordered sequential ramp."""
    return list(active_palette()["seq"])


def categorical_scale(n: int) -> list[str]:
    """A readable categorical ramp that survives the light/dark switch."""
    base = [ACCENT, "#38bdf8", "#a78bfa", "#f472b6", "#34d399", "#fbbf24"]
    if is_dark():
        base = ["#fbbf24", "#38bdf8", "#c084fc", "#f472b6", "#4ade80", "#f87171"]
    repeats = (n // len(base)) + 1
    return (base * repeats)[:n]


# --------------------------------------------------------------------------
# Small HTML helpers
# --------------------------------------------------------------------------
def header() -> None:
    """The app title and subtitle."""
    st.markdown(
        f"""
        <div class="ay-header">
          <p class="ay-title">{APP_NAME}</p>
          <span class="ay-subtitle">{APP_SUBTITLE}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def card(label: str, value: str, note: str = "") -> None:
    """One KPI tile. Use inside a st.columns block."""
    note_html = f'<div class="ay-card-note">{note}</div>' if note else ""
    st.markdown(
        f"""
        <div class="ay-card">
          <div class="ay-card-label">{label}</div>
          <div class="ay-card-value">{value}</div>
          {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def note(text: str, kind: str = "info") -> None:
    """A bordered callout: info, warning or ok."""
    css = {"info": "ay-note", "warning": "ay-note ay-warning",
           "ok": "ay-note ay-ok"}.get(kind, "ay-note")
    st.markdown(f'<div class="{css}">{text}</div>', unsafe_allow_html=True)


def badge(text: str, synthetic: bool = False) -> None:
    """A small pill label, highlighted when synthetic."""
    css = "ay-badge ay-badge-synthetic" if synthetic else "ay-badge"
    st.markdown(f'<span class="{css}">{text}</span>', unsafe_allow_html=True)


def section(title: str) -> None:
    """A section heading."""
    st.markdown(f"### {title}")
