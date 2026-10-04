"""AgriYield AI -- Indian crop production & cost forecasting.

Run with::

    python -m streamlit run app.py

Business logic lives in ``src/services`` and never imports Streamlit, so this
file is only navigation and layout.

Routing note: the ``page_*.py`` modules are plain modules that *export* a
``render()`` callable; they do not run anything on import. Navigation is
declared here with ``st.navigation`` and handed the callables directly, so
clicking a sidebar entry re-runs this script and invokes exactly one
``render()``. Do not add ``st.page_link`` calls here -- linking to a
``page_*.py`` path would make Streamlit execute that file as its own entry
point, which renders nothing.
"""

from __future__ import annotations

import streamlit as st

import page_diagnostics
import page_overview
import page_predictor
import page_recommend
from components import theme

st.set_page_config(
    page_title="AgriYield AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

theme.inject_css()

# ``url_path`` must be set explicitly: Streamlit infers it from the callable
# name, and every page exports a function called ``render``, so all four would
# otherwise collide on the path "render".
PAGES = [
    st.Page(
        page_overview.render,
        title="Overview",
        icon="📊",
        url_path="overview",
        default=True,
    ),
    st.Page(
        page_predictor.render,
        title="Predictor",
        icon="🌾",
        url_path="predictor",
    ),
    st.Page(
        page_recommend.render,
        title="Recommend",
        icon="🏆",
        url_path="recommend",
    ),
    st.Page(
        page_diagnostics.render,
        title="Diagnostics",
        icon="🔬",
        url_path="diagnostics",
    ),
]

nav = st.navigation(PAGES)

try:
    nav.run()
except FileNotFoundError as exc:
    st.error(f"A required data file is missing: {exc}")
    st.code("python scripts/bootstrap.py", language="bash")
except Exception as exc:
    st.error(f"{type(exc).__name__}: {exc}")
    st.exception(exc)
