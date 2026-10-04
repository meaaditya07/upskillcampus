"""Render every page's ``render()`` under a headless Streamlit runtime.

Catches the mistakes unit tests cannot see: a Plotly figure built from a column
that does not exist, a widget created inside a conditional, a name that only
resolves in the page's own branch.

Each page module only *exports* ``render()`` -- it runs nothing on import, so
``AppTest.from_file("page_x.py")`` would import the module and render nothing.
These tests therefore drive each page through a generated entry point that
actually calls ``render()``, then assert the page produced visible content and
no errors. Several pages swallow their own exceptions into ``st.error``, so
checking ``at.exception`` alone is not enough.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("streamlit")

PAGES = {
    "page_overview": "Overview",
    "page_predictor": "Predictor",
    "page_recommend": "Recommend",
    "page_diagnostics": "Diagnostics",
}

# Element types that count as "the page actually drew something".
CONTENT_TYPES = (
    "markdown",
    "dataframe",
    "plotly_chart",
    "selectbox",
    "multiselect",
    "slider",
    "metric",
    "title",
    "header",
    "subheader",
)


@pytest.fixture(scope="module")
def page_modules():
    modules = {name: __import__(name) for name in PAGES}
    return modules


@pytest.fixture(scope="module")
def drivers(tmp_path_factory):
    """One generated Streamlit entry point per page, each calling render()."""
    out = {}
    directory = tmp_path_factory.mktemp("drivers")
    for name in PAGES:
        driver = directory / f"driver_{name}.py"
        driver.write_text(
            "import sys\n"
            f"sys.path.insert(0, {str(ROOT)!r})\n"
            "import streamlit as st\n"
            "from components import theme\n"
            "st.set_page_config(page_title='driver', layout='wide')\n"
            "theme.inject_css()\n"
            f"import {name}\n"
            f"{name}.render()\n",
            encoding="utf-8",
        )
        out[name] = driver
    return out


def test_all_page_modules_expose_render(page_modules):
    for name, module in page_modules.items():
        assert callable(getattr(module, "render", None)), (
            f"{name} must export a callable render(); app.py hands it to st.Page"
        )


def test_app_module_imports():
    assert pytest.importorskip("app") is not None


@pytest.mark.parametrize("name", list(PAGES))
def test_page_renders_without_error(name, drivers):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(drivers[name]), default_timeout=300)
    at.run()

    # ``at.exception`` is an ElementList, not None: test emptiness, not identity.
    assert len(at.exception) == 0, f"{name}.render() raised {at.exception[0].message}"

    # Pages catch their own failures and show st.error; surface those too.
    errors = [e.value for e in at.error]
    assert not errors, f"{name}.render() displayed errors: {errors}"

    drawn = sum(len(at.get(kind)) for kind in CONTENT_TYPES)
    assert drawn > 0, (
        f"{name}.render() drew nothing. A page module that only defines "
        "render() and never calls it will pass an import-only test silently."
    )


def test_app_entry_point_renders_overview():
    """The real launch target must boot and draw the default page.

    This is deliberately a test of ``app.py`` itself, not of a generated
    driver. Routing bugs live in ``app.py``: an earlier version registered four
    pages whose callables were all named ``render``, so Streamlit inferred the
    same URL path for all of them and refused to start -- while every per-page
    test passed, because each one bypassed the navigation entirely.
    """
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
    at.run()

    assert len(at.exception) == 0, f"app.py failed to start: {at.exception[0].message}"
    assert not [e.value for e in at.error], "app.py rendered errors"
    headings = [
        m.value for m in at.markdown
        if isinstance(m.value, str) and m.value.startswith("#")
    ]
    assert any("Overview" in h for h in headings), (
        f"app.py did not render the default page; headings were {headings}"
    )


def test_app_registers_unique_page_urls():
    """Every st.Page needs a distinct url_path; the callables are all ``render``.

    Checked against the source with ``ast`` rather than by importing ``app``:
    ``Page.url_path`` is only populated once a page is registered with a live
    ``st.navigation`` call, so it raises outside a real script run.
    """
    import ast

    tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
    paths = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "Page"):
            continue
        for keyword in node.keywords:
            if keyword.arg == "url_path" and isinstance(keyword.value, ast.Constant):
                paths.append(keyword.value.value)

    assert len(paths) == 4, f"expected 4 explicit url_path values, found {paths}"
    assert len(set(paths)) == len(paths), f"duplicate url_path values: {paths}"


@pytest.mark.parametrize("name", list(PAGES))
def test_page_survives_missing_artefacts(name, drivers, monkeypatch):
    """A user who skipped bootstrap must see guidance, not a stack trace."""
    from streamlit.testing.v1 import AppTest

    from src.services import registry

    # ``training_status()`` gates on ``missing_artifacts()``, which is what the
    # sidebar reads, so patch the real gate rather than artifacts_ready().
    monkeypatch.setattr(
        registry,
        "missing_artifacts",
        lambda: ["xgboost_yield.joblib", "metrics.json"],
        raising=False,
    )
    at = AppTest.from_file(str(drivers[name]), default_timeout=300)
    at.run()

    assert len(at.exception) == 0, (
        f"{name} crashed without artefacts: {at.exception[0].message}"
    )
    # Guidance for a user who skipped bootstrap is expected to be loud: the
    # sidebar raises st.error, and pages add warnings/notes of their own.
    notices = [e.value for e in at.error] + [e.value for e in at.warning]
    notices += [e.value for e in at.info]
    assert notices, f"{name} gave no guidance when artefacts were missing"


@pytest.mark.parametrize(
    "module_name", ["charts", "theme", "kpi_cards", "sidebar"]
)
def test_component_callables_are_documented(module_name):
    # Every public helper in the presentation layer earns its keep by being
    # readable at the call site.
    module = __import__(f"components.{module_name}", fromlist=[module_name])
    undocumented = [
        attribute
        for attribute in dir(module)
        if not attribute.startswith("_")
        and callable(getattr(module, attribute))
        and not getattr(module, attribute).__doc__
    ]
    assert not undocumented, f"components.{module_name}: {undocumented}"


def test_sidebar_exposes_reference_lists():
    # Empty dropdowns are the failure mode when a component asks the wrong
    # module for known states/crops, so assert they are actually populated.
    from components import sidebar

    states = sidebar._safe_list(sys.modules["src.services.predictor"], "known_states")
    crops = sidebar._safe_list(sys.modules["src.services.predictor"], "crops")
    assert len(states) > 30, f"expected the full state list, got {len(states)}"
    assert len(crops) > 10, f"expected the full crop list, got {len(crops)}"


def test_theme_injects_css_without_error():
    from components import theme

    theme.set_mode("dark")
    assert theme.is_dark()
    theme.set_mode("light")
    assert not theme.is_dark()
    theme.set_mode("dark")

    layout = theme.plotly_layout()
    assert "paper_bgcolor" in layout
    assert theme.categorical_scale(3)
    assert theme.sequential_scale()


def test_kpi_formatters():
    from components import kpi_cards

    assert kpi_cards.compact(1_500_000) == "1.5M"
    assert kpi_cards.compact(999) == "999"
    assert "₹" in kpi_cards.rupees(1_234_567)
    assert kpi_cards.percent(81.306) == "81.3%"


def test_benchmark_summary_reads_per_model_blocks():
    from page_diagnostics import _benchmark_summary

    metrics = {
        "models": {
            "production_quintals": {
                "metrics": {"production_quintals_r2": 0.8667},
                "lgbm": {"production_quintals_r2": 0.8738},
            },
            "cost": {
                "metrics": {"cost_r2": 0.8217},
                "lgbm": {"cost_r2": 0.8290},
            },
        }
    }
    summary = _benchmark_summary(metrics)
    assert "+0.0071" in summary and "+0.0073" in summary

    missing = _benchmark_summary({"models": {"cost": {"metrics": {"cost_r2": 0.8}}}})
    assert "not benchmarked" in missing
