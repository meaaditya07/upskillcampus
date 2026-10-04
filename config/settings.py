"""Central configuration for AgriYield AI.

Holds filesystem layout, reproducibility seed, UI colour tokens and the
reference price assumptions used by the profitability calculator.

Nothing in this module imports pandas, numpy or streamlit, so it is safe to
import from anywhere including lightweight CLI scripts.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Filesystem layout
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
GEO_DIR = DATA_DIR / "geo"

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

for _d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, GEO_DIR, MODEL_DIR, FIGURE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SYNTHETIC_CSV = RAW_DIR / "crop_production_synthetic.csv"
RAW_CSV_GLOB = "*.csv"
CLEANED_CSV = INTERIM_DIR / "crop_production_clean.csv"
GEOJSON_PATH = GEO_DIR / "india_states.geojson"
GEOJSON_RAW_PATH = GEO_DIR / "india_raw.geojson"

METRICS_PATH = MODEL_DIR / "metrics.json"
REPORT_PATH = REPORT_DIR / "training_report.md"

APP_NAME = "AgriYield AI"
APP_SUBTITLE = "Crop Production & Cost Forecasting Platform"
APP_COUNTRY = "India"

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
SEED = 20240101

# --------------------------------------------------------------------------
# Reference period covered by the synthetic generator
# --------------------------------------------------------------------------
YEAR_MIN = 2001
YEAR_MAX = 2014

# Number of synthetic observations. Override with SYNTHETIC_ROWS env var.
SYNTHETIC_ROWS = 60_000

# --------------------------------------------------------------------------
# Units
# --------------------------------------------------------------------------
# Indian agriculture uses the 50 kg quintal, so:
#     1 tonne        = 1000 / 50 = 20 quintals
#     1 kg           = 1 / 50     = 0.02 quintals
#     170 kg cotton bale     = 170 / 50 = 3.4 quintals
# A 100 kg sugarcane bundle, the other trade unit that shows up in these
# extracts, is 2 quintals. These are the multipliers that convert one unit of
# `production` into quintals; getting one wrong silently rescales a whole
# column, so they are asserted in tests/test_units.py.
QUINTAL_PER_TONNE = 20.0
KG_PER_QUINTAL = 50.0

UNIT_TO_QUINTAL_FACTOR = {
    "quintal": 1.0,
    "quintals": 1.0,
    "qtl": 1.0,
    "qtl.": 1.0,
    "kilogram": 1.0 / KG_PER_QUINTAL,
    "kg": 1.0 / KG_PER_QUINTAL,
    "kilogram(s)": 1.0 / KG_PER_QUINTAL,
    "kgs": 1.0 / KG_PER_QUINTAL,
    "ton": QUINTAL_PER_TONNE,
    "tons": QUINTAL_PER_TONNE,
    "tonne": QUINTAL_PER_TONNE,
    "tonnes": QUINTAL_PER_TONNE,
    "t": QUINTAL_PER_TONNE,
    "mt": QUINTAL_PER_TONNE,
    "metric ton": QUINTAL_PER_TONNE,
    "metric tons": QUINTAL_PER_TONNE,
    "metric tonne": QUINTAL_PER_TONNE,
    "metric tonnes": QUINTAL_PER_TONNE,
    "bale": 170.0 / KG_PER_QUINTAL,        # 170 kg cotton bale
    "bales": 170.0 / KG_PER_QUINTAL,
    "bundle": 100.0 / KG_PER_QUINTAL,      # 100 kg sugarcane bundle
    "bundles": 100.0 / KG_PER_QUINTAL,
}

#: Derived, so the quintal conversions are self-consistent by construction.
assert abs(UNIT_TO_QUINTAL_FACTOR["ton"] / QUINTAL_PER_TONNE - 1.0) < 1e-9
assert abs(UNIT_TO_QUINTAL_FACTOR["kg"] * KG_PER_QUINTAL - 1.0) < 1e-9

# --------------------------------------------------------------------------
# Seasons
# --------------------------------------------------------------------------
SEASON_NAMES = ("Kharif", "Rabi", "Zaid", "Whole Year")
SEASON_DURATIONS = ("Short", "Medium", "Long")

# How a raw `Season` string maps onto (season_name, season_duration). The brief
# describes Season as "a duration category or range (e.g. Medium, Long, Kharif,
# Rabi)", so a single column may carry either concept -- both are recovered.
SEASON_LOOKUP = {
    "kharif": ("Kharif", "Medium"),
    "swarna": ("Kharif", "Medium"),
    "kharif (swarna)": ("Kharif", "Medium"),
    "rabi": ("Rabi", "Medium"),
    "rabi (winter)": ("Rabi", "Long"),
    "winter": ("Rabi", "Long"),
    "autumn": ("Rabi", "Medium"),
    "zaid": ("Zaid", "Short"),
    "summer": ("Zaid", "Short"),
    "zayad": ("Zaid", "Short"),
    "kharif (long)": ("Kharif", "Long"),
    "whole year": ("Whole Year", "Long"),
    "whole year crop": ("Whole Year", "Long"),
    "annual": ("Whole Year", "Long"),
    "short": ("Kharif", "Short"),
    "medium": ("Rabi", "Medium"),
    "long": ("Kharif", "Long"),
}

# Crops and the seasonal window they are typically grown in. Drives both the
# generator and the season recommendation heatmap.
CROP_SEASON_AFFINITY = {
    "Rice": ("Kharif",),
    "Maize": ("Kharif", "Rabi"),
    "Jute": ("Kharif",),
    "Cotton": ("Kharif",),
    "Groundnut": ("Kharif",),
    "Soybean": ("Kharif",),
    "Millets": ("Kharif", "Rabi"),
    "Tur": ("Kharif",),
    "Urad": ("Kharif", "Rabi"),
    "Wheat": ("Rabi",),
    "Gram": ("Rabi",),
    "Rapeseed": ("Rabi",),
    "Mustard": ("Rabi",),
    "Sugarcane": ("Whole Year",),
    "Onion": ("Rabi", "Zaid"),
    "Potato": ("Rabi",),
}

CROP_GROUP = {
    "Rice": "Cereals",
    "Wheat": "Cereals",
    "Maize": "Cereals",
    "Millets": "Cereals",
    "Jute": "Cash Crops",
    "Cotton": "Cash Crops",
    "Sugarcane": "Cash Crops",
    "Groundnut": "Oilseeds",
    "Soybean": "Oilseeds",
    "Rapeseed": "Oilseeds",
    "Mustard": "Oilseeds",
    "Tur": "Pulses",
    "Urad": "Pulses",
    "Gram": "Pulses",
    "Onion": "Vegetables",
    "Potato": "Vegetables",
}

# --------------------------------------------------------------------------
# Indicative farm-gate prices  (ASSUMPTION -- not an official MSP series)
# --------------------------------------------------------------------------
# Representative modern-era MSP levels in INR per quintal. These are used only
# by the profitability calculator. They are explicitly *assumed*, not measured,
# and the UI labels every derived rupee figure as price-assumption dependent.
INDICATIVE_PRICE_PER_QUINTAL = {
    "Rice": 2300.0,
    "Wheat": 2275.0,
    "Maize": 2225.0,
    "Millets": 2600.0,
    "Cotton": 7710.0,
    "Jute": 3000.0,
    "Sugarcane": 340.0,
    "Groundnut": 6783.0,
    "Soybean": 5328.0,
    "Rapeseed": 5950.0,
    "Mustard": 5950.0,
    "Tur": 7000.0,
    "Urad": 6400.0,
    "Gram": 5875.0,
    "Onion": 1600.0,
    "Potato": 1180.0,
}

PRICE_ASSUMPTION_NOTE = (
    "Indicative farm-gate prices (assumed). Not an official MSP time series."
)

# The data window is 2001-2014 but the indicative prices above are modern-era.
# Applying them flat would be anachronistic, so a linear vintage adjustment
# anchors YEAR_MAX at the indicative value and scales earlier years down.
PRICE_VINTAGE_FLOOR = 0.45


def indicative_price(crop: str, year: float | None = None) -> tuple[float, int]:
    """Return (effective price in INR/quintal, reference year used).

    Without ``year`` the raw indicative price is returned. With ``year`` the
    value is vintage-adjusted across YEAR_MIN..YEAR_MAX so that profitability
    figures stay internally consistent with the dataset's time span.
    """
    base = INDICATIVE_PRICE_PER_QUINTAL.get(str(crop).strip().title())
    if base is None:
        # Unknown crop: fall back to the catalogue median so the profitability
        # calculator still returns a finite, clearly-flagged estimate.
        ordered = sorted(INDICATIVE_PRICE_PER_QUINTAL.values())
        mid = len(ordered) // 2
        base = (
            ordered[mid]
            if len(ordered) % 2
            else (ordered[mid - 1] + ordered[mid]) / 2.0
        )
    if year is None:
        return base, YEAR_MAX
    span = max(YEAR_MAX - YEAR_MIN, 1)
    t = min(max((float(year) - YEAR_MIN) / span, 0.0), 1.0)
    factor = PRICE_VINTAGE_FLOOR + (1.0 - PRICE_VINTAGE_FLOOR) * t
    return base * factor, round(float(year))


# --------------------------------------------------------------------------
# UI theme
# --------------------------------------------------------------------------
PRIMARY = "#16a34a"
PRIMARY_DARK = "#15803d"
PRIMARY_LIGHT = "#4ade80"
ACCENT = "#f59e0b"
DANGER = "#dc2626"
INFO = "#0ea5e9"

# Diverging pair used for the correlation heatmap (cost <-> efficiency).
DIVERGING_LOW = "#b91c1c"
DIVERGING_MID = "#f8fafc"
DIVERGING_HIGH = "#166534"

THEMES = {
    "dark": {
        "mode": "dark",
        "bg": "#0b1220",
        "surface": "#111a2b",
        "surface_alt": "#16223a",
        "border": "#22304d",
        "text": "#e6edf3",
        "muted": "#93a4bd",
        "grid": "#22304d",
        "seq": ["#0b1220", "#14532d", "#16a34a", "#4ade80", "#bbf7d0"],
        "font": "#e6edf3",
        "paper": "#0b1220",
    },
    "light": {
        "mode": "light",
        "bg": "#f8fafc",
        "surface": "#ffffff",
        "surface_alt": "#f1f5f9",
        "border": "#d9e2ec",
        "text": "#0f172a",
        "muted": "#5b6b83",
        "grid": "#d9e2ec",
        "seq": ["#f0fdf4", "#bbf7d0", "#4ade80", "#16a34a", "#14532d"],
        "font": "#0f172a",
        "paper": "#ffffff",
    },
}

# --------------------------------------------------------------------------
# Display labels for canonical columns
# --------------------------------------------------------------------------
COLUMN_LABELS = {
    "crop": "Crop",
    "variety": "Variety",
    "state": "State",
    "quantity": "Cultivation Quantity",
    "production": "Production",
    "season": "Season (raw)",
    "season_name": "Season",
    "season_duration": "Season Duration",
    "unit": "Unit",
    "cost": "Cultivation Cost",
    "recommended_zone": "Recommended Zone",
    "year": "Year",
    "cost_per_unit": "Cost per Unit",
    "production_efficiency": "Production Efficiency",
    "zone_scope": "Zone Scope",
    "zone_name": "Zone Name",
    "crop_group": "Crop Group",
    "variety_class": "Variety Class",
    "quantity_tons": "Quantity (tonnes)",
    "production_tons": "Production (tonnes)",
}

CURRENCY = "INR"

# Recommended-variety scoring weights (user-adjustable in the UI).
DEFAULT_SCORE_WEIGHTS = {
    "efficiency": 0.5,
    "cost": 0.3,
    "reliability": 0.2,
}

# Below this many observations a recommendation is flagged low-confidence.
LOW_CONFIDENCE_N = 30

# Bootstrap / interval configuration
DEFAULT_BOOTSTRAP_N = 20
INTERVAL_ALPHA = 0.10          # -> 80% prediction interval
INTERVAL_QUANTILE_LOW = 0.10
INTERVAL_QUANTILE_HIGH = 0.90
