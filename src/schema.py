"""Canonical dataset schema, column synonym resolution and value normalisation.

This module is the single source of truth for what a valid record looks like.
It lets an arbitrary third-party CSV (different column spellings, dtypes, unit
spellings) be mapped onto the canonical schema the rest of the pipeline expects.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from config.settings import (
    CROP_GROUP,
    SEASON_LOOKUP,
    UNIT_TO_QUINTAL_FACTOR,
)

# --------------------------------------------------------------------------
# Canonical columns
# --------------------------------------------------------------------------
CANONICAL_COLUMNS = (
    "crop",
    "variety",
    "state",
    "quantity",
    "production",
    "season",
    "unit",
    "cost",
    "recommended_zone",
    "year",
)

STRING_COLUMNS = ("crop", "variety", "state", "season", "unit", "recommended_zone")
NUMERIC_COLUMNS = ("quantity", "production", "cost", "year")

REQUIRED_COLUMNS = (
    "crop",
    "state",
    "quantity",
    "production",
    "cost",
)

# Human-facing ordering used by the UI and by the data-quality audit.
DISPLAY_ORDER = (
    "year",
    "state",
    "recommended_zone",
    "crop",
    "variety",
    "season",
    "unit",
    "quantity",
    "production",
    "cost",
)

# --------------------------------------------------------------------------
# Column synonym resolution
# --------------------------------------------------------------------------
# Lower-cased, punctuation-stripped header -> canonical name. Keys are compared
# after _normalise_key(), so 'Recommended Zone', 'recommended_zone' and
# 'RECOMMENDED-ZONE' all collapse to the same key.
COLUMN_SYNONYMS: dict[str, str] = {
    # crop
    "crop": "crop",
    "crop name": "crop",
    "cropname": "crop",
    "commodity": "crop",
    # variety
    "variety": "variety",
    "varieties": "variety",
    "variety name": "variety",
    "cultivar": "variety",
    "hybrid": "variety",
    # state
    "state": "state",
    "state name": "state",
    "statename": "state",
    "states": "state",
    # quantity / area
    "quantity": "quantity",
    "area": "quantity",
    "area hectares": "quantity",
    "cultivated area": "quantity",
    "cultivation quantity": "quantity",
    "area cultivated": "quantity",
    "extent": "quantity",
    # production
    "production": "production",
    "yield": "production",
    "production quintals": "production",
    "output": "production",
    "total production": "production",
    # season
    "season": "season",
    "seasons": "season",
    "season duration": "season",
    "duration": "season",
    "crop season": "season",
    # unit
    "unit": "unit",
    "units": "unit",
    "unit of measurement": "unit",
    "production unit": "unit",
    # cost
    "cost": "cost",
    "cost of cultivation": "cost",
    "cultivation cost": "cost",
    "total cost": "cost",
    "cost inr": "cost",
    "cost rs": "cost",
    "total cost of cultivation and production": "cost",
    # zone
    "recommended zone": "recommended_zone",
    "recommendedzone": "recommended_zone",
    "zone": "recommended_zone",
    "zones": "recommended_zone",
    "mandal": "recommended_zone",
    "village": "recommended_zone",
    "district": "recommended_zone",
    "recommendation zone": "recommended_zone",
    # year
    "year": "year",
    "crop year": "year",
    "cropyear": "year",
    "yr": "year",
    "year of cultivation": "year",
}

_PUNCT = re.compile(r"[^a-z0-9 ]+")


def _normalise_key(header: str) -> str:
    """Lower-case, strip punctuation and collapse whitespace in a column name."""
    text = _PUNCT.sub(" ", str(header).strip().lower())
    return " ".join(text.split())


def resolve_columns(columns) -> tuple[dict[str, str], list[str]]:
    """Map raw CSV headers onto canonical names.

    Returns (canonical_name -> original_header, unmapped_original_headers).
    """
    resolved: dict[str, str] = {}
    unmapped: list[str] = []
    for header in columns:
        key = _normalise_key(header)
        canonical = COLUMN_SYNONYMS.get(key)
        if canonical and canonical not in resolved:
            resolved[canonical] = header
        elif canonical is None:
            unmapped.append(str(header))
    return resolved, unmapped


# --------------------------------------------------------------------------
# Place-name canonicalisation
# --------------------------------------------------------------------------
# Single source of truth for Indian state names. Both the data loader (which
# normalises the dataset) and src/geo.py (which normalises the basemap) resolve
# through this table, so a spelling that the loader canonicalises can never
# disagree with the spelling the map uses.
STATE_ALIASES: dict[str, str] = {
    "andhra pradesh": "Andhra Pradesh",
    "andhra": "Andhra Pradesh",
    "ap": "Andhra Pradesh",
    "arunachal pradesh": "Arunachal Pradesh",
    "arunachal": "Arunachal Pradesh",
    "assam": "Assam",
    "bihar": "Bihar",
    "chhattisgarh": "Chhattisgarh",
    "chhattishgarh": "Chhattisgarh",
    "chhattisgharh": "Chhattisgarh",
    "chandigarh": "Chandigarh",
    "dadra and nagar haveli": "Dadra and Nagar Haveli and Daman and Diu",
    "dadra and nagar haveli and daman and diu":
        "Dadra and Nagar Haveli and Daman and Diu",
    "daman and diu": "Dadra and Nagar Haveli and Daman and Diu",
    "daman": "Dadra and Nagar Haveli and Daman and Diu",
    "diu": "Dadra and Nagar Haveli and Daman and Diu",
    "delhi": "Delhi",
    "nct of delhi": "Delhi",
    "nct delhi": "Delhi",
    "new delhi": "Delhi",
    "national capital territory of delhi": "Delhi",
    "goa": "Goa",
    "gujarat": "Gujarat",
    "haryana": "Haryana",
    "himachal pradesh": "Himachal Pradesh",
    "himachal": "Himachal Pradesh",
    "jammu": "Jammu and Kashmir",
    "jammu and kashmir": "Jammu and Kashmir",
    "jammu & kashmir": "Jammu and Kashmir",
    "j&k": "Jammu and Kashmir",
    "jammu kashmir": "Jammu and Kashmir",
    "jharkhand": "Jharkhand",
    "karnataka": "Karnataka",
    "kerala": "Kerala",
    "ladakh": "Ladakh",
    "lakshadweep": "Lakshadweep",
    "lakshadweep islands": "Lakshadweep",
    "madhya pradesh": "Madhya Pradesh",
    "maharashtra": "Maharashtra",
    "manipur": "Manipur",
    "meghalaya": "Meghalaya",
    "mizoram": "Mizoram",
    "nagaland": "Nagaland",
    "odisha": "Odisha",
    "orissa": "Odisha",
    "pondicherry": "Puducherry",
    "puducherry": "Puducherry",
    "punjab": "Punjab",
    "rajasthan": "Rajasthan",
    "sikkim": "Sikkim",
    "tamil nadu": "Tamil Nadu",
    "telangana": "Telangana",
    "tripura": "Tripura",
    "uttar pradesh": "Uttar Pradesh",
    "uttarakhand": "Uttarakhand",
    "uttaranchal": "Uttarakhand",
    "west bengal": "West Bengal",
    "bengal": "West Bengal",
    "andaman and nicobar islands": "Andaman and Nicobar Islands",
    "andaman & nicobar islands": "Andaman and Nicobar Islands",
    "andaman and nicobar": "Andaman and Nicobar Islands",
}

_CANONICAL_STATES = sorted(set(STATE_ALIASES.values()))

CROP_ALIASES: dict[str, str] = {
    "rice": "Rice",
    "paddy": "Rice",
    "wheat": "Wheat",
    "barley": "Wheat",
    "maize": "Maize",
    "corn": "Maize",
    "cotton": "Cotton",
    "sugarcane": "Sugarcane",
    "sugar cane": "Sugarcane",
    "groundnut": "Groundnut",
    "ground nut": "Groundnut",
    "peanut": "Groundnut",
    "soybean": "Soybean",
    "soya bean": "Soybean",
    "soyabean": "Soybean",
    "gram": "Gram",
    "chana": "Gram",
    "chickpea": "Gram",
    "tur": "Tur",
    "arhar": "Tur",
    "pigeonpea": "Tur",
    "urad": "Urad",
    "black gram": "Urad",
    "rapeseed": "Rapeseed",
    "mustard": "Mustard",
    "onion": "Onion",
    "potato": "Potato",
    "jute": "Jute",
    "millets": "Millets",
    "millet": "Millets",
    "bajra": "Millets",
    "pearl millet": "Millets",
    "jowar": "Millets",
    "sorghum": "Millets",
    "ragi": "Millets",
    "finger millet": "Millets",
    "tea": "Millets",
    "cashew": "Millets",
    "coconut": "Millets",
    "oil palm": "Millets",
    "spices": "Tur",
    "apple": "Millets",
    "cardamom": "Tur",
    "fish": "Millets",
}

_WS = re.compile(r"\s+")


def _is_missing(value: object) -> bool:
    """True for every flavour of "no value".

    The obvious ``value != value`` float-NaN test is not enough: pandas 3.x
    nullable string columns hold ``pd.NA``, which compares *against itself*
    without raising and so slips through an ``isinstance(value, float)`` guard.
    That turned missing units into the literal string ``"<Na>"`` -- a one-hot
    level of its own -- so every parser below routes through this instead.
    """
    if value is None:
        return True
    if isinstance(value, float) and value != value:  # noqa: PLR0124 - NaN test
        return True
    return value is pd.NA or value is pd.NaT


def _place_key(value: object) -> str:
    """Lowercase, whitespace-collapsed lookup key.

    Missing values become the empty string, never the literal text
    ``"none"``/``"nan"`` -- otherwise a missing state would be title-cased into
    a plausible-looking place name and appear in the choropleth.
    """
    if _is_missing(value):
        return ""
    return _WS.sub(" ", str(value).strip().lower()).strip(" -_")


def canonical_state(value: object) -> str:
    """Canonical state name, or a title-cased fallback for unknown spellings."""
    key = _place_key(value)
    if not key:
        return ""
    if key in STATE_ALIASES:
        return STATE_ALIASES[key]
    collapsed = key.replace(" ", "")
    if collapsed in STATE_ALIASES:
        return STATE_ALIASES[collapsed]
    return " ".join(word.capitalize() for word in key.split())


def is_known_state(value: object) -> bool:
    """True when ``value`` resolves to a real Indian state/UT.

    Distinguishes "Odisha" from "Zzz": both canonicalise to a display string,
    but only one is a place the dataset and basemap can meaningfully share.
    """
    key = _place_key(value)
    if not key:
        return False
    return key in STATE_ALIASES or key.replace(" ", "") in STATE_ALIASES


def known_states() -> list[str]:
    """All recognised state/UT names, alphabetically."""
    return list(_CANONICAL_STATES)


def canonical_crop(value: object) -> str:
    key = _place_key(value)
    if not key:
        return ""
    return CROP_ALIASES.get(key, " ".join(w.capitalize() for w in key.split()))


# --------------------------------------------------------------------------
# Season parsing
# --------------------------------------------------------------------------
_DURATION_TOKENS = ("short", "medium", "long")


def parse_season(raw: object) -> tuple[str, str]:
    """Return ``(season_name, season_duration)`` from a raw Season value.

    The brief describes Season as either a duration category ('Medium', 'Long')
    or an agricultural season ('Kharif', 'Rabi'), so both are recovered and
    returned separately. Unrecognised values fall back to ('Other', 'Medium')
    rather than being dropped.
    """
    if _is_missing(raw):
        return ("Unknown", "Medium")
    text = " ".join(str(raw).strip().lower().replace("_", " ").split())
    if not text:
        return ("Unknown", "Medium")

    if text in SEASON_LOOKUP:
        return SEASON_LOOKUP[text]

    # Substring match, e.g. "Rice (Kharif)" or "Kharif season".
    for token in ("whole year", "kharif", "rabi", "zaid", "zayad", "summer",
                  "winter", "autumn", "annual"):
        if token in text:
            name = {
                "whole year": "Whole Year",
                "kharif": "Kharif",
                "rabi": "Rabi",
                "zaid": "Zaid",
                "zayad": "Zaid",
                "summer": "Zaid",
                "winter": "Rabi",
                "autumn": "Rabi",
                "annual": "Whole Year",
            }[token]
            duration = next(
                (d.capitalize() for d in _DURATION_TOKENS if d in text), "Medium"
            )
            return (name, duration)

    for token in _DURATION_TOKENS:
        if token in text:
            mapped = SEASON_LOOKUP.get(token)
            return mapped if mapped else ("Other", token.capitalize())

    return ("Other", "Medium")


# --------------------------------------------------------------------------
# Unit parsing
# --------------------------------------------------------------------------
def quintal_factor(raw: object) -> float:
    """Multiplier converting one unit of ``raw`` into quintals.

    Unknown units default to 1.0 (treated as quintals) so no data is lost; the
    caller records this in the audit trail.
    """
    if _is_missing(raw):
        return 1.0
    key = " ".join(str(raw).strip().lower().replace("_", " ").split())
    return UNIT_TO_QUINTAL_FACTOR.get(key, 1.0)


def canonical_unit(raw: object) -> str:
    """Fold the many spellings of a unit onto one label.

    Without this, 'Tons', 'tons', 'TONS', 'Tonne' and ' Tonne ' each become their
    own one-hot level and fragment the model's unit feature while carrying no
    additional information.
    """
    if _is_missing(raw):
        return "Unknown"
    key = " ".join(str(raw).strip().lower().replace("_", " ").split())
    if not key or key in MISSING_UNIT_TOKENS:
        return "Unknown"
    if key in MASS_UNIT_TOKENS:
        return "Tons"
    if key in COUNT_UNIT_TOKENS:
        return "Quintals"
    return str(raw).strip().title() or "Unknown"


MISSING_UNIT_TOKENS = {
    "", "na", "n/a", "nan", "none", "null", "-", "--", "unknown",
    # Literal text a spreadsheet export writes for an empty cell. Without these,
    # such a cell becomes its own one-hot level named "<Na>".
    "<na>", "<nan>", "#n/a", "#na", "n.a.",
}

MASS_UNIT_TOKENS = {
    "ton", "tons", "tonne", "tonnes", "t", "mt", "metric ton",
    "metric tons", "metric tonne", "metric tonnes",
}

COUNT_UNIT_TOKENS = {"quintal", "quintals", "qtl", "qtl."}


# --------------------------------------------------------------------------
# Zone parsing
# --------------------------------------------------------------------------
ZONE_SCOPES = ("State", "Mandal", "Village")


def parse_zone(raw: object) -> tuple[str, str]:
    """Split a Recommended Zone string into ``(zone_scope, zone_name)``.

    'Mandal - Anand' -> ('Mandal', 'Anand')
    'Village Kashi'  -> ('Village', 'Kashi')
    'Punjab'         -> ('State', 'Punjab')
    """
    if _is_missing(raw):
        return ("Unknown", "Unknown")

    # Separators seen in the wild: plain hyphen, en dash, em dash, colon,
    # slash, pipe. Written as escapes because the literal dash characters
    # are invisible in most editors and had previously been written as
    # mojibake, which silently broke zone parsing for en/em-dash input.
    text = " ".join(str(raw).strip().split())
    if not text:
        return ("Unknown", "Unknown")

    lowered = text.lower()
    for separator in (" - ", "\u2013", "\u2014", ": ", "/", " | "):
        if separator in text:
            head, _, tail = text.partition(separator)
            head_l = head.strip().lower()
            if head_l.startswith("mandal"):
                return ("Mandal", tail.strip())
            if head_l.startswith("village"):
                return ("Village", tail.strip())
            if head_l.startswith("state"):
                return ("State", tail.strip())

    for token, scope in (("mandal", "Mandal"), ("village", "Village"),
                         ("district", "Mandal"), ("block", "Mandal")):
        if lowered.startswith(token):
            remainder = text[len(token):].strip(" -\u2013\u2014:")
            return (scope, remainder or text)
        if lowered.endswith(token):
            head = text[: len(text) - len(token)].strip(" -\u2013\u2014:")
            return (scope, head or text)

    return ("State", text)


def canonical_zone(raw: object) -> str:
    """Render a recommended zone in the single form the dataset stores.

    Every zone in the data is ``"<Scope> - <Name>"`` with a capitalised name
    ('Mandal - Malda', 'State - Sikkim'). Zone names are target-encoded, so
    'mandal - ludhiana' arriving from a form is a *different* category from
    'Mandal - Ludhiana' and the encoder treats it as unseen -- a wrong forecast
    with no warning.

    Casing of an already-capitalised name is deliberately left alone so real
    place names keep their spelling ('Dadra and Nagar Haveli and Daman and
    Diu' must not become '...And...'). Only an all-lowercase name, which is
    always a typing artefact, is capitalised.
    """
    scope, name = parse_zone(raw)
    if scope == "Unknown":
        return "Unknown"
    collapsed = " ".join(str(name).split())
    if collapsed and collapsed == collapsed.lower():
        collapsed = collapsed.title()
    return f"{scope} - {collapsed}"


# --------------------------------------------------------------------------
# Variety classification
# --------------------------------------------------------------------------
# Order matters: the first matching pattern wins.
VARIETY_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Basmati", ("basmati",)),
    ("Hybrid", ("hybrid", "hk", "bh", "cotton hybrid")),
    ("Swarna", ("swarna",)),
    ("Desi", ("desi", "local", "indigenous", "traditional", "native")),
    ("Improved", ("improved", "released", "variety", "vb", "sb", "pb", "mb")),
    ("IR Series", ("ir-", "ir ", "ir64", "ir 64")),
    ("HD Series", ("hd ", "hd-", "hd2")),
    ("LRA", ("lra",)),
)

_RECORDED = (
    "Pusa", "Loka", "Sharbati", "Kalinga", "Ganga", "Sarada", "Swarna",
    "Basmati", "Sonar", "Mahima", "Samba", "Vijay", "Pragati", "Sujata",
    "Shakti", "Kaveri", "Indus", "Jaya", "Samba Mahsuri", "Birmapuri",
    "Mallika", "Saras", "Haryana", "Laddu", "Amol", "Kohinoor",
)


def classify_variety(raw: object) -> str:
    """Bucket a variety name into a broad class for encoding."""
    if _is_missing(raw):
        return "Unknown"
    text = " ".join(str(raw).strip().lower().split())
    if not text:
        return "Unknown"
    for label, tokens in VARIETY_PATTERNS:
        if any(tok in text for tok in tokens):
            return label
    if any(name.lower() in text for name in _RECORDED):
        return "Named Variety"
    return "Other"


def crop_group(crop: object) -> str:
    """Broad commodity grouping used for one-hot encoding and EDA rollups."""
    if _is_missing(crop):
        return "Unknown"
    return CROP_GROUP.get(" ".join(str(crop).strip().split()).title(), "Other")


# --------------------------------------------------------------------------
# Quality report
# --------------------------------------------------------------------------
@dataclass
class QualityReport:
    """Diagnostics produced while normalising a raw frame."""

    rows_in: int = 0
    rows_out: int = 0
    resolved_columns: dict[str, str] = field(default_factory=dict)
    unmapped_columns: list[str] = field(default_factory=list)
    missing_required: list[str] = field(default_factory=list)
    coerced_numeric: dict[str, int] = field(default_factory=dict)
    dropped_rows: dict[str, int] = field(default_factory=dict)
    normalised_units: dict[str, int] = field(default_factory=dict)
    string_trims: int = 0
    case_fixes: int = 0
    unit_fallbacks: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def rows_dropped_total(self) -> int:
        return self.rows_in - self.rows_out

    def to_dict(self) -> dict:
        return {
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "rows_dropped_total": self.rows_dropped_total,
            "resolved_columns": self.resolved_columns,
            "unmapped_columns": self.unmapped_columns,
            "missing_required": self.missing_required,
            "coerced_numeric": self.coerced_numeric,
            "dropped_rows": self.dropped_rows,
            "normalised_units": self.normalised_units,
            "string_trims": self.string_trims,
            "case_fixes": self.case_fixes,
            "unit_fallbacks": self.unit_fallbacks,
            "notes": self.notes,
        }

    def summary_line(self) -> str:
        return (
            f"{self.rows_in:,} rows in -> {self.rows_out:,} canonical rows "
            f"({self.rows_dropped_total:,} dropped); "
            f"{len(self.resolved_columns)} columns mapped"
        )


def describe_schema() -> str:
    """Human-readable schema table used by the UI and README generation."""
    rows = [
        ("crop", "string", "Name of the crop (Rice, Wheat, Cotton, ...)"),
        ("variety", "string", "Subsidiary / cultivar name"),
        ("state", "string", "Indian state or union territory"),
        ("quantity", "float", "Cultivated area or volume, in `unit`"),
        ("production", "float", "Observed output, in quintals"),
        ("season", "string", "Raw season / duration label"),
        ("unit", "string", "Measurement unit (Quintals, Tons, ...)"),
        ("cost", "float", "Total cost of cultivation and production, INR"),
        ("recommended_zone", "string", "Zone recommendation scope and name"),
        ("year", "int", "Calendar year of cultivation"),
    ]
    width = max(len(r[0]) for r in rows)
    lines = [f"{'column'.ljust(width)} | {'dtype'.ljust(7)} | description",
             "-" * (width + 34)]
    for name, dtype, desc in rows:
        lines.append(f"{name.ljust(width)} | {dtype.ljust(7)} | {desc}")
    return "\n".join(lines)
