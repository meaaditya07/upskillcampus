"""India state boundary GeoJSON: fetch, cache and harmonise.

The choropleth on the overview page needs state polygons. Plotly ships no
built-in India state layer, so a public GeoJSON is downloaded once and cached
under ``data/geo/``. If the download fails the dashboard degrades to a ranked
bar chart rather than breaking, so nothing here is allowed to raise on a
network problem.

The awkward part of any India GeoJSON is state naming: files disagree with
Census spellings, with each other, and with the dataset. :func:`geo_state_key`
maps all of them onto one canonical key so a choropleth never silently
renders an empty state.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from config.settings import GEOJSON_PATH, GEOJSON_RAW_PATH
from src.schema import canonical_state, is_known_state

GEOJSON_SOURCES = (
    "https://raw.githubusercontent.com/udit-001/india-maps-data/main/geojson/india.geojson",
    "https://raw.githubusercontent.com/geohacker/india/master/state/india_state.geojson",
)

# Property keys seen across the candidate sources, in priority order. `st_nm`
# is checked first because the widely-mirrored india-maps-data file is
# district-level and carries the state name there.
NAME_KEYS = (
    "st_nm", "ST_NM", "state_name", "STATE_NAME", "NAME_1", "name_1",
    "shape_lname", "NAME", "name",
)


def geo_state_key(name: str) -> str | None:
    """Map any spelling of an Indian state onto its canonical name.

    Returns ``None`` for anything that is not a real state/UT, so an
    unrecognised polygon name is skipped instead of being drawn as a state
    called "Zzz".
    """
    if name is None:
        return None
    text = str(name).strip()
    if not text or not is_known_state(text):
        return None
    return canonical_state(text) or None


def build_basemap(*, force: bool = False, timeout: int = 60) -> tuple[bool, str]:
    """Ensure a state-level basemap is cached at :data:`GEOJSON_PATH`.

    Downloads are cached raw at :data:`GEOJSON_RAW_PATH` so a re-run does not
    re-hit the network, while the dissolved state-level file is what the app
    actually loads.
    """
    if GEOJSON_PATH.exists() and not force:
        payload, message = load_geojson(GEOJSON_PATH)
        if payload is not None:
            return True, f"cached {message} at {GEOJSON_PATH.name}"

    raw_payload = None
    if GEOJSON_RAW_PATH.exists() and not force:
        raw_payload, _ = load_geojson(GEOJSON_RAW_PATH)

    if raw_payload is None:
        ok, message = download_geojson(timeout=timeout)
        if not ok:
            return False, message

    raw_payload, message = load_geojson(GEOJSON_RAW_PATH)
    if raw_payload is None:
        return False, message

    dissolved = dissolve_to_states(raw_payload)
    if not dissolved["features"]:
        return False, "no state polygons could be extracted from the GeoJSON"

    GEOJSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    GEOJSON_PATH.write_text(json.dumps(dissolved), encoding="utf-8")
    summary = (
        f"{len(dissolved['features'])} states, "
        f"{GEOJSON_PATH.stat().st_size / 1e6:.2f} MB (from {message})"
    )
    return True, summary


def download_geojson(*, timeout: int = 60) -> tuple[bool, str]:
    """Fetch the raw India GeoJSON into :data:`GEOJSON_RAW_PATH`."""
    try:
        import requests
    except ImportError:
        return False, "requests is not installed"

    errors: list[str] = []
    for url in GEOJSON_SOURCES:
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            if not payload.get("features"):
                errors.append(f"{url}: no features")
                continue
            GEOJSON_RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
            GEOJSON_RAW_PATH.write_text(json.dumps(payload), encoding="utf-8")
            return True, f"downloaded {len(payload['features'])} features from {url}"
        except Exception as exc:
            errors.append(f"{url.rsplit('/', 2)[-2]}: {type(exc).__name__}")
            continue

    return False, "could not fetch India GeoJSON (" + "; ".join(errors) + ")"


def load_geojson(path: Path | None = None) -> tuple[dict | None, str]:
    """Read and structurally validate a GeoJSON file."""
    target = Path(path) if path else GEOJSON_PATH
    if not target.exists():
        return None, f"{target} not found -- run scripts/bootstrap.py"
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception as exc:
        return None, f"unreadable GeoJSON: {type(exc).__name__}"
    if not isinstance(payload, dict) or not payload.get("features"):
        return None, "GeoJSON contains no features"
    return payload, f"{len(payload['features'])} features"


def feature_name(feature: dict) -> str | None:
    """Pull the human-readable state name out of a GeoJSON feature."""
    properties = feature.get("properties") or {}
    for key in NAME_KEYS:
        value = properties.get(key)
        if value:
            return str(value)
    return None


def harmonise(payload: dict) -> tuple[dict, dict]:
    """Annotate each feature with a canonical ``canonical_name``.

    Returns the payload and a mapping of unmatched raw names, so the UI can
    report which polygons it could not align with the dataset instead of
    quietly drawing nothing for them.
    """
    unmatched: dict[str, int] = {}
    for feature in payload.get("features", []):
        raw = feature_name(feature)
        canonical = geo_state_key(raw) if raw else None
        properties = feature.setdefault("properties", {})
        if canonical:
            properties["canonical_name"] = canonical
        else:
            properties["canonical_name"] = raw
            if raw:
                unmatched[raw] = unmatched.get(raw, 0) + 1
    return payload, unmatched


def geo_state_names(payload: dict) -> set[str]:
    """Canonical state names present in the basemap."""
    names = set()
    for feature in payload.get("features", []):
        properties = feature.get("properties") or {}
        canonical = properties.get("canonical_name") or feature_name(feature)
        mapped = geo_state_key(canonical) if canonical else None
        if mapped:
            names.add(mapped)
    return names


def dissolve_to_states(payload: dict, precision: int = 3) -> dict:
    """Collapse district features into one MultiPolygon feature per state.

    The readily-mirrored India GeoJSON is district-level -- 760 features and
    ~4 MB, which is heavy to ship to a browser on every page render. Grouping
    the rings by state and rounding coordinates to ``precision`` decimal
    degrees (about 100 m, far below any choropleth's visible resolution) cuts
    that to a fraction of the size and gives one feature per state, which is
    what a state-level choropleth actually wants.
    """
    polygons: dict[str, list] = defaultdict(list)

    for feature in payload.get("features", []):
        properties = feature.get("properties") or {}
        raw = None
        for key in NAME_KEYS:
            if properties.get(key):
                raw = properties[key]
                break
        canonical = geo_state_key(raw) if raw else None
        if not canonical:
            continue
        geometry = feature.get("geometry") or {}
        gtype = geometry.get("type")
        coordinates = geometry.get("coordinates") or []
        if gtype == "Polygon" or gtype == "MultiPolygon":
            polygons[canonical].extend(coordinates)

    def round_coords(obj):
        if isinstance(obj, (list, tuple)):
            if obj and isinstance(obj[0], (int, float)):
                return [round(float(obj[0]), precision), round(float(obj[1]), precision)]
            return [round_coords(item) for item in obj]
        return obj

    features = []
    for state in sorted(polygons):
        rings = [round_coords(ring) for ring in polygons[state] if ring]
        if not rings:
            continue
        features.append(
            {
                "type": "Feature",
                "properties": {"canonical_name": state, "name": state},
                "geometry": {"type": "MultiPolygon", "coordinates": [rings]},
            }
        )

    return {"type": "FeatureCollection", "features": features}


def coverage(frame, payload: dict) -> dict:
    """How well the dataset's states align with the basemap."""
    map_names = geo_state_names(payload)
    data_states = {
        str(s).strip() for s in frame["state"].dropna().unique()
    }
    matched = sorted(s for s in data_states if s in map_names)
    unmatched = sorted(s for s in data_states if s not in map_names)
    return {
        "n_data_states": len(data_states),
        "n_map_states": len(map_names),
        "n_matched": len(matched),
        "n_unmatched": len(unmatched),
        "matched": matched,
        "unmatched": unmatched,
        "coverage_pct": (len(matched) / len(data_states) * 100.0)
        if data_states
        else 0.0,
    }
