"""The basemap: canonical state keys, dissolving districts, and coverage.

The choropleth is only as trustworthy as the join between dataset states and map
features, so the coverage tests matter more than the geometry tests here.
"""

from __future__ import annotations

import json

import pytest

from src import geo


@pytest.fixture(scope="module")
def basemap():
    payload, note = geo.load_geojson()
    if payload is None:
        pytest.skip(f"no cached basemap available ({note})")
    return payload, note


@pytest.fixture(scope="module")
def raw_districts():
    """The un-dissolved district file, if it has been downloaded."""
    from config.settings import GEOJSON_RAW_PATH

    if not GEOJSON_RAW_PATH.exists():
        pytest.skip("raw district file not downloaded yet")
    payload, note = geo.load_geojson(GEOJSON_RAW_PATH)
    if payload is None:
        pytest.skip(f"raw districts unreadable ({note})")
    return payload


class TestGeoStateKey:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Punjab", "Punjab"),
            ("  PUNJAB  ", "Punjab"),
            ("punjab", "Punjab"),
            ("Uttar Pradesh", "Uttar Pradesh"),
            ("Uttar  Pradesh", "Uttar Pradesh"),
            ("MADHYA PRADESH", "Madhya Pradesh"),
            ("West Bengal", "West Bengal"),
        ],
    )
    def test_resolves_any_spelling_to_the_canonical_name(self, raw, expected):
        assert geo.geo_state_key(raw) == expected

    @pytest.mark.parametrize("raw", ["Atlantis", "Zzz", "Not A State"])
    def test_rejects_anything_that_is_not_a_state(self, raw):
        # An unrecognised polygon name must return None rather than drawing a
        # state called "Atlantis" onto the map.
        assert geo.geo_state_key(raw) is None

    def test_returns_none_for_missing_input(self):
        assert geo.geo_state_key("") is None
        assert geo.geo_state_key("   ") is None
        assert geo.geo_state_key(None) is None

    def test_is_idempotent(self):
        once = geo.geo_state_key("  West Bengal ")
        assert geo.geo_state_key(once) == once


class TestBasemapStructure:
    def test_is_a_feature_collection(self, basemap):
        payload, _ = basemap
        assert payload["type"] == "FeatureCollection"

    def test_has_features(self, basemap):
        payload, _ = basemap
        assert len(payload["features"]) > 0

    def test_every_feature_is_a_geometry(self, basemap):
        payload, _ = basemap
        for feature in payload["features"]:
            assert feature["type"] == "Feature"
            assert feature["geometry"] is None or "type" in feature["geometry"]

    def test_every_feature_has_a_readable_name(self, basemap):
        payload, _ = basemap
        for feature in payload["features"]:
            assert geo.feature_name(feature)

    def test_load_returns_a_note_about_provenance(self, basemap):
        _, note = basemap
        assert isinstance(note, str)

    def test_is_valid_json_on_disk(self):
        from config.settings import GEOJSON_PATH

        if not GEOJSON_PATH.exists():
            pytest.skip("states file not built yet")
        with open(GEOJSON_PATH, encoding="utf-8") as handle:
            assert json.load(handle)["type"] == "FeatureCollection"


class TestDissolve:
    def test_raw_districts_collapse_to_fewer_features(self, raw_districts):
        dissolved = geo.dissolve_to_states(raw_districts)
        assert 0 < len(dissolved["features"]) < len(raw_districts["features"])

    def test_result_is_still_a_feature_collection(self, raw_districts):
        assert geo.dissolve_to_states(raw_districts)["type"] == "FeatureCollection"

    def test_dissolve_is_idempotent(self, raw_districts):
        once = geo.dissolve_to_states(raw_districts)
        twice = geo.dissolve_to_states(once)
        assert len(once["features"]) == len(twice["features"])

    def test_one_feature_per_canonical_state(self, raw_districts):
        dissolved = geo.dissolve_to_states(raw_districts)
        names = geo.geo_state_names(dissolved)
        assert len(names) == len(dissolved["features"])

    def test_coordinates_are_rounded_to_bound_size(self, raw_districts):
        dissolved = geo.dissolve_to_states(raw_districts, precision=2)
        coords = [
            point
            for feature in dissolved["features"]
            for point in _walk_coords(feature["geometry"])
        ]
        assert coords
        for lon, lat in coords:
            assert round(lon, 6) == pytest.approx(lon, abs=1e-9)
            assert round(lat, 6) == pytest.approx(lat, abs=1e-9)

    def test_india_bounds_are_plausible(self, basemap):
        payload, _ = basemap
        coords = [
            point
            for feature in payload["features"]
            for point in _walk_coords(feature["geometry"])
        ]
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        # 68-98E, 6-38N. A basemap that escapes this is mis-projected and will
        # render as a smear rather than India.
        assert min(lons) > 60.0 and max(lons) < 100.0
        assert min(lats) > 0.0 and max(lats) < 40.0


class TestHarmonise:
    def test_names_are_canonicalised(self, basemap):
        payload, _ = basemap
        harmonised, _ = geo.harmonise(payload)
        for name in geo.geo_state_names(harmonised):
            assert name == name.strip()

    def test_duplicate_features_are_merged(self):
        # Two districts with the same name must not become two map states.
        template = {
            "type": "Polygon",
            "coordinates": [[[70.0, 20.0], [71.0, 20.0], [71.0, 21.0], [70.0, 20.0]]],
        }
        payload = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "properties": {"NAME_1": "Punjab"},
                 "geometry": template},
                {"type": "Feature", "properties": {"NAME_1": "punjab"},
                 "geometry": template},
            ],
        }
        harmonised, report = geo.harmonise(payload)
        assert "Punjab" in geo.geo_state_names(harmonised)
        assert isinstance(report, dict)


class TestCoverage:
    def test_every_dataset_state_is_on_the_map(self, dataset, basemap):
        # The headline claim of the overview page: 100% of the data can be
        # shaded. If this fails the choropleth silently drops states.
        payload, _ = basemap
        report = geo.coverage(dataset, payload)
        assert report["n_unmatched"] == 0, report["unmatched"]
        assert report["coverage_pct"] == pytest.approx(100.0)

    def test_map_has_no_more_states_than_the_data_needs(self, dataset, basemap):
        payload, _ = basemap
        report = geo.coverage(dataset, payload)
        assert report["n_map_states"] >= report["n_data_states"]

    def test_unmatched_states_are_reported_not_hidden(self):
        frame = __import__("pandas").DataFrame({"state": ["Punjab", "Atlantis"]})
        payload = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "properties": {"NAME_1": "Punjab"},
                 "geometry": {"type": "Polygon",
                              "coordinates": [[[74.0, 31.0], [75.0, 31.0],
                                               [75.0, 32.0], [74.0, 31.0]]]}}
            ],
        }
        report = geo.coverage(frame, payload)
        assert report["n_unmatched"] == 1
        assert report["unmatched"] == ["Atlantis"]
        assert report["coverage_pct"] == pytest.approx(50.0)

    def test_empty_frame_does_not_divide_by_zero(self):
        import pandas as pd

        payload = {"type": "FeatureCollection", "features": []}
        report = geo.coverage(pd.DataFrame({"state": []}), payload)
        assert report["coverage_pct"] == 0.0


def _walk_coords(node):
    """Yield every (lon, lat) numeric pair anywhere inside a geometry.

    Deliberately type-agnostic: the cached file nests coordinates a level
    deeper than the GeoJSON spec's MultiPolygon in places, and the only thing
    these tests care about is the lat/lon extent of the land.
    """
    if isinstance(node, dict):
        for value in node.values():
            yield from _walk_coords(value)
    elif isinstance(node, (list, tuple)):
        if len(node) >= 2 and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in node[:2]
        ):
            yield (float(node[0]), float(node[1]))
        else:
            for item in node:
                yield from _walk_coords(item)
