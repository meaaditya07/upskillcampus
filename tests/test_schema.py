"""Canonicalisation and parsing.

These functions decide what a state, unit or zone *is*, so every downstream
number depends on them. The tests are deliberately adversarial about spelling,
casing and null handling.
"""

from __future__ import annotations

import pytest

from config.settings import KG_PER_QUINTAL, QUINTAL_PER_TONNE, UNIT_TO_QUINTAL_FACTOR
from src.schema import (
    canonical_crop,
    canonical_state,
    canonical_unit,
    classify_variety,
    crop_group,
    is_known_state,
    known_states,
    parse_season,
    parse_zone,
    quintal_factor,
    resolve_columns,
)


class TestCanonicalState:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("uttar pradesh", "Uttar Pradesh"),
            ("  UTTAR PRADESH  ", "Uttar Pradesh"),
            ("Orissa", "Odisha"),
            ("odisha", "Odisha"),
            ("NCT of Delhi", "Delhi"),
            ("New Delhi", "Delhi"),
            ("J&K", "Jammu and Kashmir"),
            ("Jammu & Kashmir", "Jammu and Kashmir"),
            ("Jammu", "Jammu and Kashmir"),
            ("Daman and Diu", "Dadra and Nagar Haveli and Daman and Diu"),
            ("Pondicherry", "Puducherry"),
            ("Uttaranchal", "Uttarakhand"),
            ("Bengal", "West Bengal"),
        ],
    )
    def test_known_spellings_collapse_to_one_name(self, raw, expected):
        assert canonical_state(raw) == expected

    def test_is_idempotent(self):
        for name in known_states():
            assert canonical_state(canonical_state(name)) == name

    def test_whitespace_insensitive(self):
        assert canonical_state("  West   Bengal ") == "West Bengal"

    def test_unknown_name_is_flagged_but_still_displayable(self):
        # An unknown place still needs *a* label, but must not pass as a real
        # state, or the basemap would try to draw "Zzz" as a state.
        assert canonical_state("Zzz") == "Zzz"
        assert is_known_state("Zzz") is False
        assert is_known_state("Odisha") is True

    @pytest.mark.parametrize("blank", ["", "   ", None, "-", "_"])
    def test_blank_is_not_a_state(self, blank):
        assert is_known_state(blank) is False

    def test_every_known_state_is_recognised(self):
        for name in known_states():
            assert is_known_state(name), name

    def test_thirty_six_states_or_uts(self):
        # India has 28 states + 8 union territories.
        assert len(known_states()) == 36


class TestCanonicalCrop:
    def test_synonyms_merge(self):
        assert canonical_crop("Paddy") == "Rice"
        assert canonical_crop("paddy") == canonical_crop("RICE") == "Rice"

    def test_corn_is_maize(self):
        assert canonical_crop("Corn") == canonical_crop("maize")

    def test_unknown_crop_gets_a_title_cased_label(self):
        assert canonical_crop("  quinoa ") == "Quinoa"

    @pytest.mark.parametrize("blank", ["", "   ", None])
    def test_blank_is_empty(self, blank):
        assert canonical_crop(blank) == ""


class TestUnits:
    @pytest.mark.parametrize(
        "raw,label",
        [
            ("Quintal", "Quintals"),
            ("quintals", "Quintals"),
            ("Qtl", "Quintals"),
            ("t", "Tons"),
            ("Tonne", "Tons"),
            ("TONNES", "Tons"),
            ("  tons  ", "Tons"),
            ("Metric Tonnes", "Tons"),
        ],
    )
    def test_spellings_fold_onto_two_labels(self, raw, label):
        # Without this, 'Tons'/'tons'/'TONS'/' Tonne ' each become their own
        # one-hot level and fragment the unit feature.
        assert canonical_unit(raw) == label

    @pytest.mark.parametrize(
        "raw,factor",
        [("quintal", 1.0), ("Tons", 20.0), ("tonne", 20.0)],
    )
    def test_quintal_factor(self, raw, factor):
        assert quintal_factor(raw) == pytest.approx(factor)


class TestUnitConversions:
    """Indian agriculture uses the 50 kg quintal.

    These conversions silently rescale an entire column if wrong, so they are
    pinned against the physical definition rather than against whatever the
    table happens to say.
    """

    def test_tonne_is_twenty_quintals(self):
        assert QUINTAL_PER_TONNE == 20.0
        assert quintal_factor("tonne") == pytest.approx(20.0)

    def test_kilogram_is_two_hundredths_of_a_quintal(self):
        assert KG_PER_QUINTAL == 50.0
        assert quintal_factor("kg") == pytest.approx(0.02)

    def test_one_hundred_kilos_is_exactly_fifty_quintals(self):
        assert quintal_factor("kg") * 100 == pytest.approx(2.0)

    def test_one_tonne_is_exactly_twenty_kilos_of_quintals(self):
        assert quintal_factor("ton") / quintal_factor("kg") == pytest.approx(1000.0)

    def test_cotton_bale_is_three_point_four_quintals(self):
        # 170 kg / 50 kg
        assert quintal_factor("bale") == pytest.approx(3.4)

    def test_every_factor_is_positive(self):
        for unit, factor in UNIT_TO_QUINTAL_FACTOR.items():
            assert factor > 0, unit

    def test_missing_unit_becomes_unknown_not_a_crash(self):
        assert canonical_unit("") == "Unknown"
        assert canonical_unit(None) == "Unknown"
        assert canonical_unit("n/a") == "Unknown"

    def test_unknown_unit_is_title_cased_not_dropped(self):
        assert canonical_unit("bushel") == "Bushel"

    def test_unknown_unit_factor_defaults_to_one(self):
        # Defaulting to 1.0 keeps the row; the loader records it in the audit.
        assert quintal_factor("bushel") == 1.0

    def test_quintal_factor_survives_none(self):
        assert quintal_factor(None) == 1.0


class TestParseSeason:
    @pytest.mark.parametrize(
        "raw,name,duration",
        [
            ("Kharif 2020", "Kharif", "Medium"),
            ("kharif", "Kharif", "Medium"),
            ("Rabi", "Rabi", "Medium"),
            ("Rice (Kharif)", "Kharif", "Medium"),
            ("Zaid", "Zaid", "Short"),
            ("Summer", "Zaid", "Short"),
            ("Whole Year", "Whole Year", "Long"),
            ("Rabi (winter)", "Rabi", "Long"),
        ],
    )
    def test_season_and_duration_split(self, raw, name, duration):
        parsed_name, parsed_duration = parse_season(raw)
        assert parsed_name == name
        assert parsed_duration == duration

    def test_year_is_stripped(self):
        assert parse_season("Kharif 2013")[0] == "Kharif"

    @pytest.mark.parametrize("blank", ["", "   ", None])
    def test_blank_falls_back_without_crashing(self, blank):
        name, duration = parse_season(blank)
        assert isinstance(name, str) and name
        assert isinstance(duration, str) and duration

    def test_unrecognised_season_is_labelled_other_not_dropped(self):
        assert parse_season("Monsoon Special")[0] == "Other"

    def test_synonyms_of_seasons(self):
        assert parse_season("Winter")[0] == "Rabi"
        assert parse_season("Summer")[0] == "Zaid"


class TestParseZone:
    @pytest.mark.parametrize(
        "raw,scope,name",
        [
            ("Mandal - Ludhiana", "Mandal", "Ludhiana"),
            ("Village - Rampur Kalan", "Village", "Rampur Kalan"),
            ("State - Punjab", "State", "Punjab"),
            ("Punjab", "State", "Punjab"),
        ],
    )
    def test_scope_and_name_are_split(self, raw, scope, name):
        assert parse_zone(raw) == (scope, name)

    @pytest.mark.parametrize("dash", ["-", "\u2013", "\u2014"])
    def test_all_dash_separators_parse(self, dash):
        # A mojibake separator previously made en/em dash input fall through to
        # ("State", <whole string>), silently corrupting zone_scope.
        assert parse_zone(f"Mandal {dash} Ludhiana") == ("Mandal", "Ludhiana")

    @pytest.mark.parametrize("blank", ["", "   ", None])
    def test_blank_is_handled(self, blank):
        assert isinstance(parse_zone(blank), tuple)

    def test_district_is_treated_as_mandal_scope(self):
        assert parse_zone("District - Anand")[0] == "Mandal"


class TestClassifyVariety:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Pusa Basmati 1121", "Basmati"),
            ("Cotton Hybrid 548", "Hybrid"),
            ("Local Desi", "Desi"),
            ("HD-2967", "HD Series"),
            ("IR-64", "IR Series"),
            ("LRA-516", "LRA"),
            ("", "Unknown"),
            (None, "Unknown"),
        ],
    )
    def test_classes(self, raw, expected):
        assert classify_variety(raw) == expected

    def test_unknown_variety_still_gets_a_label(self):
        assert isinstance(classify_variety("ZZZ-999"), str)


class TestCropGroup:
    @pytest.mark.parametrize(
        "crop,group",
        [
            ("Rice", "Cereals"),
            ("Tur", "Pulses"),
            ("Cotton", "Cash Crops"),
            ("Soybean", "Oilseeds"),
            ("Onion", "Vegetables"),
        ],
    )
    def test_known_groups(self, crop, group):
        assert crop_group(crop) == group

    def test_unknown_crop_falls_back(self):
        assert crop_group("Quinoa") == "Other"


class TestResolveColumns:
    def test_real_headers_fully_resolve(self):
        headers = ["Crop", "Variety", "State", "Quantity", "production",
                   "Season", "Unit", "Cost", "Recommended Zone", "Year"]
        mapping, unmapped = resolve_columns(headers)
        assert not unmapped, f"unmapped: {unmapped}"
        assert len(mapping) == len(headers)

    def test_lowercase_real_headers_resolve(self):
        mapping, unmapped = resolve_columns(
            ["crop", "variety", "state", "quantity", "production",
             "season", "unit", "cost", "recommended zone", "year"]
        )
        assert not unmapped, f"unmapped: {unmapped}"
        assert mapping["production"] == "production"

    def test_unmapped_columns_are_reported_not_dropped(self):
        mapping, unmapped = resolve_columns({"Crop": "crop", "Weird": "x"})
        assert "Weird" not in mapping.values()
        assert "Weird" in unmapped

    def test_mapping_is_canonical_to_source_direction(self):
        # canonical name -> the column name actually found in the file
        mapping, _ = resolve_columns({"Crop": "crop", "Area": "quantity"})
        assert mapping["crop"] == "Crop"
        assert mapping["quantity"] == "Area"
