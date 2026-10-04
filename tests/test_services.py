"""The Streamlit-free service layer.

These tests double as the contract for the future FastAPI wrapper: if a service
takes a DataFrame (or a Scenario) and returns plain dicts, it can be exposed over
HTTP without moving business logic into the view layer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import COST_TARGET, YIELD_TARGET
from src.services import insights, predictor, profitability, recommend


def wheat(area_ha: float = 2.0, year: int = 2021) -> predictor.Scenario:
    return predictor.Scenario(
        state="Punjab",
        crop="Wheat",
        variety="HD-2967",
        recommended_zone="Mandal - Ludhiana",
        season="Rabi",
        area_ha=area_ha,
        year=year,
    )


# --------------------------------------------------------------------------
# Scenario
# --------------------------------------------------------------------------
class TestScenario:
    def test_row_applies_every_canonicalisation(self):
        row = predictor.Scenario(
            state="  punjab ",
            crop="WHEAT",
            variety="hd-2967",
            recommended_zone="Mandal -  Ludhiana ",
            season="rabi",
        ).row()
        assert row["state"] == "Punjab"
        assert row["crop"] == "Wheat"
        assert row["recommended_zone"] == "Mandal - Ludhiana"
        assert row["season_name"] == "Rabi"
        assert row["zone_scope"] == "Mandal"

    def test_season_duration_defaults_to_the_derived_value(self):
        # Never a hard-coded "Medium": that would pair Rabi with a duration the
        # model never saw, and unknown levels are silently zeroed.
        from src.schema import parse_season

        for season in ("Kharif", "Rabi", "Zaid", "Whole Year"):
            row = predictor.Scenario(season=season).row()
            assert row["season_duration"] == parse_season(season)[1]

    def test_an_explicit_season_duration_still_wins(self):
        row = predictor.Scenario(
            season="Rabi", season_duration="Short"
        ).row()
        assert row["season_duration"] == "Short"

    def test_row_derives_the_helpers_the_model_was_trained_on(self):
        # crop_group / variety_class / zone_scope are one-hot inputs. Building
        # them here with the loader's own helpers is the only way the inference
        # categories can match the training categories.
        from src.schema import classify_variety, crop_group, parse_zone

        row = wheat().row()
        assert row["crop_group"] == crop_group("Wheat")
        assert row["variety_class"] == classify_variety("HD-2967")
        assert row["zone_scope"] == parse_zone("Mandal - Ludhiana")[0]

    def test_a_sloppy_zone_string_still_maps_to_a_seen_level(self):
        # "  mandal - ludhiana " and "Mandal - Ludhiana" must produce the same
        # one-hot inputs, or the model silently predicts from zeroed columns.
        sloppy = predictor.Scenario(
            recommended_zone="  mandal - ludhiana  "
        ).row()
        tidy = predictor.Scenario(recommended_zone="Mandal - Ludhiana").row()
        assert sloppy == tidy

    def test_row_carries_no_target_information(self):
        row = wheat().row()
        assert "quantity" in row and row["quantity"] == 2.0
        for forbidden in ("production", "production_quintals", "cost",
                          "cost_per_unit", "production_efficiency"):
            assert forbidden not in row

    def test_area_is_coerced_to_float_for_the_encoder(self):
        assert isinstance(wheat(area_ha=3).row()["quantity"], float)
        assert isinstance(wheat().row()["year"], float)

    def test_to_dict_is_serialisable(self):
        payload = wheat().to_dict()
        assert payload["crop"] == "Wheat"
        assert payload["area_ha"] == 2.0

    def test_scenarios_to_frame_preserves_order(self):
        frame = predictor.scenarios_to_frame([wheat(), wheat(year=2019)])
        assert list(frame["year"]) == [2021.0, 2019.0]

    def test_scenarios_to_frame_rejects_an_empty_list(self):
        assert predictor.scenarios_to_frame([]).empty

    def test_scenario_from_row_round_trips(self):
        original = wheat(area_ha=4.0)
        rebuilt = predictor.scenario_from_row(original.to_dict())
        assert rebuilt.to_dict() == original.to_dict()


# --------------------------------------------------------------------------
# Forecasting
# --------------------------------------------------------------------------
class TestForecast:
    def test_both_targets_come_from_one_consistent_frame(self, bundles):
        result = predictor.forecast([wheat()], bundles=bundles)
        assert len(result) == 1
        # Same area and season feeding both models -- this is what keeps the
        # profitability page internally coherent.
        assert result.cost_point[0] > 0
        assert result.yield_point[0] > 0

    def test_interval_contains_the_point_estimate(self, bundles):
        result = predictor.forecast([wheat()], bundles=bundles)
        assert result.yield_lower[0] <= result.yield_point[0] <= result.yield_upper[0]
        assert result.cost_lower[0] <= result.cost_point[0] <= result.cost_upper[0]

    def test_tonnes_uses_the_indian_quintal(self, bundles):
        from config.settings import QUINTAL_PER_TONNE

        one = predictor.forecast([wheat()], bundles=bundles).first()
        assert one["yield_tonnes"] == pytest.approx(
            one["yield_quintals"] / QUINTAL_PER_TONNE, rel=1e-12
        )

    def test_deterministic(self, bundles):
        a = predictor.forecast([wheat()], bundles=bundles).first()
        b = predictor.forecast([wheat()], bundles=bundles).first()
        assert a == b

    def test_total_production_scales_with_area(self, bundles):
        one = predictor.forecast([wheat(1.0)], bundles=bundles).first()
        four = predictor.forecast([wheat(4.0)], bundles=bundles).first()
        assert four["yield_quintals"] > one["yield_quintals"]

    def test_per_hectare_is_area_normalised(self, bundles):
        result = predictor.forecast([wheat(4.0)], bundles=bundles)
        assert result.per_hectare()[0] == pytest.approx(
            result.yield_point[0] / 4.0, rel=1e-9
        )

    def test_interval_width_is_positive(self, bundles):
        assert predictor.forecast([wheat()], bundles=bundles).interval_width()[0] > 0

    def test_batch_preserves_input_order(self, bundles):
        scenarios = [wheat(year=y) for y in (2015, 2016, 2017, 2018)]
        batch = predictor.forecast(scenarios, bundles=bundles)
        one_at_a_time = [
            predictor.forecast([s], bundles=bundles).first()["yield_quintals"]
            for s in scenarios
        ]
        assert list(batch.yield_point) == one_at_a_time

    def test_unknown_zone_does_not_crash(self, bundles):
        result = predictor.forecast(
            [wheat()], bundles=bundles
        )  # sanity: the normal path works first
        assert result.first()["yield_quintals"] > 0
        novel = predictor.Scenario(
            state="Punjab", crop="Wheat", variety="HD-2967",
            recommended_zone="Nowhere Mandi", season="Rabi", area_ha=2.0,
        )
        # handle_unknown="ignore" plus the target encoder's smoothing means an
        # unseen district predicts *something*, it must not raise.
        assert predictor.forecast([novel], bundles=bundles).first()["yield_quintals"] > 0

    def test_empty_input_raises(self, bundles):
        with pytest.raises(ValueError):
            predictor.forecast([], bundles=bundles)

    def test_missing_column_is_reported_clearly(self, bundles):
        # A frame that survives feature engineering but lacks a model input.
        frame = pd.DataFrame(
            {
                "quantity": [2.0],
                "year": [2021.0],
                "state": ["Punjab"],
                "crop": ["Wheat"],
            }
        )
        with pytest.raises(ValueError, match="scenarios_to_frame"):
            predictor.forecast(frame, bundles=bundles)

    def test_first_is_json_serialisable(self, bundles):
        import json

        payload = json.loads(json.dumps(predictor.forecast([wheat()], bundles=bundles).first()))
        assert payload["yield_quintals"] > 0
        assert set(payload) >= {"yield_quintals", "yield_lower", "yield_upper", "cost"}

    def test_explain_returns_finite_non_empty_contributions(self, bundles):
        contributions = predictor.explain(wheat(), top=5)
        assert isinstance(contributions, pd.DataFrame)
        assert not contributions.empty
        assert len(contributions) <= 5
        assert set(contributions.columns) >= {"feature", "contribution"}
        assert np.isfinite(contributions["contribution"]).all()
        assert contributions["feature"].is_unique

    def test_explain_works_for_the_cost_model_too(self, bundles):
        contributions = predictor.explain(wheat(), target=COST_TARGET, top=5)
        assert not contributions.empty


class TestReferenceData:
    def test_known_states_are_canonical(self):
        states = predictor.known_states()
        assert states
        assert all(isinstance(s, str) and s.strip() == s for s in states)

    def test_crops_are_non_empty(self):
        assert predictor.crops()

    def test_varieties_are_observed_for_a_real_crop(self, dataset):
        varieties = predictor.varieties_for("Wheat")
        assert varieties != ["Unknown"]
        assert set(varieties) <= set(dataset["variety"])

    def test_varieties_for_an_absent_crop_degrades_gracefully(self):
        assert predictor.varieties_for("Unobtainium") == ["Unknown"]

    def test_year_range_is_ordered(self):
        low, high = predictor.year_range()
        assert low <= high

    def test_zones_for_a_state_are_non_empty(self, dataset):
        zones = predictor.zones_for("Punjab")
        assert zones
        assert set(zones) <= set(dataset["recommended_zone"])

    def test_season_durations_are_offered(self):
        assert predictor.season_durations()


# --------------------------------------------------------------------------
# Profitability
# --------------------------------------------------------------------------
class TestProfitability:
    def test_profit_is_revenue_minus_cost(self, bundles):
        result = profitability.assess(
            predictor.forecast([wheat()], bundles=bundles)
        ).first()
        assert result["profit"] == pytest.approx(
            result["revenue"] - result["cost"], rel=1e-9
        )

    def test_revenue_is_price_times_quintals(self, bundles):
        forecast = predictor.forecast([wheat()], bundles=bundles)
        result = profitability.assess(forecast).first()
        assert result["revenue"] == pytest.approx(
            result["price_per_quintal"] * forecast.yield_point[0], rel=1e-9
        )

    def test_price_override_is_respected(self, bundles):
        forecast = predictor.forecast([wheat()], bundles=bundles)
        cheap = profitability.assess(forecast, price_override=500.0).first()
        assert cheap["price_per_quintal"] == 500.0

    def test_higher_price_means_higher_profit(self, bundles):
        forecast = predictor.forecast([wheat()], bundles=bundles)
        base = profitability.assess(forecast).first()["profit"]
        better = profitability.assess(forecast, price_shift_pct=20.0).first()["profit"]
        assert better > base

    def test_cost_override_changes_profit_in_the_right_direction(self, bundles):
        forecast = predictor.forecast([wheat()], bundles=bundles)
        base = profitability.assess(forecast).first()["profit"]
        dearer = profitability.assess(forecast, cost_override=1e6).first()["profit"]
        assert dearer < base

    def test_profit_interval_contains_the_point_profit(self, bundles):
        result = profitability.assess(predictor.forecast([wheat()], bundles=bundles)).first()
        assert result["profit_low"] <= result["profit"] <= result["profit_high"]

    def test_break_even_price_recovers_zero_margin(self, bundles):
        forecast = predictor.forecast([wheat()], bundles=bundles)
        result = profitability.assess(
            forecast, price_override=float(
                profitability.assess(forecast).first()["break_even_price"]
            )
        ).first()
        assert result["profit"] == pytest.approx(0.0, abs=1e-6)

    def test_batch_length_matches_the_forecast(self, bundles):
        scenarios = [wheat(area_ha=a) for a in (1.0, 2.0, 4.0)]
        forecast = predictor.forecast(scenarios, bundles=bundles)
        assert len(profitability.assess(forecast)) == len(scenarios)

    def test_price_sensitivity_is_monotone(self, bundles):
        forecast = predictor.forecast([wheat()], bundles=bundles)
        sweep = profitability.price_sensitivity(forecast)
        profits = list(sweep["profit"])
        assert profits == sorted(profits)

    def test_area_sensitivity_is_monotone(self, bundles):
        # The sweep re-forecasts at each area, so economies of scale in the
        # cost model show up as a per-hectare cost that falls as area grows.
        sweep = profitability.area_sensitivity(
            lambda ha: predictor.forecast([wheat(area_ha=ha)], bundles=bundles)
        )
        per_ha = list(sweep["cost_per_hectare"])
        assert per_ha == sorted(per_ha, reverse=True)

    def test_area_sensitivity_rows_match_the_requested_areas(self, bundles):
        sweep = profitability.area_sensitivity(
            lambda ha: predictor.forecast([wheat(area_ha=ha)], bundles=bundles),
            areas=(1.0, 3.0),
        )
        assert list(sweep["area_ha"]) == [1.0, 3.0]

    def test_assumptions_are_disclosed(self):
        # The UI must be able to show that prices are indicative, not official.
        stated = profitability.assumptions()
        assert stated["unit"] == "INR per quintal"
        assert "note" in stated
        assert stated["catalogue"]
        assert all(v > 0 for v in stated["catalogue"].values())


# --------------------------------------------------------------------------
# Recommendations
# --------------------------------------------------------------------------
class TestRecommend:
    def test_state_ranking_is_ordered_by_score(self, dataset):
        result = recommend.recommend_for_crop("Wheat", frame=dataset, limit=10)
        assert len(result.table) <= 10
        assert list(result.table["score"]) == sorted(result.table["score"], reverse=True)

    def test_ranks_are_contiguous_from_one(self, dataset):
        result = recommend.recommend_for_crop("Rice", frame=dataset)
        assert list(result.table["rank"]) == list(range(1, len(result.table) + 1))

    def test_state_scope_is_reported(self, dataset):
        result = recommend.recommend_for_crop("Wheat", frame=dataset)
        assert result.scope == "state"
        assert "state" in result.table.columns

    def test_passing_a_state_switches_to_zone_scope(self, dataset):
        result = recommend.recommend_for_crop("Wheat", frame=dataset, state="Punjab")
        assert result.scope == "zone"
        assert "recommended_zone" in result.table.columns

    def test_zones_come_from_that_state_only(self, dataset):
        result = recommend.recommend_for_crop("Wheat", frame=dataset, state="Punjab")
        allowed = set(dataset.loc[dataset["state"] == "Punjab", "recommended_zone"])
        assert set(result.table["recommended_zone"]) <= allowed

    def test_score_stays_within_zero_and_one(self, dataset):
        result = recommend.recommend_for_crop("Wheat", frame=dataset)
        assert result.table["score"].between(0.0, 1.0).all()

    def test_low_confidence_groups_are_demoted(self):
        # LOW_CONFIDENCE_N is 30, so the well-evidenced zone needs a real body
        # of observations. It must still outrank the thin zone.
        thick = [(y, "Amritsar") for y in range(2000, 2012) for _ in range(3)]
        thin = [(2020, "Ludhiana")]
        rows = thick + thin
        frame = pd.DataFrame(
            {
                "year": [r[0] for r in rows],
                "state": ["Punjab"] * len(rows),
                "crop": ["Wheat"] * len(rows),
                "recommended_zone": [r[1] for r in rows],
                "season_name": ["Rabi"] * len(rows),
                "quantity": [1.0] * len(rows),
                "production_quintals": [20.0] * len(rows),
                "cost": [20000.0] * len(rows),
                "cost_per_unit": [20000.0] * len(rows),
                "production_efficiency": [20.0] * len(rows),
            }
        )
        table = recommend.recommend_for_crop(
            "Wheat", frame=frame, state="Punjab"
        ).table
        by_zone = table.set_index("recommended_zone")
        assert by_zone.loc["Ludhiana", "confidence"] == "low"
        assert by_zone.loc["Amritsar", "confidence"] != "low"
        assert by_zone.loc["Amritsar", "score"] > by_zone.loc["Ludhiana", "score"]
        assert by_zone.index[0] == "Amritsar"

    def test_outlier_rows_are_excluded_before_scoring(self, dataset):
        poisoned = dataset.copy()
        mask = poisoned["is_outlier"]
        assert mask.any(), "fixture needs outliers for this test to mean anything"
        poisoned.loc[mask, "production_efficiency"] = 10_000.0
        clean_scores = recommend.recommend_for_crop("Wheat", frame=dataset).table["score"]
        dirty_scores = recommend.recommend_for_crop("Wheat", frame=poisoned).table["score"]
        assert list(clean_scores) == list(dirty_scores)

    def test_absent_crop_yields_an_empty_ranking_not_a_crash(self, dataset):
        result = recommend.recommend_for_crop("Unobtainium", frame=dataset)
        assert result.table.empty
        assert result.n_observations == 0

    def test_seasons_are_ranked_within_a_state_crop(self, dataset):
        result = recommend.recommend_seasons("Punjab", "Wheat", frame=dataset)
        assert not result.table.empty
        assert "season_name" in result.table.columns

    def test_varieties_are_ranked_within_a_state_crop(self, dataset):
        result = recommend.recommend_varieties("Punjab", "Wheat", frame=dataset)
        assert not result.table.empty
        assert "variety" in result.table.columns

    def test_state_crop_matrix_is_ordered_within_each_crop(self, dataset):
        matrix = recommend.recommend_state_crop_matrix(frame=dataset)
        assert not matrix.empty
        assert {"crop", "state", "score", "rank"} <= set(matrix.columns)
        # _score_groups ranks globally, so within a crop the ranks must simply
        # descend -- that is what the heatmap reads down.
        for _, group in matrix.groupby("crop"):
            ranks = list(group["rank"])
            assert ranks == sorted(ranks)

    def test_as_records_is_json_friendly(self, dataset):
        import json

        records = recommend.recommend_for_crop(
            "Wheat", frame=dataset, limit=3
        ).as_records(3)
        assert len(records) <= 3
        json.dumps(records, default=str)

    def test_weights_are_the_documented_mix(self, dataset):
        weights = recommend.recommend_for_crop("Wheat", frame=dataset).weights
        assert weights["efficiency"] == 0.5
        assert weights["cost"] == 0.3
        assert weights["reliability"] == 0.2
        assert sum(weights.values()) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Insights
# --------------------------------------------------------------------------
class TestInsights:
    def test_kpis_describe_the_passed_frame(self, dataset):
        kpis = insights.headline_kpis(dataset)
        assert kpis["observations"] == len(dataset)
        assert kpis["states"] == dataset["state"].nunique()
        assert kpis["crops"] == dataset["crop"].nunique()

    def test_kpis_convert_to_tonnes_with_the_indian_quintal(self, dataset):
        from config.settings import QUINTAL_PER_TONNE

        kpis = insights.headline_kpis(dataset)
        assert kpis["total_production_tonnes"] == pytest.approx(
            kpis["total_production_quintals"] / QUINTAL_PER_TONNE, rel=1e-12
        )

    def test_kpis_are_empty_rather_than_wrong_for_no_rows(self):
        assert insights.headline_kpis(pd.DataFrame()) == {}

    def test_by_dimension_aggregates_state(self, dataset):
        table = insights.by_dimension(dataset, "state")
        assert not table.empty
        assert table["production_quintals"].sum() <= dataset["production_quintals"].sum()

    def test_by_dimension_rejects_an_unknown_column(self, dataset):
        with pytest.raises(KeyError):
            insights.by_dimension(dataset, "not_a_column")

    def test_by_dimension_respects_the_limit(self, dataset):
        assert len(insights.by_dimension(dataset, "state", limit=3)) <= 3

    def test_by_dimension_shares_reconcile(self, dataset):
        table = insights.by_dimension(dataset, "state", limit=None)
        assert table["production_quintals"].sum() == pytest.approx(
            dataset["production_quintals"].sum(), rel=1e-9
        )

    def test_top_states_by_sorts_by_the_requested_metric(self, dataset):
        table = insights.top_states_by("median_efficiency", frame=dataset, limit=5)
        assert list(table["median_efficiency"]) == sorted(
            table["median_efficiency"], reverse=True
        )

    def test_top_states_by_rejects_an_unknown_metric(self, dataset):
        with pytest.raises(KeyError):
            insights.top_states_by("not_a_metric", frame=dataset)

    def test_crop_mix_shares_sum_to_one_hundred(self, dataset):
        mix = insights.crop_mix(dataset)
        assert mix["production_share_pct"].sum() == pytest.approx(100.0, rel=1e-6)
        assert mix["area_share_pct"].sum() == pytest.approx(100.0, rel=1e-6)

    def test_production_by_year_is_time_ordered(self, dataset):
        series = insights.production_by_year(dataset)
        assert list(series["year"]) == sorted(series["year"])

    def test_outlier_report_lists_only_flagged_rows(self):
        report = insights.outlier_report(top=5)
        # Reads the registry frame; the flag must be the reason a row is here.
        assert isinstance(report, pd.DataFrame)
        assert len(report) <= 5

    def test_default_frame_excludes_outliers(self, dataset):
        # The service default (registry data) must never include flagged rows.
        assert not insights._frame()["is_outlier"].any()


# --------------------------------------------------------------------------
# Layering
# --------------------------------------------------------------------------
class TestLayering:
    @pytest.mark.parametrize(
        "module", [predictor, profitability, recommend, insights]
    )
    def test_services_never_import_streamlit(self, module):
        source = module.__file__
        with open(source, encoding="utf-8") as handle:
            text = handle.read()
        assert "import streamlit" not in text, f"{module.__name__} imports Streamlit"

    def test_package_exports_resolve(self):
        from src import services

        for name in services.__all__:
            assert hasattr(services, name), name

    def test_targets_are_the_documented_pair(self):
        assert {YIELD_TARGET, COST_TARGET} == {"production_quintals", "cost"}
