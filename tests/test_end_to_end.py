"""End-to-end: CSV in, forecast and recommendation out.

Catches the class of bug unit tests miss -- a column renamed in one module and
not another, or a service that works in isolation but not on the frame the
registry actually hands it.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.preprocessing import COST_TARGET, YIELD_TARGET, features_for
from src.services import insights, predictor, profitability, recommend


class TestPipelineContract:
    def test_the_registry_frame_satisfies_every_model(self, dataset, bundles):
        """The single most valuable assertion in the suite.

        registry.load_dataset() is what every page consumes; the bundles were
        fitted on a frame built by prepare(). If those two ever drift apart the
        app breaks at the first prediction instead of at deploy time.
        """
        for target in (YIELD_TARGET, COST_TARGET):
            missing = [f for f in features_for(target) if f not in dataset.columns]
            assert not missing, f"{target} features missing from the registry frame"

    def test_predict_forecast_recommend_on_one_dataset(self, dataset, bundles):
        # 1. A plausible mid-country field straight from the dataset.
        row = dataset.loc[
            (dataset["state"] == "Punjab") & (dataset["crop"] == "Wheat")
        ].iloc[0]

        # 2. Forecast it forward a year.
        scenario = predictor.Scenario(
            state=row["state"],
            crop=row["crop"],
            variety=row["variety"],
            recommended_zone=row["recommended_zone"],
            season=row["season_name"],
            season_duration=row["season_duration"],
            area_ha=float(row["quantity"]),
            year=2021,
        )
        forecast = predictor.forecast([scenario], bundles=bundles)
        numbers = forecast.first()
        assert numbers["yield_quintals"] > 0
        assert numbers["cost"] > 0
        assert numbers["yield_lower"] <= numbers["yield_quintals"] <= numbers["yield_upper"]

        # 3. Price it.
        profit = profitability.assess(forecast).first()
        assert profit["profit"] == pytest.approx(
            profit["revenue"] - profit["cost"], rel=1e-9
        )

        # 4. And confirm the recommendation engine likes the same crop there.
        best = recommend.recommend_for_crop(
            "Wheat", frame=dataset, state="Punjab", limit=1
        ).top(1)
        assert not best.empty

    def test_every_reported_kpi_is_a_finite_number(self, dataset):
        kpis = insights.headline_kpis(dataset)
        assert kpis
        for key, value in kpis.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                assert pd.notna(value), key
                assert abs(value) != float("inf"), key

    def test_training_report_and_live_dataset_agree(self, dataset, bundles):
        # metrics.json records the row count the models were fitted on. A drift
        # here means the shipped artefacts predate the current data pipeline.
        import json

        from config.settings import METRICS_PATH

        if not METRICS_PATH.exists():
            pytest.skip("metrics.json not written yet")
        with open(METRICS_PATH, encoding="utf-8") as handle:
            metrics = json.load(handle)
        trained_on = metrics.get("provenance", {}).get("rows_clean")
        if trained_on is None:
            pytest.skip("metrics.json records no clean row count")
        assert trained_on == len(dataset), (
            f"models were fitted on {trained_on} rows but the pipeline now "
            f"produces {len(dataset)} -- retrain"
        )

    def test_shipped_metrics_report_the_expected_coverage(self):
        # Interval calibration is the claim the UI puts in front of users, so
        # the number in metrics.json had better exist and be sane.
        import json

        from config.settings import METRICS_PATH

        if not METRICS_PATH.exists():
            pytest.skip("metrics.json not written yet")
        with open(METRICS_PATH, encoding="utf-8") as handle:
            metrics = json.load(handle)
        nominal = metrics.get("interval", {}).get("nominal_coverage", 0.8)
        for target in (YIELD_TARGET, COST_TARGET):
            interval = metrics["models"][target].get("interval", {})
            coverage = interval.get("test_coverage", {}).get("coverage")
            if coverage is None:
                pytest.skip(f"no coverage recorded for {target}")
            # metrics.json stores coverage as a percentage.
            pct = float(coverage)
            assert 60.0 < pct < 100.0, (
                f"{target}: {pct:.1f}% coverage on held-out data "
                f"(nominal {nominal:.0%}) -- the published intervals are "
                "miscalibrated"
            )

    def test_a_plausible_range_of_fields_all_predict(self, dataset, bundles):
        # Ten real observations, re-predicted from their own features.
        sample = dataset.sample(10, random_state=7)
        scenarios = [
            predictor.Scenario(
                state=r.state,
                crop=r.crop,
                variety=r.variety,
                recommended_zone=r.recommended_zone,
                season=r.season_name,
                area_ha=float(r.quantity),
                year=int(r.year),
            )
            for r in sample.itertuples()
        ]
        forecast = predictor.forecast(scenarios, bundles=bundles)
        assert len(forecast) == 10
        assert (forecast.yield_point > 0).all()
        assert (forecast.cost_point > 0).all()
        assert (forecast.yield_lower <= forecast.yield_point).all()
        assert (forecast.yield_point <= forecast.yield_upper).all()
        assert (forecast.interval_width() > 0).all()
