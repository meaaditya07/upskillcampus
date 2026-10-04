"""Ingestion, cleaning and provenance.

The loader is the boundary between a messy real file and everything downstream,
so these tests assert that nothing bad survives and that everything dropped is
counted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_loader import (
    MISSING_TOKENS,
    DataValidationError,
    find_source_csv,
    load_dataset,
    order_columns,
)


class TestCleaning:
    def test_all_required_columns_present(self, clean_frame):
        frame, _, _ = clean_frame
        for column in ("crop", "variety", "state", "quantity", "production",
                       "season_name", "unit", "cost", "recommended_zone", "year"):
            assert column in frame.columns, f"missing {column}"

    def test_no_missing_values_in_essentials(self, clean_frame):
        frame, _, _ = clean_frame
        for column in ("crop", "state", "quantity", "production", "cost", "year"):
            assert frame[column].notna().all(), f"{column} has nulls"
            assert frame[column].astype(str).str.strip().ne("").all(), f"{column} has blanks"

    def test_quantity_is_strictly_positive(self, clean_frame):
        frame, _, _ = clean_frame
        assert (frame["quantity"] > 0).all()

    def test_production_and_cost_non_negative(self, clean_frame):
        frame, _, _ = clean_frame
        assert (frame["production_quintals"] >= 0).all()
        assert (frame["cost"] >= 0).all()

    def test_year_within_declared_window(self, clean_frame):
        from config.settings import YEAR_MAX, YEAR_MIN

        frame, _, _ = clean_frame
        assert frame["year"].between(YEAR_MIN, YEAR_MAX).all()

    def test_units_fold_onto_three_labels(self, clean_frame):
        frame, _, _ = clean_frame
        # "Unknown" is a legitimate third label: the generator injects missing
        # units, and those rows must land somewhere explicit rather than as a
        # "<Na>" one-hot level of their own.
        assert set(frame["unit"].unique()) <= {"Quintals", "Tons", "Unknown"}

    def test_missing_units_are_counted_not_hidden(self, clean_frame):
        frame, report, _ = clean_frame
        n_unknown = int((frame["unit"] == "Unknown").sum())
        assert report.unit_fallbacks == n_unknown

    def test_no_pandas_placeholder_strings_survive_anywhere(self, clean_frame):
        frame, _, _ = clean_frame
        # "<Na>" is what pd.NA stringifies to, and it previously leaked through
        # the loader as if it were a real category.
        for column in frame.select_dtypes(include=["object", "string"]).columns:
            values = frame[column].astype("string")
            for placeholder in ("<NA>", "<Na>", "<nan>", "None", "NaN"):
                assert not (values == placeholder).any(), (
                    f"{column} contains the placeholder {placeholder!r}"
                )

    def test_states_are_canonical(self, clean_frame):
        from src.schema import is_known_state

        frame, _, _ = clean_frame
        unknown = {s for s in frame["state"].unique() if not is_known_state(s)}
        assert not unknown, f"non-canonical states survived: {unknown}"

    def test_crops_are_canonical(self, clean_frame):
        from src.schema import CROP_ALIASES

        frame, _, _ = clean_frame
        unknown = set(frame["crop"].unique()) - set(CROP_ALIASES.values())
        assert not unknown, f"non-canonical crops survived: {unknown}"

    def test_derived_ratios_are_consistent(self, clean_frame):
        frame, _, _ = clean_frame
        expected = frame["production_quintals"] / frame["quantity"]
        np.testing.assert_allclose(
            frame["production_efficiency"].to_numpy(),
            expected.to_numpy(),
            rtol=1e-6,
        )

    def test_quintal_conversion_applied(self, clean_frame):
        # Rows reported in Tons must carry 20x the quintals of the same figure.
        frame, _, _ = clean_frame
        tons = frame.loc[frame["unit"] == "Tons"]
        if tons.empty:
            pytest.skip("no Tons rows in this sample")
        assert (tons["production_quintals"] > 0).all()
        assert (tons["production_tons"] * 20.0 - tons["production_quintals"]).abs().max() < 1e-3


class TestDerivedColumns:
    def test_season_split(self, clean_frame):
        frame, _, _ = clean_frame
        assert frame["season_name"].notna().all()
        assert frame["season_duration"].notna().all()

    def test_zone_split(self, clean_frame):
        frame, _, _ = clean_frame
        assert frame["zone_scope"].notna().all()
        assert set(frame["zone_scope"].unique()) <= {
            "State", "Mandal", "Village", "Unknown"
        }

    def test_variety_classified(self, clean_frame):
        frame, _, _ = clean_frame
        assert frame["variety_class"].notna().all()

    def test_no_categorical_sentinels_survive(self, clean_frame):
        frame, _, _ = clean_frame
        for column in ("crop", "state", "recommended_zone"):
            values = frame[column].astype(str).str.lower()
            for token in MISSING_TOKENS:
                if not token:
                    continue
                assert not (values == token).any(), f"{column} contains {token!r}"


class TestReport:
    def test_report_counts_add_up(self, clean_frame):
        frame, report, _ = clean_frame
        assert report.rows_out == len(frame)
        assert report.rows_in >= report.rows_out

    def test_dropped_rows_are_explained(self, clean_frame):
        _, report, _ = clean_frame
        total = report.dropped_rows.get("total", 0)
        reasons = {k: v for k, v in report.dropped_rows.items() if k != "total"}
        assert reasons, "drops happened but none were attributed to a reason"
        # Reasons deliberately overlap -- a row can be missing both production
        # and cost -- so they sum to at least the true row count, and no single
        # reason may exceed it.
        assert sum(reasons.values()) >= total
        assert max(reasons.values()) <= total

    def test_resolved_columns_are_canonical_to_source(self, clean_frame):
        _, report, _ = clean_frame
        assert report.resolved_columns["crop"]
        assert not report.missing_required

    def test_unit_normalisation_is_counted(self, clean_frame):
        _, report, _ = clean_frame
        assert isinstance(report.normalised_units, dict)


class TestProvenance:
    def test_synthetic_input_is_marked_synthetic(self, clean_frame):
        _, _, provenance = clean_frame
        assert provenance.is_synthetic is True
        assert provenance.sha256

    def test_provenance_records_window(self, clean_frame):
        _, _, provenance = clean_frame
        assert provenance.rows_raw > 0

    def test_explicit_path_is_honoured(self, raw_csv):
        assert find_source_csv(explicit=raw_csv) == raw_csv


class TestFailures:
    def test_missing_file_raises_rather_than_silently_substituting(self, tmp_path):
        # Asking for a specific file and getting different data back would mean
        # drawing conclusions from the wrong numbers.
        with pytest.raises(DataValidationError):
            load_dataset(source=tmp_path / "does_not_exist.csv")

    def test_file_without_required_columns_raises(self, tmp_path):
        path = tmp_path / "bad.csv"
        pd.DataFrame({"foo": [1], "bar": [2]}).to_csv(path, index=False)
        with pytest.raises(DataValidationError):
            load_dataset(source=path)

    def test_empty_file_raises(self, tmp_path):
        path = tmp_path / "empty.csv"
        pd.DataFrame(columns=["Crop", "State", "Quantity"]).to_csv(path, index=False)
        with pytest.raises(DataValidationError):
            load_dataset(source=path)


class TestColumnOrder:
    def test_order_columns_is_idempotent(self, clean_frame):
        frame, _, _ = clean_frame
        once = order_columns(frame)
        twice = order_columns(once)
        assert list(once.columns) == list(twice.columns)

    def test_ordering_preserves_every_column(self, clean_frame):
        frame, _, _ = clean_frame
        assert set(order_columns(frame).columns) == set(frame.columns)
