"""Feature engineering, the leakage firewall and splitting.

The leakage tests are the point of this file. A yield model that sees ``cost``
or ``production_efficiency`` scores beautifully on held-out data and is
worthless in the field, and the failure is invisible without an explicit check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.settings import YEAR_MIN
from src.preprocessing import (
    BASE_FEATURES,
    COST_EXCLUDED,
    COST_TARGET,
    YIELD_EXCLUDED,
    YIELD_TARGET,
    assert_no_leakage,
    audit_feature_sets,
    build_preprocessor,
    collapse_display_names,
    engineer_features,
    expanded_to_display,
    feature_display_names,
    features_for,
    flag_outliers,
    prepare,
)


class TestLeakageFirewall:
    @pytest.mark.parametrize(
        "forbidden",
        ["production", "production_quintals", "production_tons",
         "production_efficiency", "cost", "cost_per_unit"],
    )
    def test_yield_features_exclude_every_target_derived_column(self, forbidden):
        assert forbidden not in features_for(YIELD_TARGET)

    def test_cost_features_exclude_yield_columns(self):
        for forbidden in ("production", "production_quintals", "production_efficiency"):
            assert forbidden not in features_for(COST_TARGET)

    def test_cost_features_exclude_cost_itself(self):
        assert "cost" not in features_for(COST_TARGET)

    def test_outlier_flag_never_reaches_a_model(self):
        # is_outlier is computed from both targets, so it is pure leakage.
        assert "is_outlier" not in features_for(YIELD_TARGET)
        assert "is_outlier" not in features_for(COST_TARGET)

    def test_area_and_year_are_allowed(self):
        # The inputs a farmer actually knows must survive.
        for target in (YIELD_TARGET, COST_TARGET):
            assert "quantity" in features_for(target)
            assert "year" in features_for(target)

    def test_assert_no_leakage_passes_on_the_real_feature_sets(self):
        for target in (YIELD_TARGET, COST_TARGET):
            assert_no_leakage(features_for(target), target)

    def test_assert_no_leakage_actually_raises(self):
        with pytest.raises(ValueError, match="leakage"):
            assert_no_leakage(["quantity", "cost"], YIELD_TARGET)

    def test_assert_no_leakage_rejects_target_in_its_own_features(self):
        with pytest.raises(ValueError):
            assert_no_leakage(["quantity", YIELD_TARGET], YIELD_TARGET)

    @pytest.mark.parametrize("target", [YIELD_TARGET, COST_TARGET])
    def test_no_excluded_column_leaks_through_a_shared_name(self, target):
        excluded = YIELD_EXCLUDED if target == YIELD_TARGET else COST_EXCLUDED
        features = features_for(target)
        assert not (set(features) & excluded)

    def test_audit_frame_lists_every_target(self):
        audit = audit_feature_sets()
        assert set(audit["target"]) == {YIELD_TARGET, COST_TARGET}
        assert (audit["n_features"] > 0).all()

    def test_exclusions_are_documented_as_sets(self):
        assert isinstance(YIELD_EXCLUDED, frozenset)
        assert isinstance(COST_EXCLUDED, frozenset)
        # Both targets must be protected from each other, not just themselves.
        assert "cost" in YIELD_EXCLUDED
        assert "production_efficiency" in COST_EXCLUDED


class TestEngineerFeatures:
    def test_derived_numeric_columns_appear(self, prepared):
        for column in ("log_quantity", "sqrt_quantity", "year_index"):
            assert column in prepared.columns

    def test_log_and_sqrt_are_correct(self, prepared):
        quantity = prepared["quantity"].to_numpy(dtype=float)
        np.testing.assert_allclose(
            prepared["log_quantity"].to_numpy(dtype=float),
            np.log1p(quantity),
            rtol=1e-9,
        )
        np.testing.assert_allclose(
            prepared["sqrt_quantity"].to_numpy(dtype=float),
            np.sqrt(quantity),
            rtol=1e-9,
        )

    def test_year_index_starts_at_zero(self):
        frame = engineer_features(
            pd.DataFrame(
                {
                    "quantity": [1.0],
                    "year": [float(YEAR_MIN)],
                    "state": ["Punjab"],
                    "crop": ["Wheat"],
                    "variety": ["HD-2967"],
                    "recommended_zone": ["State - Punjab"],
                    "season_name": ["Rabi"],
                    "season_duration": ["Medium"],
                }
            )
        )
        assert frame["year_index"].iloc[0] == 0.0

    def test_does_not_mutate_the_input(self, clean_frame):
        frame, _, _ = clean_frame
        before = frame.copy()
        engineer_features(frame)
        pd.testing.assert_frame_equal(frame, before)

    def test_categoricals_have_no_nulls(self, prepared):
        for column in ("state", "crop", "variety", "recommended_zone",
                       "season_name", "season_duration"):
            assert prepared[column].notna().all(), column

    def test_negative_quantity_is_clipped_not_logged(self):
        frame = engineer_features(
            pd.DataFrame({"quantity": [-5.0, 0.0], "year": [2001, 2001]})
        )
        assert (frame["quantity"] >= 0).all()
        assert np.isfinite(frame["log_quantity"]).all()


class TestOutlierFlag:
    def test_flag_is_boolean_and_present(self, prepared):
        assert "is_outlier" in prepared.columns
        assert prepared["is_outlier"].dtype == bool

    def test_flags_a_planted_extreme(self):
        base = pd.DataFrame(
            {
                "crop": ["Wheat"] * 50,
                "production_efficiency": [20.0] * 50,
                "cost_per_unit": [20000.0] * 50,
            }
        )
        base.loc[base.index[-1], "production_efficiency"] = 5000.0
        flagged = flag_outliers(base)
        assert bool(flagged["is_outlier"].iloc[-1])
        assert int(flagged["is_outlier"].sum()) == 1

    def test_normal_rows_are_not_flagged(self):
        base = pd.DataFrame(
            {
                "crop": ["Rice"] * 100,
                "production_efficiency": np.linspace(15, 35, 100),
                "cost_per_unit": np.linspace(18000, 26000, 100),
            }
        )
        assert int(flag_outliers(base)["is_outlier"].sum()) == 0

    def test_scales_within_crop(self):
        # A value normal for sugarcane must not be flagged just because it is
        # large next to wheat.
        frame = pd.DataFrame(
            {
                "crop": ["Wheat"] * 60 + ["Sugarcane"] * 60,
                "production_efficiency": [20.0] * 60 + [800.0] * 60,
                "cost_per_unit": [20000.0] * 120,
            }
        )
        assert int(flag_outliers(frame)["is_outlier"].sum()) == 0


class TestPreprocessor:
    @pytest.mark.parametrize("target", [YIELD_TARGET, COST_TARGET])
    def test_fits_and_transforms_to_a_dense_matrix(self, target, prepared):
        features = features_for(target)
        pre = build_preprocessor(target)
        # The high-cardinality branch is a supervised TargetEncoder, so the fit
        # needs the target itself -- that is exactly why it is fitted inside
        # cross-validation folds rather than once on the whole dataset.
        matrix = pre.fit_transform(prepared[features], prepared[target])
        assert matrix.shape[0] == len(prepared)
        assert np.isfinite(matrix).all()

    @pytest.mark.parametrize("target", [YIELD_TARGET, COST_TARGET])
    def test_unseen_categories_do_not_crash(self, target, prepared):
        features = features_for(target)
        pre = build_preprocessor(target).fit(
            prepared[features], prepared[target]
        )
        novel = prepared[features].head(3).copy()
        novel["state"] = "Atlantis"
        novel["crop"] = "Unobtainium"
        matrix = pre.transform(novel)
        assert matrix.shape[0] == 3
        assert np.isfinite(matrix).all()

    def test_collapsed_names_align_with_the_matrix_width(self, prepared):
        target = YIELD_TARGET
        features = features_for(target)
        pre = build_preprocessor(target).fit(
            prepared[features], prepared[target]
        )
        matrix = pre.transform(prepared[features])
        # One label per design-matrix column, so importance can be grouped.
        collapsed = collapse_display_names(list(pre.get_feature_names_out()))
        assert len(collapsed) == matrix.shape[1]

    def test_display_names_are_fewer_readable_blocks_than_columns(self, prepared):
        target = YIELD_TARGET
        features = features_for(target)
        pre = build_preprocessor(target).fit(
            prepared[features], prepared[target]
        )
        matrix = pre.transform(prepared[features])
        blocks = feature_display_names(pre)
        # 39 one-hot levels become a handful of chartable feature blocks.
        assert len(blocks) < matrix.shape[1]
        assert len(set(blocks)) == len(blocks)
        assert all(block and "__" not in block for block in blocks)

    def test_expanded_to_display_matches_collapse(self, prepared):
        target = YIELD_TARGET
        features = features_for(target)
        pre = build_preprocessor(target).fit(
            prepared[features], prepared[target]
        )
        expanded = list(pre.get_feature_names_out())
        np.testing.assert_array_equal(
            expanded_to_display(expanded),
            np.array(collapse_display_names(expanded), dtype=object),
        )
        assert expanded_to_display([]).size == 0


class TestBaseFeatures:
    def test_base_feature_list_has_no_duplicates(self):
        assert len(BASE_FEATURES) == len(set(BASE_FEATURES))

    @pytest.mark.parametrize("target", [YIELD_TARGET, COST_TARGET])
    def test_feature_lists_have_no_duplicates(self, target):
        features = features_for(target)
        assert len(features) == len(set(features))

    @pytest.mark.parametrize("target", [YIELD_TARGET, COST_TARGET])
    def test_every_feature_exists_in_a_prepared_frame(self, target, prepared):
        missing = [f for f in features_for(target) if f not in prepared.columns]
        assert not missing, f"{target} needs missing columns: {missing}"


class TestPrepare:
    def test_prepare_does_not_mutate(self, clean_frame):
        frame, _, _ = clean_frame
        before = frame.copy()
        prepare(frame)
        pd.testing.assert_frame_equal(frame, before)

    def test_flag_false_disables_outlier_marking(self, clean_frame):
        frame, _, _ = clean_frame
        out = prepare(frame, flag=False)
        assert not out["is_outlier"].any()
