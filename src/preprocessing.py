"""Feature engineering, leakage control and preprocessor construction.

Design notes
------------
**Leakage firewall.** Two models are trained from one table. Each therefore gets
an explicit exclusion list. The yield model may not see anything derived from
``production`` or ``cost``; the cost model may not see anything derived from
``production`` or ``cost``. :data:`YIELD_EXCLUDED` / :data:`COST_EXCLUDED`
encode this and are asserted by ``tests/test_no_leakage.py``.

**Target encoding needs the target.** ``sklearn``'s ``TargetEncoder`` fits
supervised encodings, so a preprocessor is built per target rather than shared.

**Heavy-tailed targets.** Production spans roughly four orders of magnitude and
cost spans three. Both are therefore modelled on ``log1p`` via
``TransformedTargetRegressor``, which also guarantees non-negative predictions.
Metrics are reported in the original units (see ``src/evaluate.py``) alongside
the log-scale figures the models were actually fit on.

**Outlier rows.** The generator injects entry-error outliers (a production
figure 15-60x plausible). They are flagged by :func:`flag_outliers` and excluded
from *training* while remaining visible in the EDA views, which is what an
analyst would actually do rather than silently deleting them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config.settings import YEAR_MIN

# --------------------------------------------------------------------------
# Feature definitions
# --------------------------------------------------------------------------
# High-cardinality columns: supervised target encoding, not one-hot. `Variety`
# and `Recommended Zone` would otherwise explode the feature space.
HIGH_CARDINALITY = ("state", "crop", "variety", "recommended_zone")

# Low-cardinality columns: one-hot is both stable and directly interpretable.
LOW_CARDINALITY = (
    "season_name",
    "season_duration",
    "unit",
    "zone_scope",
    "crop_group",
    "variety_class",
)

NUMERIC_FEATURES = ("quantity", "year", "log_quantity", "sqrt_quantity", "year_index")

#: Columns that must never reach the yield model, including derived ratios.
YIELD_EXCLUDED = frozenset({
    "production",
    "production_quintals",
    "production_tons",
    "production_efficiency",
    "cost",
    "cost_per_unit",
    "is_outlier",
})

#: Columns that must never reach the cost model.
COST_EXCLUDED = frozenset({
    "production",
    "production_quintals",
    "production_tons",
    "production_efficiency",
    "cost",
    "cost_per_unit",
    "is_outlier",
})

YIELD_TARGET = "production_quintals"
COST_TARGET = "cost"

BASE_FEATURES = HIGH_CARDINALITY + LOW_CARDINALITY + NUMERIC_FEATURES


def features_for(target: str) -> list[str]:
    """Feature list for a target, honouring the leakage firewall."""
    excluded = YIELD_EXCLUDED if target == YIELD_TARGET else COST_EXCLUDED
    return [f for f in BASE_FEATURES if f not in excluded]


# --------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------
def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the numeric transforms the models consume.

    Operates on a copy. Only features derivable without peeking at either
    target are added here, so this function is safe to call once and share.
    """
    out = frame.copy()

    quantity = pd.to_numeric(out["quantity"], errors="coerce").clip(lower=0.0)
    out["log_quantity"] = np.log1p(quantity)
    out["sqrt_quantity"] = np.sqrt(quantity)
    out["year_index"] = (
        pd.to_numeric(out["year"], errors="coerce").fillna(YEAR_MIN) - YEAR_MIN
    ).astype("float64")

    # Categorical columns are filled, not left as NA: the one-hot encoder would
    # otherwise dedicate a column to "missing" and the target encoder would
    # treat NA as its own category.
    for column in HIGH_CARDINALITY + LOW_CARDINALITY:
        if column in out.columns:
            out[column] = out[column].fillna("Unknown").astype("string")

    out["quantity"] = quantity
    return out


def flag_outliers(
    frame: pd.DataFrame,
    efficiency_column: str = "production_efficiency",
    cost_column: str = "cost_per_unit",
    z_threshold: float = 12.0,
) -> pd.DataFrame:
    """Flag likely data-entry errors using a within-crop robust z-score.

    Median absolute deviation is used rather than a standard deviation because a
    handful of 50x outliers would inflate sigma and hide themselves.
    """
    out = frame.copy()
    flags = pd.Series(False, index=out.index)

    def _robust_z(series: pd.Series) -> pd.Series:
        values = pd.to_numeric(series, errors="coerce")
        median = values.median()
        mad = (values - median).abs().median()
        if not np.isfinite(mad) or mad == 0:
            return pd.Series(0.0, index=out.index)
        # 0.6745 makes MAD a consistent estimator of sigma for normal data.
        return ((values - median).abs() / (mad / 0.6745))

    for column, key in ((efficiency_column, "crop"), (cost_column, "crop")):
        if column not in out.columns or key not in out.columns:
            continue
        z = out.groupby(key, observed=True)[column].transform(_robust_z)
        flags |= (z > z_threshold).fillna(False)

    # A record cannot be valid if its output is orders of magnitude off its own
    # crop, regardless of how the crop normally behaves.
    if efficiency_column in out.columns:
        median_eff = out.groupby("crop", observed=True)[efficiency_column].transform(
            "median"
        )
        ratio = out[efficiency_column] / median_eff.replace(0, np.nan)
        flags |= (ratio > 25).fillna(False) | (ratio < 1 / 25).fillna(False)

    out["is_outlier"] = flags.astype(bool)
    return out


def prepare(
    frame: pd.DataFrame, flag: bool = True
) -> pd.DataFrame:
    """Full preparation: engineer features and flag outliers."""
    out = engineer_features(frame)
    if flag:
        out = flag_outliers(out)
    else:
        out["is_outlier"] = False
    return out


# --------------------------------------------------------------------------
# Preprocessor
# --------------------------------------------------------------------------
def _target_encoder(random_state: int = 0):
    """TargetEncoder configured for sklearn >=1.9.

    ``shuffle``/``random_state`` were deprecated in 1.9 in favour of passing a
    shuffled CV splitter, so a ``KFold`` instance is supplied to keep the
    internal cross-fitting randomised *and* warning-free.
    """
    from sklearn.preprocessing import TargetEncoder

    return TargetEncoder(
        cv=KFold(n_splits=5, shuffle=True, random_state=random_state),
        smooth="auto",
        target_type="continuous",
    )


def build_preprocessor(
    target: str,
    random_state: int = 0,
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """Construct the preprocessing pipeline for one target."""
    features = features_for(target)
    high = [f for f in HIGH_CARDINALITY if f in features]
    low = [f for f in LOW_CARDINALITY if f in features]
    numeric = [f for f in NUMERIC_FEATURES if f in features]

    transformers: list[tuple] = []
    if high:
        transformers.append(("target_enc", _target_encoder(random_state), high))
    if low:
        transformers.append(
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                low,
            )
        )
    if numeric:
        steps: list[tuple] = [("impute", _numeric_imputer())]
        if scale_numeric:
            steps.append(("scale", StandardScaler()))
        transformers.append(("numeric", Pipeline(steps), numeric))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )


def _numeric_imputer():
    from sklearn.impute import SimpleImputer

    return SimpleImputer(strategy="median")


def collapse_display_names(
    expanded_names, low_cardinality=LOW_CARDINALITY
) -> list[str]:
    """Collapse an expanded design-matrix header back to its feature block.

    ``onehot__season_name_Kharif`` -> ``season_name``. Feature importance over
    39 one-hot levels is unreadable; over 8 feature blocks it is a chart a
    planner can act on.
    """
    blocks: list[str] = []
    for name in expanded_names:
        parts = str(name).split("__", 1)
        token = parts[1] if len(parts) == 2 else str(name)
        collapsed = token
        for column in low_cardinality:
            # Longest match first so `season_name` is not shadowed by `season`.
            if token == column or token.startswith(f"{column}_"):
                collapsed = column
                break
        blocks.append(collapsed)
    return blocks


def feature_display_names(preprocessor: ColumnTransformer) -> list[str]:
    """Ordered, de-duplicated feature blocks produced by a fitted preprocessor."""
    try:
        expanded = list(preprocessor.get_feature_names_out())
    except Exception:  # pragma: no cover - preprocessor not yet fitted
        return []
    seen: dict[str, None] = {}
    for block in collapse_display_names(expanded):
        seen.setdefault(block, None)
    return list(seen)


def _block_of(expanded_name: str) -> str:
    parts = str(expanded_name).split("__", 1)
    return parts[1] if len(parts) == 2 else str(expanded_name)


def expanded_to_display(expanded_names) -> np.ndarray:
    """Map every expanded column of the design matrix to its display block."""
    return np.array(collapse_display_names(expanded_names), dtype=object)


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------
@dataclass
class SplitResult:
    train: pd.DataFrame
    test: pd.DataFrame
    strategy: str
    description: str


def make_split(
    frame: pd.DataFrame,
    strategy: str = "random",
    test_size: float = 0.2,
    random_state: int = 0,
) -> SplitResult:
    """Create a train/test split under the requested strategy.

    ``random``
        Plain shuffle. Optimistic but the standard default.
    ``group_state``
        Hold out entire states, answering "how well does this work in a state
        the model has never seen?"
    ``group_zone``
        Hold out entire recommended zones -- the strictest test, and the one
        that matters for the recommendation portal.
    ``chronological``
        Train on the earlier years, test on the later ones. The only strategy
        that gives an honest forward-looking estimate.
    """
    if strategy == "random":
        from sklearn.model_selection import train_test_split

        train, test = train_test_split(
            frame, test_size=test_size, random_state=random_state
        )
        description = "Random 80/20 shuffle"

    elif strategy == "group_state":
        train, test = _group_split(frame, "state", test_size, random_state)
        description = "Held-out states (model never sees them in training)"

    elif strategy == "group_zone":
        key = "zone_name" if "zone_name" in frame.columns else "recommended_zone"
        train, test = _group_split(frame, key, test_size, random_state)
        description = f"Held-out zones by `{key}`"

    elif strategy == "chronological":
        years = pd.to_numeric(frame["year"], errors="coerce")
        cutoff = years.quantile(1 - test_size)
        train = frame.loc[years <= cutoff]
        test = frame.loc[years > cutoff]
        description = f"Chronological (train <= {int(cutoff)}, test > {int(cutoff)})"

    else:
        raise ValueError(
            f"Unknown split strategy {strategy!r}. "
            "Choose one of: random, group_state, group_zone, chronological"
        )

    return SplitResult(train, test, strategy, description)


def split_train_calibration(
    train: pd.DataFrame, calibration_size: float = 0.15, random_state: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carve a calibration slice out of the training set.

    Kept disjoint from both train and test so the interval widening factor and
    the reported coverage are both measured on rows no model was fitted on.
    """
    from sklearn.model_selection import train_test_split

    fit, calibration = train_test_split(
        train, test_size=calibration_size, random_state=random_state
    )
    return fit, calibration


def _group_split(
    frame: pd.DataFrame, key: str, test_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import GroupShuffleSplit

    if key not in frame.columns:
        raise ValueError(f"Cannot group-split on missing column {key!r}")

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_idx, test_idx = next(splitter.split(frame, groups=frame[key]))
    return frame.iloc[train_idx].copy(), frame.iloc[test_idx].copy()


# --------------------------------------------------------------------------
# Leakage assertions
# --------------------------------------------------------------------------
def assert_no_leakage(features: list[str], target: str) -> None:
    """Raise if a feature set overlaps its own target's exclusions."""
    excluded = YIELD_EXCLUDED if target == YIELD_TARGET else COST_EXCLUDED
    offenders = sorted(set(features) & excluded)
    if offenders:
        raise ValueError(
            f"Target leakage: {offenders} must not be features of the "
            f"{target!r} model."
        )
    if target in features:
        raise ValueError(f"Target {target!r} appears in its own feature list.")


def audit_feature_sets() -> pd.DataFrame:
    """Table describing what each model is allowed to see, for the UI."""
    rows = []
    for label, target in (("Production (yield)", YIELD_TARGET),
                          ("Cultivation cost", COST_TARGET)):
        excluded = YIELD_EXCLUDED if target == YIELD_TARGET else COST_EXCLUDED
        features = features_for(target)
        rows.append(
            {
                "model": label,
                "target": target,
                "n_features": len(features),
                "high_cardinality_encoded": ", ".join(
                    f for f in HIGH_CARDINALITY if f in features
                ),
                "one_hot_encoded": ", ".join(
                    f for f in LOW_CARDINALITY if f in features
                ),
                "numeric": ", ".join(f for f in NUMERIC_FEATURES if f in features),
                "excluded_to_avoid_leakage": ", ".join(sorted(excluded)),
            }
        )
    return pd.DataFrame(rows)
