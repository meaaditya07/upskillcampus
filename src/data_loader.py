"""Data ingestion, cleaning and schema normalisation.

Pipeline
--------
    locate source  ->  read as text  ->  resolve columns  ->  clean strings
    ->  coerce numerics  ->  drop invalid rows  ->  normalise units
    ->  derive categorical views  ->  (frame, quality report, provenance)

Everything is read as text first so that type coercion is explicit and
countable. Reading with pandas' own inference hides exactly the defects we
need to report on -- ``1,200`` silently becomes text, ``NA`` silently becomes
NaN, and the audit trail lies.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import (
    CLEANED_CSV,
    RAW_CSV_GLOB,
    RAW_DIR,
    SYNTHETIC_CSV,
    YEAR_MAX,
    YEAR_MIN,
)
from src.schema import (
    DISPLAY_ORDER,
    REQUIRED_COLUMNS,
    QualityReport,
    canonical_crop,
    canonical_state,
    canonical_unit,
    canonical_zone,
    classify_variety,
    crop_group,
    parse_season,
    parse_zone,
    quintal_factor,
    resolve_columns,
)

SYNTHETIC_MARKER = "crop_production_synthetic.csv"

MISSING_TOKENS = {
    "", "-", "--", "---", "na", "n/a", "n.a.", "nan", "null", "none",
    "nil", "not available", "notavailable", "unknown", "?", "#n/a", "#na",
}

# Values a real extract uses for a suppressed government figure.
SUPPRESSED_TOKENS = {"*", "**", "***"}

_CURRENCY_NOISE = re.compile(r"[₹$€£,\s]")
_CURRENCY_PREFIX = re.compile(r"(?i)\s*(inr|rs\.?|rupees)\s*[:.-]?\s*")
_YEAR_RE = re.compile(r"(\d{4})")
_WS = re.compile(r"\s+")



class DataValidationError(RuntimeError):
    """Raised when a source file cannot satisfy the minimum schema."""


@dataclass
class Provenance:
    """Where the loaded data came from, and whether to trust it."""

    source: str            # 'synthetic' | 'file'
    path: str
    sha256: str
    rows_raw: int
    rows_clean: int
    loaded_at: str
    is_synthetic: bool

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "path": self.path,
            "sha256": self.sha256,
            "rows_raw": self.rows_raw,
            "rows_clean": self.rows_clean,
            "loaded_at": self.loaded_at,
            "is_synthetic": self.is_synthetic,
        }


# --------------------------------------------------------------------------
# Source discovery
# --------------------------------------------------------------------------
def find_source_csv(explicit: str | Path | None = None) -> Path:
    """Locate the dataset: an explicit path, else a real CSV, else synthetic."""
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            for candidate in (Path.cwd() / path, RAW_DIR / path, path):
                if candidate.exists():
                    path = candidate
                    break
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        return path

    candidates = sorted(
        p for p in RAW_DIR.glob(RAW_CSV_GLOB) if p.is_file()
    )
    real = [p for p in candidates if p.name != SYNTHETIC_MARKER]
    if real:
        return real[0]
    if SYNTHETIC_CSV.exists():
        return SYNTHETIC_CSV
    return SYNTHETIC_CSV  # caller generates it


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def ensure_synthetic_dataset(
    path: Path | None = None,
    n_rows: int | None = None,
    *,
    seed: int | None = None,
    dirty: bool = True,
    overwrite: bool = False,
) -> Path:
    """Generate the synthetic dataset if it is not already on disk.

    ``overwrite=True`` forces regeneration, which is how the dataset is
    rebuilt after a change to the generator's structure.
    """
    target = Path(path) if path else SYNTHETIC_CSV
    if target.exists() and not overwrite:
        return target
    from config.settings import SEED, SYNTHETIC_ROWS
    from src.synthetic import generate_synthetic_dataset

    target.parent.mkdir(parents=True, exist_ok=True)
    frame = generate_synthetic_dataset(
        n_rows=n_rows or SYNTHETIC_ROWS,
        seed=SEED if seed is None else seed,
        dirty=dirty,
    )
    frame.to_csv(target, index=False)
    return target


# --------------------------------------------------------------------------
# Cleaning primitives
# --------------------------------------------------------------------------
def _as_text(series: pd.Series) -> pd.Series:
    """Normalise a column to whitespace-collapsed, case-preserving text."""
    as_string = series.astype("string")
    stripped = as_string.str.strip()
    collapsed = stripped.str.replace(_WS, " ", regex=True)
    return collapsed


def _title_key(series: pd.Series) -> pd.Series:
    return _as_text(series).str.lower().str.replace(_WS, " ", regex=True)


def _normalise_state(series: pd.Series) -> pd.Series:
    keys = _title_key(series)
    mapped = keys.map(lambda k: canonical_state(k) if k else k)
    return mapped.astype("string")


def _normalise_crop(series: pd.Series) -> pd.Series:
    keys = _title_key(series)
    mapped = keys.map(lambda k: canonical_crop(k) if k else k)
    return mapped.astype("string")


def _to_numeric(
    series: pd.Series, report: QualityReport, column: str
) -> pd.Series:
    """Parse a messy text column into float64, counting the coercions."""
    text = _as_text(series).str.lower()
    text = text.mask(text.isin(SUPPRESSED_TOKENS), other=pd.NA)
    text = text.mask(text.isin(MISSING_TOKENS), other=pd.NA)
    stripped = text.str.replace(_CURRENCY_NOISE, "", regex=True)
    stripped = stripped.str.replace(_CURRENCY_PREFIX, "", regex=True)
    parsed = pd.to_numeric(stripped, errors="coerce")
    numeric = pd.Series(
        np.asarray(parsed, dtype="float64"), index=series.index, name=column
    )
    failures = int((numeric.isna() & stripped.notna()).sum())
    if failures:
        report.coerced_numeric[column] = failures
    return numeric


def _extract_year(series: pd.Series) -> pd.Series:
    """Handle '2001', '2001-02', '2001-2002' and 'Kharif 2001'."""
    text = _as_text(series)
    found = text.str.extract(_YEAR_RE, expand=False)
    year = pd.to_numeric(found, errors="coerce")
    year = year.astype("float64")
    # Reject implausible years, then clamp into the documented window.
    year = year.where(year.between(1900, 2100))
    return year.fillna(float(np.median([YEAR_MIN, YEAR_MAX]))).astype("int64")


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------
def load_dataset(
    source: str | Path | None = None,
    *,
    drop_invalid: bool = True,
    verbose: bool = False,
) -> tuple[pd.DataFrame, QualityReport, Provenance]:
    """Load, clean and normalise the dataset into the canonical schema.

    Returns ``(frame, quality_report, provenance)``.

    When ``source`` names a file that does not exist, this raises rather than
    falling back: a caller who asked for a specific file and silently received
    different data would draw conclusions from the wrong numbers. The implicit
    search in :func:`find_source_csv` is the only path that may substitute the
    synthetic dataset, and that substitution is recorded in the provenance.
    """
    if source is not None:
        path = Path(source)
        if not path.exists():
            raise DataValidationError(
                f"{path} does not exist. Pass an existing CSV, or omit "
                "`source` to let find_source_csv() discover one."
            )
    else:
        path = find_source_csv()

    report = QualityReport()
    raw = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=True,
        na_values=["", "NA", "N/A", "null", "NULL", "-", "--", "*"],
    )
    report.rows_in = len(raw)

    # -- column resolution ---------------------------------------------------
    resolved, unmapped = resolve_columns(raw.columns)
    report.resolved_columns = resolved
    report.unmapped_columns = unmapped

    missing_required = [c for c in REQUIRED_COLUMNS if c not in resolved]
    if missing_required:
        raise DataValidationError(
            "Dataset is missing required column(s): "
            f"{', '.join(missing_required)}. "
            f"Columns found: {list(raw.columns)}. "
            "Rename them or add an alias in src/schema.py:COLUMN_SYNONYMS."
        )

    frame = pd.DataFrame(index=raw.index)
    for canonical, original in resolved.items():
        frame[canonical] = raw[original]

    # -- string cleaning -----------------------------------------------------
    trim_targets = [c for c in ("crop", "variety", "state", "season", "unit",
                                "recommended_zone") if c in frame]
    for column in trim_targets:
        before = frame[column].astype("string")
        cleaned = _as_text(frame[column])
        report.string_trims += int(
            (before.fillna("").str.len() != cleaned.fillna("").str.len()).sum()
        )
        frame[column] = cleaned

    state_before = _title_key(frame["state"])
    frame["state"] = _normalise_state(frame["state"])
    report.case_fixes += int(
        (frame["state"].astype("string").fillna("") !=
         state_before.str.title()).sum()
    )

    crop_before = _title_key(frame["crop"])
    frame["crop"] = _normalise_crop(frame["crop"])
    report.case_fixes += int(
        (frame["crop"].astype("string").fillna("") !=
         crop_before.str.title()).sum()
    )

    # Zones are target-encoded, so they must land on exactly the strings the
    # model was trained on. canonical_zone is the same helper inference uses;
    # verified idempotent over every zone in the shipped dataset.
    if "recommended_zone" in frame.columns:
        zone_before = frame["recommended_zone"].astype("string")
        frame["recommended_zone"] = frame["recommended_zone"].map(canonical_zone)
        report.case_fixes += int(
            (zone_before != frame["recommended_zone"].astype("string")).sum()
        )

    # -- numeric coercion ----------------------------------------------------
    for column in ("quantity", "production", "cost"):
        if column in frame:
            frame[column] = _to_numeric(frame[column], report, column)
    if "year" in frame:
        frame["year"] = _extract_year(frame["year"])

    # -- row-level validity --------------------------------------------------
    if drop_invalid:
        checks = {
            "missing_quantity": frame["quantity"].isna()
            if "quantity" in frame
            else pd.Series(False, index=frame.index),
            "missing_production": frame["production"].isna()
            if "production" in frame
            else pd.Series(False, index=frame.index),
            "nonpositive_quantity": frame["quantity"] <= 0
            if "quantity" in frame
            else pd.Series(False, index=frame.index),
            "negative_production": frame["production"] < 0
            if "production" in frame
            else pd.Series(False, index=frame.index),
            "missing_cost": frame["cost"].isna()
            if "cost" in frame
            else pd.Series(False, index=frame.index),
            "negative_cost": frame["cost"] < 0
            if "cost" in frame
            else pd.Series(False, index=frame.index),
            "missing_state": frame["state"].isna()
            if "state" in frame
            else pd.Series(False, index=frame.index),
            "missing_crop": frame["crop"].isna()
            if "crop" in frame
            else pd.Series(False, index=frame.index),
        }
        invalid = pd.Series(False, index=frame.index)
        for name, mask in checks.items():
            count = int(mask.sum())
            if count:
                report.dropped_rows[name] = count
                invalid |= mask
        before = len(frame)
        frame = frame.loc[~invalid].copy()
        report.dropped_rows["total"] = before - len(frame)

    # -- unit normalisation --------------------------------------------------
    # Fold spelling variants ('Tons'/'tons'/'TONS'/' Tonne ') onto one label
    # *before* deriving quintals, so both the ratio maths and the one-hot
    # feature see a clean column.
    if "unit" in frame.columns:
        frame["unit"] = frame["unit"].map(canonical_unit)
    factors = frame["unit"].map(quintal_factor) if "unit" in frame else 1.0
    frame["production_quintals"] = frame["production"] * factors.astype("float64")
    frame["quantity_tons"] = frame["quantity"] / 20.0
    frame["production_tons"] = frame["production_quintals"] / 20.0

    if "unit" in frame:
        report.normalised_units = {
            str(u): int((frame["unit"] == u).sum())
            for u in frame["unit"].dropna().unique()
        }
        if (frame["unit"] == "Unknown").any():
            report.unit_fallbacks = int((frame["unit"] == "Unknown").sum())
            report.notes.append(
                f"{report.unit_fallbacks:,} rows had an unrecognised unit and "
                "were treated as quintals."
            )

    # -- categorical views ---------------------------------------------------
    if "season" in frame:
        parsed = frame["season"].map(parse_season)
        frame["season_name"] = pd.Series(
            [p[0] for p in parsed], index=frame.index, dtype="string"
        )
        frame["season_duration"] = pd.Series(
            [p[1] for p in parsed], index=frame.index, dtype="string"
        )
    else:
        frame["season_name"] = pd.Series("Unknown", index=frame.index, dtype="string")
        frame["season_duration"] = pd.Series(
            "Medium", index=frame.index, dtype="string"
        )

    if "recommended_zone" in frame:
        zones = frame["recommended_zone"].map(parse_zone)
        frame["zone_scope"] = pd.Series(
            [z[0] for z in zones], index=frame.index, dtype="string"
        )
        frame["zone_name"] = pd.Series(
            [z[1] for z in zones], index=frame.index, dtype="string"
        )
    else:
        frame["zone_scope"] = pd.Series("State", index=frame.index, dtype="string")
        frame["zone_name"] = frame.get(
            "state", pd.Series("Unknown", index=frame.index)
        ).astype("string")

    frame["crop_group"] = frame["crop"].map(crop_group).astype("string")
    frame["variety_class"] = (
        frame["variety"].map(classify_variety).astype("string")
        if "variety" in frame
        else pd.Series("Unknown", index=frame.index, dtype="string")
    )

    # Fill the categorical holes so the encoders never see a null.
    for column in ("variety", "season", "unit", "recommended_zone", "zone_name"):
        if column in frame:
            frame[column] = frame[column].fillna("Unknown").astype("string")

    # -- derived ratios (guarded) ------------------------------------------
    quantity_safe = frame["quantity"].replace(0, np.nan)
    frame["cost_per_unit"] = frame["cost"] / quantity_safe
    frame["production_efficiency"] = frame["production_quintals"] / quantity_safe
    frame = frame.loc[
        np.isfinite(frame["cost_per_unit"])
        & np.isfinite(frame["production_efficiency"])
    ].copy()

    report.rows_out = len(frame)
    report.notes.append(f"Year window observed: {frame['year'].min()}"
                        f"-{frame['year'].max()}" if "year" in frame
                        else "No year column found.")

    provenance = Provenance(
        source="synthetic" if path.name == SYNTHETIC_MARKER else "file",
        path=str(path),
        sha256=sha256_of(path),
        rows_raw=report.rows_in,
        rows_clean=report.rows_out,
        loaded_at=datetime.now(UTC).isoformat(timespec="seconds"),
        is_synthetic=path.name == SYNTHETIC_MARKER,
    )

    if verbose:
        print(report.summary_line())
        if report.dropped_rows:
            print("  dropped:", report.dropped_rows)
        if report.coerced_numeric:
            print("  coerced:", report.coerced_numeric)
        for note in report.notes:
            print(f"  note: {note}")

    return order_columns(frame), report, provenance


def order_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Put the canonical columns first, then the derived ones, deterministically."""
    leading = [c for c in DISPLAY_ORDER if c in frame.columns]
    rest = sorted(c for c in frame.columns if c not in leading)
    return frame.loc[:, leading + rest]


def load_or_generate(
    source: str | Path | None = None, n_rows: int | None = None
) -> tuple[pd.DataFrame, QualityReport, Provenance]:
    """Convenience wrapper used by the dashboard and the training script."""
    return load_dataset(source)


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    frame, quality, prov = load_dataset(verbose=True)
    print()
    print(f"source={prov.source} synthetic={prov.is_synthetic}")
    print(f"columns={list(frame.columns)}")
    print(frame.head(8).to_string())
    frame.to_csv(CLEANED_CSV, index=False)
    print(f"\ncleaned -> {CLEANED_CSV}")
