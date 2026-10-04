"""Artifact registry: the single place the app loads trained models.

Everything expensive -- joblib deserialisation, the 57k-row cleaned frame, the
prediction cache, the basemap -- is loaded at most once per process and reused.
That matters for Streamlit, which re-runs the script top to bottom on every
interaction.

Caching is keyed on each file's ``(size, mtime)``, so retraining in a terminal
invalidates the running app's cache automatically instead of leaving it serving
stale models until someone hits restart.

This module is deliberately Streamlit-free.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from config.settings import (
    METRICS_PATH,
    MODEL_DIR,
    PROCESSED_DIR,
)


class ArtifactMissing(RuntimeError):
    """Raised when a trained artefact is absent, with a fix in the message."""


REMEDY = (
    "Run `python -m src.train_models` (add `--fast` for a quick pass) and "
    "reload. `python scripts/bootstrap.py` does this for you."
)


@dataclass(frozen=True)
class Stamped:
    """A value plus the file stamp it was loaded from."""

    value: Any
    stamp: tuple[int, float]


def _stamp(path: Path) -> tuple[int, float]:
    stat = path.stat()
    return (stat.st_size, stat.st_mtime)


class _Cache:
    """Stamp-keyed memo, so a retrain transparently invalidates entries."""

    def __init__(self) -> None:
        self._store: dict[str, Stamped] = {}

    def get_or_load(self, key: str, path: Path, loader: Callable[[], Any]) -> Any:
        current = _stamp(path)
        cached = self._store.get(key)
        if cached is not None and cached.stamp == current:
            return cached.value
        value = loader()
        self._store[key] = Stamped(value, current)
        return value

    def clear(self) -> None:
        self._store.clear()

    def describe(self) -> pd.DataFrame:
        rows = [
            {
                "artifact": key,
                "size_kb": round(entry.stamp[0] / 1024, 1),
                "loaded_at": datetime.fromtimestamp(
                    entry.stamp[1], tz=UTC
                ).isoformat(timespec="seconds"),
            }
            for key, entry in self._store.items()
        ]
        return pd.DataFrame(rows)


_CACHE = _Cache()


def bundle_paths() -> dict[str, Path]:
    """Where each target's bundle is expected to live."""
    return {
        "production_quintals": MODEL_DIR / "xgboost_yield.joblib",
        "cost": MODEL_DIR / "xgboost_cost.joblib",
    }


def missing_artifacts() -> list[str]:
    """Human-readable names of artefacts that are not on disk yet."""
    missing = [path.name for path in bundle_paths().values() if not path.exists()]
    if not METRICS_PATH.exists():
        missing.append(METRICS_PATH.name)
    return missing


def artifacts_ready() -> bool:
    return not missing_artifacts()


def load_bundle(target: str):
    """Load a trained :class:`~src.models.ModelBundle` for ``target``."""
    paths = bundle_paths()
    if target not in paths:
        raise KeyError(f"unknown target {target!r}; expected one of {sorted(paths)}")
    path = paths[target]
    if not path.exists():
        raise ArtifactMissing(f"{path.name} not found. {REMEDY}")

    import joblib

    return _CACHE.get_or_load(f"bundle:{target}", path, lambda: joblib.load(path))


def load_bundles() -> dict[str, Any]:
    return {target: load_bundle(target) for target in bundle_paths()}


def load_metrics() -> dict[str, Any]:
    """The training run's metrics/provenance record."""
    if not METRICS_PATH.exists():
        raise ArtifactMissing(f"{METRICS_PATH.name} not found. {REMEDY}")
    return _CACHE.get_or_load(
        "metrics",
        METRICS_PATH,
        lambda: json.loads(METRICS_PATH.read_text(encoding="utf-8")),
    )


def cache_path() -> Path:
    return PROCESSED_DIR / "prediction_cache.parquet"


def load_prediction_cache() -> pd.DataFrame:
    """Held-out predictions and intervals, for the diagnostics page."""
    path = cache_path()
    if not path.exists():
        raise ArtifactMissing(f"{path.name} not found. {REMEDY}")
    return _CACHE.get_or_load("cache", path, lambda: pd.read_parquet(path))


def load_clean() -> pd.DataFrame:
    """The cleaned dataset as loaded, before feature engineering.

    Has no ``is_outlier`` column: flagging happens in :func:`prepare`, which is
    part of the training pipeline, not the loader.
    """
    from src.data_loader import find_source_csv
    from src.data_loader import load_dataset as _load_dataset

    try:
        source = find_source_csv()
    except Exception:
        return _load_dataset()[0]
    return _CACHE.get_or_load("clean", source, lambda: _load_dataset()[0])


def load_dataset() -> pd.DataFrame:
    """The dataset prepared exactly as training prepared it.

    This runs :func:`src.preprocessing.prepare`, which engineers the numeric
    features *and* adds the ``is_outlier`` flag. Callers must use this rather
    than :func:`load_clean`, because aggregations over the unflagged frame would
    silently include the entry-error rows that training excluded -- the app
    would then disagree with its own training report.
    """
    from src.data_loader import find_source_csv
    from src.preprocessing import prepare

    try:
        source = find_source_csv()
    except Exception:
        return prepare(load_clean())

    def _build() -> pd.DataFrame:
        clean, _, _ = _raw_loader()
        return prepare(clean)

    return _CACHE.get_or_load("dataset", source, _build)


def _raw_loader():
    from src.data_loader import load_dataset

    return load_dataset()


def load_geojson() -> tuple[dict | None, str]:
    """The India basemap, as ``(payload, message)``.

    Absent basemap is not an error -- the overview falls back to a bar chart.
    """
    from config.settings import GEOJSON_PATH
    from src.geo import load_geojson as _load

    if not GEOJSON_PATH.exists():
        return None, "basemap not fetched (run scripts/bootstrap.py)"

    def _read() -> dict:
        payload, _ = _load(GEOJSON_PATH)
        if payload is None:
            raise ArtifactMissing(f"{GEOJSON_PATH.name} is unreadable")
        return payload

    try:
        payload = _CACHE.get_or_load("geojson", GEOJSON_PATH, _read)
    except ArtifactMissing as exc:
        return None, str(exc)
    return payload, f"{len(payload.get('features', []))} state polygons"


def training_status() -> dict[str, Any]:
    """Everything the sidebar needs to describe the current artefacts."""
    missing = missing_artifacts()
    status: dict[str, Any] = {
        "ready": not missing,
        "missing": missing,
        "generated_at": None,
        "seed": None,
        "provenance": None,
        "is_synthetic": True,
        "rows": None,
        "split": None,
        "interval": None,
        "lightgbm": None,
        "metrics": {},
        "libraries": {},
        "data_quality": {},
        "elapsed_seconds": None,
    }
    if missing:
        return status
    try:
        metrics = load_metrics()
    except ArtifactMissing:
        return status

    status.update(
        generated_at=metrics.get("generated_at"),
        seed=metrics.get("seed"),
        provenance=metrics.get("provenance", {}),
        is_synthetic=bool(metrics.get("provenance", {}).get("synthetic", True)),
        rows=metrics.get("data_quality", {}).get("rows_out"),
        split=metrics.get("split", {}),
        interval=metrics.get("interval", {}),
        lightgbm={
            target: (block.get("lgbm") or {}).get(f"{target}_r2")
            for target, block in metrics.get("models", {}).items()
        },
        metrics=metrics.get("models", {}),
        libraries=metrics.get("libraries", {}),
        data_quality=metrics.get("data_quality", {}),
        elapsed_seconds=metrics.get("elapsed_seconds"),
    )
    return status


def clear_cache() -> None:
    """Drop every cached artefact (used by tests and the reload button)."""
    _CACHE.clear()


def cache_report() -> pd.DataFrame:
    return _CACHE.describe()
