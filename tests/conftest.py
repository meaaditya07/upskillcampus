"""Shared fixtures.

The trained artefacts are session-scoped: they cost ~35s to produce and the
tests only read them, so loading once per session keeps the suite fast without
letting one test's mutation leak into another.
"""

from __future__ import annotations

import warnings

import pytest

warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def raw_csv(tmp_path_factory):
    """A small, freshly generated synthetic CSV with all defects present."""
    from config.settings import SEED
    from src.synthetic import generate_synthetic_dataset

    frame = generate_synthetic_dataset(n_rows=3_000, seed=SEED, dirty=True)
    path = tmp_path_factory.mktemp("raw") / "crop_production_synthetic.csv"
    frame.to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def clean_frame(raw_csv):
    """The loader's cleaned output for :func:`raw_csv`."""
    from src.data_loader import load_dataset

    frame, report, provenance = load_dataset(source=raw_csv)
    return frame, report, provenance


@pytest.fixture(scope="session")
def prepared(clean_frame):
    from src.preprocessing import prepare

    frame, _, _ = clean_frame
    return prepare(frame)


@pytest.fixture(scope="session")
def dataset():
    """The real prepared dataset (needs the generated CSV on disk)."""
    from src.services import registry

    return registry.load_dataset()


@pytest.fixture(scope="session")
def bundles():
    """Trained bundles; skips the whole module if artefacts are absent."""
    from src.services import registry

    missing = registry.missing_artifacts()
    if missing:
        pytest.skip(f"trained artefacts missing ({', '.join(missing)}); run src.train_models")
    return registry.load_bundles()
