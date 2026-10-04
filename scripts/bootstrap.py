"""First-run setup: generate data, build the basemap, train the models.

    python scripts/bootstrap.py              # full run, trains if needed
    python scripts/bootstrap.py --force      # regenerate and retrain
    python scripts/bootstrap.py --fast       # quick models for development
    python scripts/bootstrap.py --data-only  # skip training entirely

Safe to run repeatedly: each step is skipped when its output already exists,
unless --force is given.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import (  # noqa: E402 - needs the sys.path insert above
    GEOJSON_PATH,
    METRICS_PATH,
    MODEL_PATHS,
    RAW_DATA_PATH,
    YEAR_MAX,
    YEAR_MIN,
)


def _step(number: int, total: int, title: str) -> None:
    print(f"\n[{number}/{total}] {title}", flush=True)


def ensure_dataset(force: bool, rows: int | None) -> Path:
    if RAW_DATA_PATH.exists() and not force:
        print(f"    already present: {RAW_DATA_PATH}")
        return RAW_DATA_PATH

    from config.settings import SEED
    from src.synthetic import generate_synthetic_dataset

    print(f"    generating {rows or 'default'} rows (seed {SEED}) ...")
    frame = generate_synthetic_dataset(
        n_rows=rows or 60_000, seed=SEED, dirty=True
    )
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RAW_DATA_PATH, index=False)
    print(f"    wrote {len(frame):,} rows -> {RAW_DATA_PATH}")
    return RAW_DATA_PATH


def ensure_basemap(force: bool) -> None:
    if GEOJSON_PATH.exists() and not force:
        print(f"    already present: {GEOJSON_PATH}")
        return
    from src.geo import build_basemap

    ok, note = build_basemap(force=force, timeout=90)
    print(f"    {'ok' if ok else 'FAILED'}: {note}")


def train(force: bool, fast: bool) -> None:
    present = all(path.exists() for path in MODEL_PATHS.values()) and METRICS_PATH.exists()
    if present and not force:
        print("    artefacts already present; pass --force to retrain")
        return

    command = [
        sys.executable, "-m", "src.train_models",
        "--split", "random",
        "--bootstrap", "5" if fast else "20",
    ]
    if fast:
        command.append("--fast")
    print("    " + " ".join(command))
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode != 0:
        raise SystemExit(f"training failed with code {result.returncode}")


def verify() -> bool:
    print("\nVerifying")
    ok = True
    for label, path in (
        ("dataset", RAW_DATA_PATH),
        ("basemap", GEOJSON_PATH),
        *[(f"model {name}", p) for name, p in MODEL_PATHS.items()],
        ("metrics", METRICS_PATH),
    ):
        exists = path.exists()
        ok = ok and exists
        size = f"{path.stat().st_size / 1_048_576:,.1f} MB" if exists else "-"
        print(f"    {'OK ' if exists else 'MISSING'}  {label:<16} {size}")

    try:
        from src.services import registry

        frame = registry.load_dataset()
        print(f"    OK   prepared frame  {len(frame):,} rows x {len(frame.columns)} cols")
        print(f"    OK   states {frame['state'].nunique()}, "
              f"crops {frame['crop'].nunique()}, "
              f"zones {frame['recommended_zone'].nunique()}")
        forecast = registry.load_bundles()
        print(f"    OK   bundles {sorted(forecast)}")
    except Exception as exc:
        ok = False
        print(f"    FAILED to load artefacts: {type(exc).__name__}: {exc}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="regenerate data and retrain even if outputs exist")
    parser.add_argument("--fast", action="store_true",
                        help="quick training run (5 bootstrap members, no CV)")
    parser.add_argument("--data-only", action="store_true",
                        help="generate the dataset and basemap but do not train")
    parser.add_argument("--rows", type=int, default=None,
                        help="rows to generate (default 60000)")
    args = parser.parse_args()

    started = time.time()
    total = 2 if args.data_only else 3
    print(f"AgriYield AI bootstrap  ({YEAR_MIN}-{YEAR_MAX} data window)")

    _step(1, total, "Dataset")
    ensure_dataset(args.force, args.rows)

    _step(2, total, "Basemap")
    ensure_basemap(args.force)

    if not args.data_only:
        _step(3, total, "Models")
        train(args.force, args.fast)

    print(f"\nDone in {time.time() - started:,.1f}s")
    if verify():
        print("\nRun the app:\n    python -m streamlit run app.py")
        return 0
    print("\nSetup incomplete. See the failures above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
