"""Training entry point.

    python -m src.train_models --split random --bootstrap 20
    python -m src.train_models --fast            # quick iteration
    python -m src.train_models --split group_zone --no-bootstrap

Produces, under ``models/``:

* ``yield_model.joblib`` / ``cost_model.joblib`` -- point + quantile + bootstrap
* ``yield_model_lgbm.joblib`` / ``cost_model_lgbm.joblib`` -- benchmark arm
  (skipped with a recorded reason if LightGBM cannot be imported)
* ``metrics.json`` -- every number the dashboard and the report quote
* ``prediction_cache.parquet`` -- test-set predictions and intervals, so the
  diagnostics page never has to re-predict 11,000 rows on every rerun
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from config.settings import (
    APP_NAME,
    CURRENCY,
    DEFAULT_BOOTSTRAP_N,
    INTERVAL_QUANTILE_HIGH,
    INTERVAL_QUANTILE_LOW,
    METRICS_PATH,
    MODEL_DIR,
    PROCESSED_DIR,
    SEED,
)
from src.data_loader import load_dataset
from src.evaluate import (
    baseline_table,
    calibration_table,
    cross_validate,
    evaluate_baselines,
    interval_coverage,
    permutation_importance,
    residual_frame,
    summarise_folds,
)
from src.models import (
    COST_TARGET,
    YIELD_TARGET,
    build_pipeline,
    fit_bundle,
    fit_comparison_model,
    lightgbm_available,
)
from src.preprocessing import (
    audit_feature_sets,
    features_for,
    make_split,
    prepare,
    split_train_calibration,
)

TARGET_LABELS = {
    YIELD_TARGET: "Production (quintals)",
    COST_TARGET: f"Cultivation cost ({CURRENCY})",
}


def log(message: str = "") -> None:
    print(message, flush=True)


def banner(title: str) -> None:
    log("")
    log("=" * 72)
    log(title)
    log("=" * 72)


def train(
    split_strategy: str = "random",
    n_bootstrap: int = DEFAULT_BOOTSTRAP_N,
    calibration_size: float = 0.15,
    test_size: float = 0.2,
    fast: bool = False,
    run_cv: bool = True,
    run_permutation: bool = True,
    source: str | None = None,
    include_lgbm: bool = True,
) -> dict:
    """Run the full training pipeline and write every artefact."""
    started = time.time()
    timestamp = datetime.now(UTC).isoformat(timespec="seconds")

    banner(f"{APP_NAME} -- model training")
    log(f"started      {timestamp}")
    log(f"python       {platform.python_version()} ({platform.system()})")
    log(f"split        {split_strategy}")
    log(f"bootstrap    {n_bootstrap}")

    # ---------------------------------------------------------------- data
    banner("1/6  Load and clean")
    frame, quality, provenance = load_dataset(source, verbose=True)
    log(f"  provenance: {provenance.source} ({provenance.path})")
    if provenance.is_synthetic:
        log("  *** SYNTHETIC DATA -- results are illustrative, not evidence ***")

    frame = prepare(frame)
    outliers = int(frame["is_outlier"].sum())
    log(f"  flagged {outliers:,} likely entry-error rows (excluded from training)")

    # Only train on clean rows, but keep the full frame for EDA.
    clean = frame.loc[~frame["is_outlier"]].reset_index(drop=True)

    # --------------------------------------------------------------- split
    banner("2/6  Split")
    split = make_split(clean, split_strategy, test_size, SEED)
    fit_frame, calib_frame = split_train_calibration(
        split.train.reset_index(drop=True), calibration_size, SEED
    )
    log(f"  {split.description}")
    log(f"  fit={len(fit_frame):,}  calibration={len(calib_frame):,}  "
        f"test={len(split.test):,}")

    metrics: dict = {
        "app": APP_NAME,
        "generated_at": timestamp,
        "seed": SEED,
        "python": platform.python_version(),
        "platform": platform.system(),
        "libraries": _library_versions(),
        "provenance": provenance.to_dict(),
        "data_quality": quality.to_dict(),
        "split": {
            "strategy": split_strategy,
            "description": split.description,
            "test_size": test_size,
            "calibration_size": calibration_size,
            "n_fit": len(fit_frame),
            "n_calibration": len(calib_frame),
            "n_test": len(split.test),
            "n_outliers_excluded": outliers,
        },
        "interval": {
            "nominal_coverage": 1.0 - 2.0 * INTERVAL_QUANTILE_LOW,
            "quantile_alphas": [INTERVAL_QUANTILE_LOW, INTERVAL_QUANTILE_HIGH],
            "method": (
                "Quantile gradient boosting (reg:quantileerror) on log1p(target), "
                "widened by a scalar factor fitted on a held-out calibration "
                "split. The bootstrap ensemble is reported as a stability "
                "diagnostic only -- see the note in predict_with_interval."
            ),
        },
        "feature_audit": audit_feature_sets().to_dict(orient="records"),
        "models": {},
    }

    artifacts: dict[str, object] = {}

    # -------------------------------------------------------------- models
    banner("3/6  Train models")
    for target in (YIELD_TARGET, COST_TARGET):
        label = TARGET_LABELS[target]
        log(f"  -- {label}")
        bundle = fit_bundle(
            target,
            fit_frame,
            split.test,
            calibration=calib_frame,
            kind="xgboost",
            n_bootstrap=n_bootstrap,
            progress=lambda m: log("     " + m),
        )

        X_test = split.test.loc[:, bundle.features]
        y_test = split.test[target]
        interval = bundle.predict_with_interval(X_test)
        coverage = interval_coverage(
            y_test.to_numpy(dtype=float), interval.lower, interval.upper
        )
        bundle.set_interval_coverage(coverage["coverage"])

        quant_only = interval_coverage(
            y_test.to_numpy(dtype=float),
            interval.quantile_lower,
            interval.quantile_upper,
        )
        log(
            f"     R2={bundle.metrics.get(target + '_r2', float('nan')):.4f}  "
            f"RMSE={bundle.metrics.get(target + '_rmse', float('nan')):,.1f}  "
            f"coverage={coverage['coverage']:.1f}% (nominal "
            f"{metrics['interval']['nominal_coverage'] * 100:.0f}%, raw quantiles "
            f"{quant_only['coverage']:.1f}%)"
        )

        baselines = evaluate_baselines(fit_frame, split.test, target)
        table = baseline_table(baselines, bundle.metrics, target)
        log("     " + table.to_string(index=False).replace("\n", "\n     "))

        record: dict = {
            "label": label,
            "target": target,
            "kind": bundle.kind,
            "metrics": bundle.metrics,
            "baselines": baselines,
            "baseline_table": table.to_dict(orient="records"),
            "interval": {
                "calibration_factor": bundle.calibration_factor,
                "test_coverage": coverage,
                "raw_quantile_coverage": quant_only,
                "bootstrap_stability_pct": bundle.bootstrap_stability(
                    split.test.head(800)
                ),
                "n_bootstrap_members": len(bundle.bootstrap),
            },
            "feature_importance": bundle.feature_importances(30).to_dict(
                orient="records"
            ),
            "features": bundle.features,
            "n_fit": bundle.train_rows,
            "n_test": bundle.test_rows,
        }

        if run_permutation:
            log("     permutation importance...")
            perm = permutation_importance(
                bundle.pipeline,
                split.test.loc[:, bundle.features],
                y_test.to_numpy(dtype=float),
                bundle.features,
                n_repeats=3,
                n_rows=1200,
                random_state=SEED,
            )
            record["permutation_importance"] = perm.to_dict(orient="records")
            log("     " + perm.head(6).to_string(index=False).replace("\n", "\n     "))

        if run_cv:
            log("     5-fold cross-validation...")
            folds = cross_validate(
                lambda target=target: build_pipeline(target, kind="xgboost"),
                fit_frame,
                target,
                features_for(target),
                n_splits=5,
                seed=SEED,
            )
            record["cv_folds"] = folds.to_dict(orient="records")
            record["cv_summary"] = summarise_folds(folds, target).to_dict(
                orient="records"
            )
            log("     " + summarise_folds(folds, target).to_string(index=False)
                .replace("\n", "\n     "))

        # Cache predictions for the diagnostics page.
        cache = residual_frame(y_test.to_numpy(dtype=float), interval.point)
        cache["lower"] = interval.lower
        cache["upper"] = interval.upper
        cache["bootstrap_std"] = interval.bootstrap_std
        cache["state"] = split.test["state"].to_numpy()
        cache["crop"] = split.test["crop"].to_numpy()
        cache["target"] = target
        artifacts[f"cache_{target}"] = cache
        record["calibration_curve"] = calibration_table(
            y_test.to_numpy(dtype=float), interval.lower, interval.upper
        ).to_dict(orient="records")

        metrics["models"][target] = record
        artifacts[f"model_{target}"] = bundle

    # ------------------------------------------------------- benchmark arm
    banner("4/6  Benchmark booster (LightGBM)")
    available, reason = lightgbm_available()
    if include_lgbm and available:
        for target in (YIELD_TARGET, COST_TARGET):
            pipeline, err = fit_comparison_model(target, fit_frame, split.test)
            if pipeline is None:
                log(f"  {target}: skipped -- {err}")
                metrics["models"].setdefault(target, {})["lgbm_error"] = err
                continue
            features = features_for(target)
            predicted = np.clip(
                pipeline.predict(split.test.loc[:, features]), 0.0, None
            )
            from src.evaluate import regression_metrics

            lgbm_metrics = regression_metrics(
                split.test[target].to_numpy(dtype=float), predicted, target=target
            )
            metrics["models"][target]["lgbm"] = lgbm_metrics
            artifacts[f"lgbm_{target}"] = pipeline
            log(
                f"  {target}: R2={lgbm_metrics[target + '_r2']:.4f} "
                f"(XGBoost {metrics['models'][target]['metrics'][target + '_r2']:.4f})"
            )
    else:
        log(f"  LightGBM unavailable ({reason or 'disabled by flag'}); skipping.")
        metrics["lightgbm_available"] = False
        metrics["lightgbm_reason"] = reason or "disabled by --no-lgbm"

    # ---------------------------------------------------------- write out
    banner("5/6  Save artefacts")
    import joblib

    for key, bundle in artifacts.items():
        if key.startswith("model_"):
            path = MODEL_DIR / f"{bundle.kind}_{_short(bundle.target)}.joblib"
        elif key.startswith("lgbm_"):
            path = MODEL_DIR / f"lightgbm_{_short(key.replace('lgbm_', ''))}.joblib"
        else:
            continue
        joblib.dump(bundle, path)
        log(f"  wrote {path.name} ({path.stat().st_size / 1e6:.2f} MB)")

    cache_frame = pd.concat(
        [v for k, v in artifacts.items() if k.startswith("cache_")],
        ignore_index=True,
    )
    cache_path = PROCESSED_DIR / "prediction_cache.parquet"
    cache_frame.to_parquet(cache_path, index=False)
    log(f"  wrote {cache_path.name} ({len(cache_frame):,} rows)")

    metrics["elapsed_seconds"] = round(time.time() - started, 1)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, default=str))
    log(f"  wrote {METRICS_PATH.name}")

    banner("6/6  Summary")
    for target, record in metrics["models"].items():
        m = record.get("metrics", {})
        r2 = m.get(target + "_r2", float("nan"))
        base = next(
            (b for b in record.get("baselines", []) if b["model"] == "Global mean"),
            {},
        )
        base_r2 = base.get(target + "_r2", float("nan"))
        log(
            f"  {TARGET_LABELS[target]:<28} R2={r2:>7.4f}  "
            f"vs global-mean baseline {base_r2:>7.4f}  "
            f"coverage={record['interval']['test_coverage']['coverage']:.1f}%"
        )
    log(f"\ndone in {metrics['elapsed_seconds']}s")

    return metrics


def _short(target: str) -> str:
    return "yield" if target == YIELD_TARGET else "cost"


def _library_versions() -> dict:
    versions = {}
    for name in ("pandas", "numpy", "sklearn", "xgboost", "lightgbm", "joblib"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[name] = "not installed"
    return versions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.train_models",
        description=f"{APP_NAME}: train the yield and cost models.",
    )
    parser.add_argument(
        "--split",
        default="random",
        choices=["random", "group_state", "group_zone", "chronological"],
        help="how to hold out test data (default: random)",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=DEFAULT_BOOTSTRAP_N,
        help="bootstrap ensemble size (default: %(default)s, 0 disables)",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="test fraction (default: 0.2)"
    )
    parser.add_argument(
        "--calibration-size",
        type=float,
        default=0.15,
        help="fraction of train held out to calibrate intervals (default: 0.15)",
    )
    parser.add_argument("--data", default=None, help="path to a CSV to load")
    parser.add_argument(
        "--fast", action="store_true", help="skip CV, permutation and LightGBM"
    )
    parser.add_argument("--no-cv", action="store_true", help="skip cross-validation")
    parser.add_argument(
        "--no-permutation", action="store_true", help="skip permutation importance"
    )
    parser.add_argument("--no-lgbm", action="store_true", help="skip LightGBM arm")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    train(
        split_strategy=args.split,
        n_bootstrap=args.bootstrap,
        calibration_size=args.calibration_size,
        test_size=args.test_size,
        fast=args.fast,
        run_cv=not (args.no_cv or args.fast),
        run_permutation=not (args.no_permutation or args.fast),
        source=args.data,
        include_lgbm=not (args.no_lgbm or args.fast),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
