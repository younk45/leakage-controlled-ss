"""Independent structural and numerical checks for a completed analysis run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    relative = np.abs(observed - predicted) / observed
    return {
        "MAE": float(mean_absolute_error(observed, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(observed, predicted))),
        "Pred25": float(100.0 * np.mean(relative <= 0.25)),
        "R2": float(r2_score(observed, predicted)),
    }


def verify(run_dir: Path, config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise AssertionError(f"Run is not complete: {manifest['status']}")

    dataset_path = (config_path.parent / config["dataset"]["path"]).resolve()
    dataset_hash = sha256_file(dataset_path)
    if dataset_hash != config["dataset"]["sha256"]:
        raise AssertionError("Dataset hash does not match config")
    data = pd.read_csv(dataset_path)
    id_column = config["dataset"]["id_column"]
    target_column = config["dataset"]["target_column"]

    fold_assignments = pd.read_csv(run_dir / "outer_fold_assignments.csv")
    repetitions = len(config["outer_cv"]["seeds"])
    expected_assignments = len(data) * repetitions
    if len(fold_assignments) != expected_assignments:
        raise AssertionError("Unexpected number of outer test assignments")
    project_repeat_counts = fold_assignments.groupby(["repetition", "project_id"]).size()
    if not (project_repeat_counts == 1).all():
        raise AssertionError("A project is not assigned to exactly one test fold per repetition")

    bootstrap = np.load(run_dir / "bootstrap_project_indices.npy", allow_pickle=False)
    expected_bootstrap_shape = (config["bootstrap"]["replicates"], len(data))
    if bootstrap.shape != expected_bootstrap_shape:
        raise AssertionError(f"Bootstrap shape {bootstrap.shape} != {expected_bootstrap_shape}")
    if bootstrap.min() < 0 or bootstrap.max() >= len(data):
        raise AssertionError("Bootstrap contains an invalid project index")

    predictions = pd.read_csv(run_dir / "oof_predictions_full_precision.csv.gz")
    group_columns = ["family", "feature_set", "method"]
    key_columns = [*group_columns, "repetition", "project_id"]
    if predictions.duplicated(key_columns).any():
        raise AssertionError("Duplicate out-of-fold prediction key")
    if not np.isfinite(predictions["predicted_effort"]).all():
        raise AssertionError("Non-finite prediction")
    if (predictions["predicted_effort"] < config["prediction_floor"]).any():
        raise AssertionError("Prediction below configured floor")
    group_sizes = predictions.groupby(group_columns).size()
    expected_group_size = len(data) * repetitions
    if not (group_sizes == expected_group_size).all():
        raise AssertionError("At least one prediction group is incomplete")

    observed_lookup = data.set_index(id_column)[target_column]
    expected_observed = predictions["project_id"].map(observed_lookup).to_numpy(dtype=float)
    if not np.array_equal(expected_observed, predictions["observed_effort"].to_numpy(dtype=float)):
        raise AssertionError("Observed effort in predictions does not match the hashed dataset")

    stored_metrics = pd.read_csv(run_dir / "metrics_full_precision.csv").set_index(
        [*group_columns, "metric"]
    )
    maximum_metric_error = 0.0
    for group_key, group in predictions.groupby(group_columns, sort=True):
        recalculated = metrics(
            group["observed_effort"].to_numpy(dtype=float),
            group["predicted_effort"].to_numpy(dtype=float),
        )
        for metric_name, value in recalculated.items():
            stored = float(stored_metrics.loc[(*group_key, metric_name), "estimate_full_precision"])
            maximum_metric_error = max(maximum_metric_error, abs(value - stored))
            if not np.isclose(value, stored, rtol=0.0, atol=1e-10):
                raise AssertionError(f"Stored {metric_name} differs for {group_key}: {stored} vs {value}")

    central = predictions[
        (predictions["family"] == "scale_nested_et")
        & (predictions["feature_set"] == "strict6")
        & (predictions["method"] == "Global")
    ].sort_values(["repetition", "fold", "project_id"])
    benchmark = predictions[
        (predictions["family"] == "benchmark")
        & (predictions["feature_set"] == "strict6")
        & (predictions["method"] == "Tuned Extra Trees")
    ].sort_values(["repetition", "fold", "project_id"])
    if not np.array_equal(
        central["predicted_effort"].to_numpy(dtype=float),
        benchmark["predicted_effort"].to_numpy(dtype=float),
    ):
        raise AssertionError("Duplicated Global prediction vectors are not identical")
    for metric_name in ("MAE", "RMSE", "Pred25", "R2"):
        central_summary = stored_metrics.loc[("scale_nested_et", "strict6", "Global", metric_name)]
        benchmark_summary = stored_metrics.loc[("benchmark", "strict6", "Tuned Extra Trees", metric_name)]
        columns = ["estimate_full_precision", "ci_2_5", "ci_97_5"]
        if not np.array_equal(
            central_summary[columns].to_numpy(dtype=float),
            benchmark_summary[columns].to_numpy(dtype=float),
        ):
            raise AssertionError(f"Identical predictions have different {metric_name} summaries")

    hash_mismatches: dict[str, dict[str, str]] = {}
    for relative_path, expected_hash in manifest["artifact_sha256"].items():
        artifact = run_dir / relative_path
        observed_hash = sha256_file(artifact)
        if observed_hash != expected_hash:
            hash_mismatches[relative_path] = {"expected": expected_hash, "observed": observed_hash}
    if hash_mismatches:
        raise AssertionError(f"Artifact hash mismatch: {hash_mismatches}")

    return {
        "status": "verified",
        "run_dir": str(run_dir.resolve()),
        "dataset_sha256": dataset_hash,
        "rows": len(data),
        "outer_test_assignments": len(fold_assignments),
        "prediction_groups": int(len(group_sizes)),
        "predictions_per_group": int(expected_group_size),
        "total_predictions": int(len(predictions)),
        "bootstrap_shape": list(bootstrap.shape),
        "maximum_absolute_metric_recalculation_error": maximum_metric_error,
        "identical_global_prediction_vectors": True,
        "identical_global_metric_estimates_and_intervals": True,
        "manifest_artifacts_rehashed": len(manifest["artifact_sha256"]),
        "manifest_hash_mismatches": hash_mismatches,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    report = verify(args.run_dir.resolve(), args.config.resolve())
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.resolve().write_text(rendered, encoding="utf-8")
    print(rendered, end="")

