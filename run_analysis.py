"""Deterministic reviewer-revision analysis for IJIES manuscript 20265081.

The implementation prioritizes temporal feature provenance, outer-fold isolation,
matched comparisons, reusable bootstrap indices, and inspectable artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/ijies-20265081-matplotlib")

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
    category=UserWarning,
)

import lightgbm
import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost
from joblib import parallel_backend
from lightgbm import LGBMRegressor
from scipy.stats import rankdata, t as student_t, wilcoxon
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, StackingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from xgboost import XGBRegressor


SCENARIOS = ("Global", "Global + MI", "Quantile", "Quantile + MI")
CATEGORIES = ("Small", "Medium", "Large")
METRIC_NAMES = ("MAE", "RMSE", "Pred25", "R2")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def derive_seed(namespace: str, *parts: Any) -> int:
    material = "|".join([namespace, *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:4], byteorder="big", signed=False)


def load_config(path: Path, smoke: bool) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if smoke:
        config["protocol_id"] += "-smoke"
        config["outer_cv"]["n_splits"] = 3
        config["outer_cv"]["seeds"] = [config["outer_cv"]["seeds"][0]]
        config["inner_cv"]["n_splits"] = 2
        config["inner_cv"]["search_iterations"] = 2
        config["bootstrap"]["replicates"] = 100
        config["feature_sets"] = {"strict6": config["feature_sets"]["strict6"], "leaky14": config["feature_sets"]["leaky14"]}
    return config


def load_and_validate_data(config: dict[str, Any], config_path: Path) -> tuple[pd.DataFrame, Path]:
    dataset_config = config["dataset"]
    dataset_path = (config_path.parent / dataset_config["path"]).resolve()
    actual_hash = sha256_file(dataset_path)
    if actual_hash != dataset_config["sha256"]:
        raise ValueError(f"Dataset hash mismatch: expected {dataset_config['sha256']}, observed {actual_hash}")

    data = pd.read_csv(dataset_path)
    if len(data) != dataset_config["expected_rows"]:
        raise ValueError(f"Expected {dataset_config['expected_rows']} rows, found {len(data)}")
    required = {
        dataset_config["id_column"],
        dataset_config["target_column"],
        *(feature for features in config["feature_sets"].values() for feature in features),
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if data[dataset_config["id_column"]].duplicated().any():
        raise ValueError("Project identifiers are not unique")
    if data[dataset_config["target_column"]].isna().any():
        raise ValueError("Target contains missing values")
    if (data[dataset_config["target_column"]] <= 0).any():
        raise ValueError("Pred(25) requires strictly positive observed effort")
    return data, dataset_path


@dataclass(frozen=True)
class OuterFold:
    repetition: int
    outer_seed: int
    fold: int
    train_index: np.ndarray
    test_index: np.ndarray


def make_outer_folds(data: pd.DataFrame, config: dict[str, Any]) -> list[OuterFold]:
    folds: list[OuterFold] = []
    n_splits = int(config["outer_cv"]["n_splits"])
    for repetition, outer_seed in enumerate(config["outer_cv"]["seeds"], start=1):
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=int(outer_seed))
        for fold, (train_index, test_index) in enumerate(splitter.split(data), start=1):
            folds.append(OuterFold(repetition, int(outer_seed), fold, train_index, test_index))
    return folds


def save_outer_folds(folds: list[OuterFold], data: pd.DataFrame, id_column: str, output_dir: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    project_ids = data[id_column].to_numpy()
    for split in folds:
        for row_index in split.test_index:
            records.append(
                {
                    "repetition": split.repetition,
                    "outer_seed": split.outer_seed,
                    "fold": split.fold,
                    "row_index_zero_based": int(row_index),
                    "project_id": project_ids[row_index],
                    "role": "test",
                    "train_is_complement": True,
                }
            )
    frame = pd.DataFrame.from_records(records)
    frame.to_csv(output_dir / "outer_fold_assignments.csv", index=False)
    expected = len(data) * len({split.repetition for split in folds})
    if len(frame) != expected:
        raise AssertionError(f"Expected {expected} test assignments, found {len(frame)}")
    return frame


def make_bootstrap_indices(data: pd.DataFrame, config: dict[str, Any], output_dir: Path) -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(int(config["bootstrap"]["seed"])))
    indices = rng.integers(
        low=0,
        high=len(data),
        size=(int(config["bootstrap"]["replicates"]), len(data)),
        endpoint=False,
        dtype=np.int32,
    )
    np.save(output_dir / "bootstrap_project_indices.npy", indices, allow_pickle=False)
    descriptor = {
        **config["bootstrap"],
        "shape": list(indices.shape),
        "dtype": str(indices.dtype),
        "indexing": "zero-based row positions in dataset order",
        "file_sha256": sha256_file(output_dir / "bootstrap_project_indices.npy"),
    }
    write_json(output_dir / "bootstrap_descriptor.json", descriptor)
    return indices


def feature_frame(data: pd.DataFrame, features: list[str], indices: np.ndarray) -> pd.DataFrame:
    return data.iloc[indices][features].apply(pd.to_numeric, errors="coerce")


def model_pipeline(model: Any) -> Pipeline:
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])


def make_model(model_name: str, seed: int) -> Any:
    if model_name == "extra_trees":
        return ExtraTreesRegressor(random_state=seed, n_jobs=1, criterion="squared_error")
    if model_name == "random_forest":
        return RandomForestRegressor(random_state=seed, n_jobs=1, criterion="squared_error")
    if model_name == "xgboost":
        return XGBRegressor(
            random_state=seed,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=1,
            verbosity=0,
        )
    if model_name == "lightgbm":
        return LGBMRegressor(
            random_state=seed,
            n_jobs=1,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
            subsample_freq=1,
        )
    raise KeyError(model_name)


def prefixed_search_space(config: dict[str, Any], model_name: str) -> dict[str, list[Any]]:
    return {f"model__{name}": values for name, values in config["search_spaces"][model_name].items()}


@dataclass
class TunedFit:
    estimator: Pipeline
    best_parameters: dict[str, Any]
    best_inner_mae: float
    model_seed: int
    inner_seed: int
    search_seed: int


def tune_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    config: dict[str, Any],
    context: tuple[Any, ...],
    tuning_records: list[dict[str, Any]],
) -> TunedFit:
    namespace = config["model_seed_namespace"]
    model_seed = derive_seed(namespace, *context, model_name, "model")
    inner_seed = derive_seed(namespace, *context, model_name, "inner")
    search_seed = derive_seed(namespace, *context, model_name, "search")
    estimator = model_pipeline(make_model(model_name, model_seed))
    inner_cv = KFold(
        n_splits=min(int(config["inner_cv"]["n_splits"]), len(X_train)),
        shuffle=True,
        random_state=inner_seed,
    )
    parameter_space = prefixed_search_space(config, model_name)
    combinations = math.prod(len(values) for values in parameter_space.values())
    n_iter = min(int(config["inner_cv"]["search_iterations"]), combinations)
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=parameter_space,
        n_iter=n_iter,
        scoring="neg_mean_absolute_error",
        cv=inner_cv,
        random_state=search_seed,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
        error_score="raise",
    )
    # The threading backend is explicit so the run does not depend on process-
    # semaphore availability and remains reproducible in restricted systems.
    with parallel_backend("threading"):
        search.fit(X_train, y_train)

    for candidate_index, parameters in enumerate(search.cv_results_["params"]):
        tuning_records.append(
            {
                "context": "|".join(str(part) for part in context),
                "model": model_name,
                "training_rows": len(X_train),
                "model_seed": model_seed,
                "inner_seed": inner_seed,
                "search_seed": search_seed,
                "candidate": candidate_index + 1,
                "parameters_json": json.dumps(json_safe(parameters), sort_keys=True),
                "mean_inner_mae": -float(search.cv_results_["mean_test_score"][candidate_index]),
                "std_inner_mae": float(search.cv_results_["std_test_score"][candidate_index]),
                "rank": int(search.cv_results_["rank_test_score"][candidate_index]),
            }
        )

    best_parameters = {
        name.removeprefix("model__"): json_safe(value) for name, value in search.best_params_.items()
    }
    return TunedFit(
        estimator=search.best_estimator_,
        best_parameters=best_parameters,
        best_inner_mae=-float(search.best_score_),
        model_seed=model_seed,
        inner_seed=inner_seed,
        search_seed=search_seed,
    )


def record_best_fit(
    records: list[dict[str, Any]],
    fit: TunedFit,
    context: tuple[Any, ...],
    model_name: str,
    features: list[str],
    training_rows: int,
) -> None:
    records.append(
        {
            "context": "|".join(str(part) for part in context),
            "model": model_name,
            "training_rows": training_rows,
            "features": ";".join(features),
            "best_parameters_json": json.dumps(fit.best_parameters, sort_keys=True),
            "best_inner_mae": fit.best_inner_mae,
            "model_seed": fit.model_seed,
            "inner_seed": fit.inner_seed,
            "search_seed": fit.search_seed,
        }
    )


def select_mi_features(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    config: dict[str, Any],
    context: tuple[Any, ...],
    selection_records: list[dict[str, Any]],
) -> list[str]:
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X_train)
    seed = derive_seed(config["model_seed_namespace"], *context, "mutual_information")
    scores = mutual_info_regression(X_imputed, y_train, random_state=seed)
    keep = max(1, int(math.ceil(len(X_train.columns) * float(config["mi_keep_fraction"]))))
    order = np.argsort(-scores, kind="mergesort")
    selected_positions = set(int(position) for position in order[:keep])
    selected = [name for position, name in enumerate(X_train.columns) if position in selected_positions]
    selected.sort(key=lambda name: int(np.where(X_train.columns == name)[0][0]))
    for position, feature in enumerate(X_train.columns):
        selection_records.append(
            {
                "context": "|".join(str(part) for part in context),
                "mi_seed": seed,
                "training_rows": len(X_train),
                "feature": feature,
                "mi_score": float(scores[position]),
                "selected": position in selected_positions,
                "k": keep,
            }
        )
    return selected


def quantile_thresholds(afp_training: pd.Series) -> tuple[float, float]:
    values = pd.to_numeric(afp_training, errors="coerce")
    median = float(values.median())
    clean = values.fillna(median).to_numpy(dtype=float)
    q1, q2 = np.quantile(clean, [1 / 3, 2 / 3], method="linear")
    return float(q1), float(q2)


def assign_quantile_categories(afp: pd.Series, thresholds: tuple[float, float]) -> np.ndarray:
    q1, q2 = thresholds
    values = pd.to_numeric(afp, errors="coerce").fillna(q1).to_numpy(dtype=float)
    return np.where(values <= q1, "Small", np.where(values <= q2, "Medium", "Large"))


def clip_predictions(values: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    return np.maximum(float(config["prediction_floor"]), np.asarray(values, dtype=float))


def prediction_rows(
    data: pd.DataFrame,
    split: OuterFold,
    predictions: np.ndarray,
    family: str,
    feature_set: str,
    method: str,
    id_column: str,
    target_column: str,
    categories: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    if categories is None:
        categories = np.repeat("All", len(split.test_index))
    rows: list[dict[str, Any]] = []
    for local_position, row_index in enumerate(split.test_index):
        observed = float(data.iloc[row_index][target_column])
        predicted = float(predictions[local_position])
        rows.append(
            {
                "family": family,
                "feature_set": feature_set,
                "method": method,
                "repetition": split.repetition,
                "outer_seed": split.outer_seed,
                "fold": split.fold,
                "row_index_zero_based": int(row_index),
                "project_id": data.iloc[row_index][id_column],
                "category": str(categories[local_position]),
                "observed_effort": observed,
                "predicted_effort": predicted,
                "absolute_error": abs(observed - predicted),
                "squared_error": (observed - predicted) ** 2,
                "relative_error": abs(observed - predicted) / observed,
            }
        )
    return rows


def fit_scale_scenario(
    data: pd.DataFrame,
    split: OuterFold,
    feature_set_name: str,
    features: list[str],
    scenario: str,
    config: dict[str, Any],
    target_column: str,
    tuning_records: list[dict[str, Any]],
    best_fit_records: list[dict[str, Any]],
    selection_records: list[dict[str, Any]],
    category_records: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, dict[str, TunedFit]]:
    X_train_all = feature_frame(data, features, split.train_index)
    X_test_all = feature_frame(data, features, split.test_index)
    y_train_all = data.iloc[split.train_index][target_column].to_numpy(dtype=float)
    model_context = (
        "scale_nested_et",
        feature_set_name,
        scenario,
        split.repetition,
        split.fold,
    )
    fitted: dict[str, TunedFit] = {}

    if scenario in {"Global", "Global + MI"}:
        selected = features
        if scenario == "Global + MI":
            selected = select_mi_features(
                X_train_all,
                y_train_all,
                config,
                (*model_context, "All"),
                selection_records,
            )
        fit = tune_model(
            "extra_trees",
            X_train_all[selected],
            y_train_all,
            config,
            (*model_context, "All"),
            tuning_records,
        )
        record_best_fit(
            best_fit_records,
            fit,
            (*model_context, "All"),
            "extra_trees",
            selected,
            len(X_train_all),
        )
        predictions = clip_predictions(fit.estimator.predict(X_test_all[selected]), config)
        fitted["All"] = fit
        return predictions, np.repeat("All", len(split.test_index)), fitted

    thresholds = quantile_thresholds(data.iloc[split.train_index]["AFP"])
    train_categories = assign_quantile_categories(data.iloc[split.train_index]["AFP"], thresholds)
    test_categories = assign_quantile_categories(data.iloc[split.test_index]["AFP"], thresholds)
    predictions = np.full(len(split.test_index), np.nan, dtype=float)
    for category in CATEGORIES:
        train_mask = train_categories == category
        test_mask = test_categories == category
        category_records.append(
            {
                "feature_set": feature_set_name,
                "strategy": "Training-fold AFP quantiles",
                "scenario": scenario,
                "repetition": split.repetition,
                "outer_seed": split.outer_seed,
                "fold": split.fold,
                "category": category,
                "training_count": int(train_mask.sum()),
                "test_count": int(test_mask.sum()),
                "q1": thresholds[0],
                "q2": thresholds[1],
                "minimum_required": int(config["minimum_specialist_training_size"]),
                "adequate": int(train_mask.sum()) >= int(config["minimum_specialist_training_size"]),
            }
        )
        if train_mask.sum() < int(config["minimum_specialist_training_size"]):
            raise ValueError(f"Quantile category {category} is too small in {model_context}")
        selected = features
        category_X_train = X_train_all.loc[train_mask]
        category_y_train = y_train_all[train_mask]
        if scenario == "Quantile + MI":
            selected = select_mi_features(
                category_X_train,
                category_y_train,
                config,
                (*model_context, category),
                selection_records,
            )
        fit = tune_model(
            "extra_trees",
            category_X_train[selected],
            category_y_train,
            config,
            (*model_context, category),
            tuning_records,
        )
        record_best_fit(
            best_fit_records,
            fit,
            (*model_context, category),
            "extra_trees",
            selected,
            int(train_mask.sum()),
        )
        if test_mask.any():
            predictions[test_mask] = clip_predictions(fit.estimator.predict(X_test_all.loc[test_mask, selected]), config)
        fitted[category] = fit
    if np.isnan(predictions).any():
        raise AssertionError(f"Unfilled specialist predictions in {model_context}")
    return predictions, test_categories, fitted


def fit_benchmark_model(
    data: pd.DataFrame,
    split: OuterFold,
    features: list[str],
    model_name: str,
    config: dict[str, Any],
    target_column: str,
    tuning_records: list[dict[str, Any]],
    best_fit_records: list[dict[str, Any]],
) -> tuple[np.ndarray, TunedFit]:
    X_train = feature_frame(data, features, split.train_index)
    X_test = feature_frame(data, features, split.test_index)
    y_train = data.iloc[split.train_index][target_column].to_numpy(dtype=float)
    context = ("benchmark", "strict6", model_name, split.repetition, split.fold, "All")
    fit = tune_model(model_name, X_train, y_train, config, context, tuning_records)
    record_best_fit(best_fit_records, fit, context, model_name, features, len(X_train))
    return clip_predictions(fit.estimator.predict(X_test), config), fit


def fit_stacking(
    data: pd.DataFrame,
    split: OuterFold,
    features: list[str],
    tuned_bases: dict[str, TunedFit],
    config: dict[str, Any],
    target_column: str,
    best_fit_records: list[dict[str, Any]],
) -> np.ndarray:
    X_train = feature_frame(data, features, split.train_index)
    X_test = feature_frame(data, features, split.test_index)
    y_train = data.iloc[split.train_index][target_column].to_numpy(dtype=float)
    context = ("benchmark", "strict6", "stacking", split.repetition, split.fold, "All")
    stack_seed = derive_seed(config["model_seed_namespace"], *context, "stack_cv")
    estimators = [
        ("rf", clone(tuned_bases["random_forest"].estimator)),
        ("et", clone(tuned_bases["extra_trees"].estimator)),
        ("xgb", clone(tuned_bases["xgboost"].estimator)),
    ]
    stack = StackingRegressor(
        estimators=estimators,
        final_estimator=LinearRegression(),
        cv=KFold(n_splits=5, shuffle=True, random_state=stack_seed),
        n_jobs=-1,
        passthrough=False,
    )
    with parallel_backend("threading"):
        stack.fit(X_train, y_train)
    best_fit_records.append(
        {
            "context": "|".join(str(part) for part in context),
            "model": "stacking",
            "training_rows": len(X_train),
            "features": ";".join(features),
            "best_parameters_json": json.dumps(
                {
                    "base_models": ["tuned_random_forest", "tuned_extra_trees", "tuned_xgboost"],
                    "meta_learner": "LinearRegression",
                    "stack_cv_splits": 5,
                    "stack_cv_seed": stack_seed,
                },
                sort_keys=True,
            ),
            "best_inner_mae": np.nan,
            "model_seed": np.nan,
            "inner_seed": stack_seed,
            "search_seed": np.nan,
        }
    )
    return clip_predictions(stack.predict(X_test), config)


def fit_dummy(data: pd.DataFrame, split: OuterFold, features: list[str], target_column: str, config: dict[str, Any]) -> np.ndarray:
    X_train = feature_frame(data, features, split.train_index)
    X_test = feature_frame(data, features, split.test_index)
    y_train = data.iloc[split.train_index][target_column].to_numpy(dtype=float)
    dummy = model_pipeline(DummyRegressor(strategy="median"))
    dummy.fit(X_train, y_train)
    return clip_predictions(dummy.predict(X_test), config)


def fit_fixed_pipeline(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    config: dict[str, Any],
    context: tuple[Any, ...],
    best_fit_records: list[dict[str, Any]],
) -> Pipeline:
    seed = derive_seed(config["model_seed_namespace"], *context, "fixed_extra_trees")
    parameters = dict(config["fixed_extra_trees"])
    model = ExtraTreesRegressor(**parameters, random_state=seed, n_jobs=-1)
    pipeline = model_pipeline(model)
    pipeline.fit(X_train, y_train)
    best_fit_records.append(
        {
            "context": "|".join(str(part) for part in context),
            "model": "fixed_extra_trees",
            "training_rows": len(X_train),
            "features": ";".join(X_train.columns),
            "best_parameters_json": json.dumps(parameters, sort_keys=True),
            "best_inner_mae": np.nan,
            "model_seed": seed,
            "inner_seed": np.nan,
            "search_seed": np.nan,
        }
    )
    return pipeline


def fit_fixed_scale_scenario(
    data: pd.DataFrame,
    split: OuterFold,
    feature_set_name: str,
    features: list[str],
    scenario: str,
    config: dict[str, Any],
    target_column: str,
    best_fit_records: list[dict[str, Any]],
    selection_records: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    X_train_all = feature_frame(data, features, split.train_index)
    X_test_all = feature_frame(data, features, split.test_index)
    y_train_all = data.iloc[split.train_index][target_column].to_numpy(dtype=float)
    context = (
        "scale_fixed_et_sensitivity",
        feature_set_name,
        scenario,
        split.repetition,
        split.fold,
    )

    if scenario in {"Global", "Global + MI"}:
        selected = features
        if scenario == "Global + MI":
            selected = select_mi_features(
                X_train_all,
                y_train_all,
                config,
                (*context, "All"),
                selection_records,
            )
        pipeline = fit_fixed_pipeline(
            X_train_all[selected],
            y_train_all,
            config,
            (*context, "All"),
            best_fit_records,
        )
        prediction = clip_predictions(pipeline.predict(X_test_all[selected]), config)
        return prediction, np.repeat("All", len(split.test_index))

    thresholds = quantile_thresholds(data.iloc[split.train_index]["AFP"])
    train_categories = assign_quantile_categories(data.iloc[split.train_index]["AFP"], thresholds)
    test_categories = assign_quantile_categories(data.iloc[split.test_index]["AFP"], thresholds)
    predictions = np.full(len(split.test_index), np.nan, dtype=float)
    for category in CATEGORIES:
        train_mask = train_categories == category
        test_mask = test_categories == category
        selected = features
        category_X_train = X_train_all.loc[train_mask]
        category_y_train = y_train_all[train_mask]
        if scenario == "Quantile + MI":
            selected = select_mi_features(
                category_X_train,
                category_y_train,
                config,
                (*context, category),
                selection_records,
            )
        pipeline = fit_fixed_pipeline(
            category_X_train[selected],
            category_y_train,
            config,
            (*context, category),
            best_fit_records,
        )
        if test_mask.any():
            predictions[test_mask] = clip_predictions(
                pipeline.predict(X_test_all.loc[test_mask, selected]),
                config,
            )
    if np.isnan(predictions).any():
        raise AssertionError(f"Unfilled fixed-model specialist predictions in {context}")
    return predictions, test_categories


def kmeans_category_diagnostics(
    data: pd.DataFrame,
    split: OuterFold,
    feature_set_name: str,
    features: list[str],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    minimum = int(config["minimum_specialist_training_size"])
    strategies = {"AFP K-Means": ["AFP"], "Multivariate K-Means": features}
    for strategy, strategy_features in strategies.items():
        X_train = feature_frame(data, strategy_features, split.train_index)
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        transformed = scaler.fit_transform(imputer.fit_transform(X_train))
        seed = derive_seed(
            config["model_seed_namespace"],
            "category_diagnostic",
            feature_set_name,
            strategy,
            split.repetition,
            split.fold,
        )
        kmeans = KMeans(n_clusters=3, n_init=50, random_state=seed)
        raw_labels = kmeans.fit_predict(transformed)
        counts = pd.Series(raw_labels).value_counts().reindex(range(3), fill_value=0)
        mean_afp = pd.DataFrame(
            {"cluster": raw_labels, "AFP": data.iloc[split.train_index]["AFP"].to_numpy(dtype=float)}
        ).groupby("cluster")["AFP"].mean()
        ordered_clusters = list(mean_afp.sort_values().index)
        ordered_names = {cluster: CATEGORIES[position] for position, cluster in enumerate(ordered_clusters)}
        for cluster in range(3):
            records.append(
                {
                    "feature_set": feature_set_name,
                    "strategy": strategy,
                    "scenario": "Adequacy diagnostic only",
                    "repetition": split.repetition,
                    "outer_seed": split.outer_seed,
                    "fold": split.fold,
                    "category": ordered_names[cluster],
                    "training_count": int(counts.loc[cluster]),
                    "test_count": np.nan,
                    "q1": np.nan,
                    "q2": np.nan,
                    "minimum_required": minimum,
                    "adequate": bool((counts >= minimum).all()),
                    "kmeans_seed": seed,
                }
            )
    return records


def save_checkpoint(
    output_dir: Path,
    predictions: list[dict[str, Any]],
    tuning: list[dict[str, Any]],
    best_fits: list[dict[str, Any]],
    selections: list[dict[str, Any]],
    categories: list[dict[str, Any]],
) -> None:
    pd.DataFrame.from_records(predictions).to_csv(output_dir / "checkpoint_predictions.csv.gz", index=False)
    pd.DataFrame.from_records(tuning).to_csv(output_dir / "checkpoint_tuning_candidates.csv.gz", index=False)
    pd.DataFrame.from_records(best_fits).to_csv(output_dir / "checkpoint_selected_parameters.csv", index=False)
    pd.DataFrame.from_records(selections).to_csv(output_dir / "checkpoint_mi_records.csv.gz", index=False)
    pd.DataFrame.from_records(categories).to_csv(output_dir / "checkpoint_category_diagnostics.csv", index=False)


def metric_values(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    relative = np.abs(observed - predicted) / observed
    return {
        "MAE": float(mean_absolute_error(observed, predicted)),
        "RMSE": float(math.sqrt(mean_squared_error(observed, predicted))),
        "Pred25": float(100.0 * np.mean(relative <= 0.25)),
        "R2": float(r2_score(observed, predicted)),
    }


def bootstrap_metric_distributions(
    group: pd.DataFrame,
    project_ids: np.ndarray,
    repetitions: list[int],
    bootstrap_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    observed_by_project = (
        group.drop_duplicates("project_id").set_index("project_id").reindex(project_ids)["observed_effort"].to_numpy(dtype=float)
    )
    prediction_matrix = (
        group.pivot(index="project_id", columns="repetition", values="predicted_effort")
        .reindex(index=project_ids, columns=repetitions)
        .to_numpy(dtype=float)
    )
    if np.isnan(prediction_matrix).any():
        raise AssertionError("Incomplete project-by-repetition prediction matrix")
    observed_matrix = np.repeat(observed_by_project[:, None], len(repetitions), axis=1)
    absolute = np.abs(observed_matrix - prediction_matrix)
    squared = (observed_matrix - prediction_matrix) ** 2
    within = absolute / observed_matrix <= 0.25

    distributions = {name: np.empty(len(bootstrap_indices), dtype=float) for name in METRIC_NAMES}
    chunk_size = 200
    for start in range(0, len(bootstrap_indices), chunk_size):
        stop = min(start + chunk_size, len(bootstrap_indices))
        sampled = bootstrap_indices[start:stop]
        sampled_abs = absolute[sampled]
        sampled_sq = squared[sampled]
        sampled_within = within[sampled]
        distributions["MAE"][start:stop] = sampled_abs.mean(axis=(1, 2))
        distributions["RMSE"][start:stop] = np.sqrt(sampled_sq.mean(axis=(1, 2)))
        distributions["Pred25"][start:stop] = 100.0 * sampled_within.mean(axis=(1, 2))
        sampled_observed = observed_by_project[sampled]
        sampled_prediction = prediction_matrix[sampled]
        grand_mean = sampled_observed.mean(axis=1)
        numerator = ((sampled_observed[:, :, None] - sampled_prediction) ** 2).sum(axis=(1, 2))
        denominator = (
            ((sampled_observed - grand_mean[:, None]) ** 2).sum(axis=1) * len(repetitions)
        )
        distributions["R2"][start:stop] = 1.0 - numerator / denominator
    return distributions


def aggregate_metrics(
    predictions: pd.DataFrame,
    data: pd.DataFrame,
    id_column: str,
    bootstrap_indices: np.ndarray,
    output_dir: Path,
) -> pd.DataFrame:
    project_ids = data[id_column].to_numpy()
    repetitions = sorted(int(value) for value in predictions["repetition"].unique())
    records: list[dict[str, Any]] = []
    for keys, group in predictions.groupby(["family", "feature_set", "method"], sort=True):
        family, feature_set, method = keys
        expected = len(project_ids) * len(repetitions)
        if len(group) != expected:
            raise AssertionError(f"{keys} has {len(group)} predictions; expected {expected}")
        point = metric_values(group["observed_effort"].to_numpy(), group["predicted_effort"].to_numpy())
        distributions = bootstrap_metric_distributions(group, project_ids, repetitions, bootstrap_indices)
        for metric in METRIC_NAMES:
            low, high = np.quantile(distributions[metric], [0.025, 0.975], method="linear")
            records.append(
                {
                    "family": family,
                    "feature_set": feature_set,
                    "method": method,
                    "metric": metric,
                    "estimate_full_precision": point[metric],
                    "ci_2_5": float(low),
                    "ci_97_5": float(high),
                    "bootstrap_replicates": len(bootstrap_indices),
                }
            )
    result = pd.DataFrame.from_records(records)
    result.to_csv(output_dir / "metrics_full_precision.csv", index=False, float_format="%.15g")
    return result


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(p_values), dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    m = len(values)
    for rank_position, original_position in enumerate(order):
        candidate = (m - rank_position) * values[original_position]
        running = max(running, candidate)
        adjusted[original_position] = min(1.0, running)
    return adjusted


def signed_rank_effect(delta: np.ndarray) -> float:
    nonzero = delta[np.abs(delta) > 0]
    if len(nonzero) == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero), method="average")
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    return (positive - negative) / (positive + negative)


def paired_comparison(
    reference: pd.DataFrame,
    comparator: pd.DataFrame,
    project_ids: np.ndarray,
    bootstrap_indices: np.ndarray,
    fold_assignments: pd.DataFrame,
) -> dict[str, float]:
    key_columns = ["project_id", "repetition", "fold"]
    merged = reference[key_columns + ["absolute_error"]].merge(
        comparator[key_columns + ["absolute_error"]],
        on=key_columns,
        suffixes=("_reference", "_comparator"),
        validate="one_to_one",
    )
    project_delta = (
        merged.assign(delta=merged["absolute_error_reference"] - merged["absolute_error_comparator"])
        .groupby("project_id")["delta"]
        .mean()
        .reindex(project_ids)
        .to_numpy(dtype=float)
    )
    if np.allclose(project_delta, 0.0):
        wilcoxon_statistic, wilcoxon_p = 0.0, 1.0
    else:
        signed = wilcoxon(project_delta, alternative="two-sided", zero_method="wilcox", method="auto")
        wilcoxon_statistic, wilcoxon_p = float(signed.statistic), float(signed.pvalue)
    bootstrap_means = project_delta[bootstrap_indices].mean(axis=1)
    bootstrap_low, bootstrap_high = np.quantile(bootstrap_means, [0.025, 0.975], method="linear")

    fold_delta = (
        merged.assign(delta=merged["absolute_error_reference"] - merged["absolute_error_comparator"])
        .groupby(["repetition", "fold"])["delta"]
        .mean()
        .to_numpy(dtype=float)
    )
    fold_variance = float(np.var(fold_delta, ddof=1))
    test_counts = fold_assignments.groupby(["repetition", "fold"]).size().to_numpy(dtype=float)
    train_counts = len(project_ids) - test_counts
    test_train_ratio = float(np.mean(test_counts / train_counts))
    corrected_se = math.sqrt((1.0 / len(fold_delta) + test_train_ratio) * fold_variance)
    corrected_t = float(np.mean(fold_delta) / corrected_se) if corrected_se > 0 else 0.0
    corrected_p = float(2.0 * student_t.sf(abs(corrected_t), df=len(fold_delta) - 1))
    return {
        "mean_mae_difference_full_precision": float(np.mean(project_delta)),
        "mean_mae_difference_ci_2_5": float(bootstrap_low),
        "mean_mae_difference_ci_97_5": float(bootstrap_high),
        "median_project_difference": float(np.median(project_delta)),
        "wilcoxon_W": wilcoxon_statistic,
        "wilcoxon_p_raw": wilcoxon_p,
        "rank_biserial": signed_rank_effect(project_delta),
        "corrected_repeated_cv_t": corrected_t,
        "corrected_repeated_cv_df": len(fold_delta) - 1,
        "corrected_repeated_cv_p_raw": corrected_p,
        "corrected_standard_error": corrected_se,
        "mean_test_train_ratio": test_train_ratio,
        "positive_difference_means_comparator_has_lower_mae": True,
    }


def comparison_family(
    predictions: pd.DataFrame,
    family: str,
    feature_set: str,
    reference_method: str,
    comparator_methods: list[str],
    data: pd.DataFrame,
    id_column: str,
    bootstrap_indices: np.ndarray,
    fold_assignments: pd.DataFrame,
) -> pd.DataFrame:
    project_ids = data[id_column].to_numpy()
    reference = predictions[
        (predictions["family"] == family)
        & (predictions["feature_set"] == feature_set)
        & (predictions["method"] == reference_method)
    ]
    records: list[dict[str, Any]] = []
    for comparator_method in comparator_methods:
        comparator = predictions[
            (predictions["family"] == family)
            & (predictions["feature_set"] == feature_set)
            & (predictions["method"] == comparator_method)
        ]
        record = paired_comparison(
            reference,
            comparator,
            project_ids,
            bootstrap_indices,
            fold_assignments,
        )
        records.append(
            {
                "family": family,
                "feature_set": feature_set,
                "reference": reference_method,
                "comparator": comparator_method,
                **record,
            }
        )
    frame = pd.DataFrame.from_records(records)
    frame["wilcoxon_p_holm"] = holm_adjust(frame["wilcoxon_p_raw"])
    frame["corrected_repeated_cv_p_holm"] = holm_adjust(frame["corrected_repeated_cv_p_raw"])
    return frame


def provenance_comparisons(
    predictions: pd.DataFrame,
    data: pd.DataFrame,
    id_column: str,
    bootstrap_indices: np.ndarray,
    fold_assignments: pd.DataFrame,
) -> pd.DataFrame:
    project_ids = data[id_column].to_numpy()
    records: list[dict[str, Any]] = []
    for method in SCENARIOS:
        strict = predictions[
            (predictions["family"] == "scale_fixed_et_sensitivity")
            & (predictions["feature_set"] == "strict6")
            & (predictions["method"] == method)
        ]
        uncertain = predictions[
            (predictions["family"] == "scale_fixed_et_sensitivity")
            & (predictions["feature_set"] == "uncertain9")
            & (predictions["method"] == method)
        ]
        if strict.empty or uncertain.empty:
            continue
        result = paired_comparison(strict, uncertain, project_ids, bootstrap_indices, fold_assignments)
        records.append(
            {
                "family": "temporal_provenance_sensitivity",
                "reference": f"strict6 {method}",
                "comparator": f"uncertain9 {method}",
                **result,
            }
        )
    frame = pd.DataFrame.from_records(records)
    if not frame.empty:
        frame["wilcoxon_p_holm"] = holm_adjust(frame["wilcoxon_p_raw"])
        frame["corrected_repeated_cv_p_holm"] = holm_adjust(frame["corrected_repeated_cv_p_raw"])
    return frame


def verify_identical_global_predictions(predictions: pd.DataFrame) -> dict[str, Any]:
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
    identical = len(central) == len(benchmark) and np.array_equal(
        central["predicted_effort"].to_numpy(), benchmark["predicted_effort"].to_numpy()
    )
    if not identical:
        raise AssertionError("Central Global and benchmark Tuned Extra Trees predictions differ")
    return {
        "identical_prediction_vectors": True,
        "rows_compared": len(central),
        "comparison": "scale_nested_et/strict6/Global versus benchmark/strict6/Tuned Extra Trees",
    }


def category_summary(category_frame: pd.DataFrame) -> pd.DataFrame:
    per_fold = (
        category_frame.groupby(["feature_set", "strategy", "scenario", "repetition", "fold"], dropna=False)
        .agg(minimum_training_count=("training_count", "min"), all_categories_adequate=("adequate", "all"))
        .reset_index()
    )
    return (
        per_fold.groupby(["feature_set", "strategy", "scenario"], dropna=False)
        .agg(
            folds=("minimum_training_count", "size"),
            adequate_folds=("all_categories_adequate", "sum"),
            minimum_observed=("minimum_training_count", "min"),
            median_minimum=("minimum_training_count", "median"),
            maximum_minimum=("minimum_training_count", "max"),
        )
        .reset_index()
    )


def artifact_hashes(output_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "run_manifest.json" and not path.name.startswith("checkpoint_"):
            hashes[str(path.relative_to(output_dir))] = sha256_file(path)
    return hashes


def run(config_path: Path, output_dir: Path, smoke: bool) -> None:
    start = time.time()
    config = load_config(config_path, smoke)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "run_complete.marker").exists():
        raise FileExistsError(f"Completed output already exists: {output_dir}")
    write_json(output_dir / "config_snapshot.json", config)

    manifest: dict[str, Any] = {
        "protocol_id": config["protocol_id"],
        "status": "running",
        "started_utc": utc_now(),
        "command": " ".join(sys.argv),
        "config_source": str(config_path.resolve()),
        "config_snapshot_sha256": sha256_file(output_dir / "config_snapshot.json"),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "lightgbm": lightgbm.__version__,
        },
    }
    write_json(output_dir / "run_manifest.json", manifest)

    try:
        data, dataset_path = load_and_validate_data(config, config_path)
        id_column = config["dataset"]["id_column"]
        target_column = config["dataset"]["target_column"]
        manifest["dataset"] = {
            "path": str(dataset_path),
            "sha256": sha256_file(dataset_path),
            "rows": len(data),
            "columns": len(data.columns),
        }

        folds = make_outer_folds(data, config)
        fold_assignments = save_outer_folds(folds, data, id_column, output_dir)
        bootstrap_indices = make_bootstrap_indices(data, config, output_dir)

        predictions: list[dict[str, Any]] = []
        tuning_records: list[dict[str, Any]] = []
        best_fit_records: list[dict[str, Any]] = []
        selection_records: list[dict[str, Any]] = []
        category_records: list[dict[str, Any]] = []

        analysis_feature_sets = [name for name in ("strict6",) if name in config["feature_sets"]]
        completed_fold_counter = 0
        for feature_set_name in analysis_feature_sets:
            features = list(config["feature_sets"][feature_set_name])
            print(f"[{utc_now()}] Starting nested scale analysis for {feature_set_name}", flush=True)
            for split in folds:
                strict_global_fit: TunedFit | None = None
                strict_global_prediction: np.ndarray | None = None
                for scenario in SCENARIOS:
                    print(
                        f"[{utc_now()}] {feature_set_name} repetition={split.repetition} fold={split.fold} scenario={scenario}",
                        flush=True,
                    )
                    scenario_prediction, test_categories, fitted = fit_scale_scenario(
                        data,
                        split,
                        feature_set_name,
                        features,
                        scenario,
                        config,
                        target_column,
                        tuning_records,
                        best_fit_records,
                        selection_records,
                        category_records,
                    )
                    predictions.extend(
                        prediction_rows(
                            data,
                            split,
                            scenario_prediction,
                            "scale_nested_et",
                            feature_set_name,
                            scenario,
                            id_column,
                            target_column,
                            test_categories,
                        )
                    )
                    if feature_set_name == "strict6" and scenario == "Global":
                        strict_global_fit = fitted["All"]
                        strict_global_prediction = scenario_prediction.copy()

                category_records.extend(
                    kmeans_category_diagnostics(data, split, feature_set_name, features, config)
                )

                if feature_set_name == "strict6":
                    if strict_global_fit is None or strict_global_prediction is None:
                        raise AssertionError("Strict global Extra Trees fit was not retained")
                    predictions.extend(
                        prediction_rows(
                            data,
                            split,
                            strict_global_prediction,
                            "benchmark",
                            "strict6",
                            "Tuned Extra Trees",
                            id_column,
                            target_column,
                        )
                    )
                    tuned_bases: dict[str, TunedFit] = {"extra_trees": strict_global_fit}
                    benchmark_labels = {
                        "random_forest": "Random Forest",
                        "xgboost": "XGBoost",
                        "lightgbm": "LightGBM",
                    }
                    for model_name, display_name in benchmark_labels.items():
                        print(
                            f"[{utc_now()}] benchmark repetition={split.repetition} fold={split.fold} model={display_name}",
                            flush=True,
                        )
                        benchmark_prediction, benchmark_fit = fit_benchmark_model(
                            data,
                            split,
                            features,
                            model_name,
                            config,
                            target_column,
                            tuning_records,
                            best_fit_records,
                        )
                        tuned_bases[model_name] = benchmark_fit
                        predictions.extend(
                            prediction_rows(
                                data,
                                split,
                                benchmark_prediction,
                                "benchmark",
                                "strict6",
                                display_name,
                                id_column,
                                target_column,
                            )
                        )

                    print(
                        f"[{utc_now()}] benchmark repetition={split.repetition} fold={split.fold} model=Stacking",
                        flush=True,
                    )
                    stacking_prediction = fit_stacking(
                        data,
                        split,
                        features,
                        tuned_bases,
                        config,
                        target_column,
                        best_fit_records,
                    )
                    predictions.extend(
                        prediction_rows(
                            data,
                            split,
                            stacking_prediction,
                            "benchmark",
                            "strict6",
                            "Stacking (RF + ET + XGB)",
                            id_column,
                            target_column,
                        )
                    )
                    dummy_prediction = fit_dummy(data, split, features, target_column, config)
                    predictions.extend(
                        prediction_rows(
                            data,
                            split,
                            dummy_prediction,
                            "benchmark",
                            "strict6",
                            "Training-median dummy",
                            id_column,
                            target_column,
                        )
                    )

                completed_fold_counter += 1
                if completed_fold_counter % 5 == 0:
                    save_checkpoint(
                        output_dir,
                        predictions,
                        tuning_records,
                        best_fit_records,
                        selection_records,
                        category_records,
                    )
                    print(f"[{utc_now()}] Checkpoint after {completed_fold_counter} feature-set folds", flush=True)

        print(f"[{utc_now()}] Starting matched fixed-model temporal-provenance sensitivity", flush=True)
        fixed_sensitivity_feature_sets = [
            name for name in ("strict6", "uncertain9") if name in config["feature_sets"]
        ]
        for feature_set_name in fixed_sensitivity_feature_sets:
            features = list(config["feature_sets"][feature_set_name])
            for split in folds:
                for scenario in SCENARIOS:
                    fixed_prediction, fixed_categories = fit_fixed_scale_scenario(
                        data,
                        split,
                        feature_set_name,
                        features,
                        scenario,
                        config,
                        target_column,
                        best_fit_records,
                        selection_records,
                    )
                    predictions.extend(
                        prediction_rows(
                            data,
                            split,
                            fixed_prediction,
                            "scale_fixed_et_sensitivity",
                            feature_set_name,
                            scenario,
                            id_column,
                            target_column,
                            fixed_categories,
                        )
                    )

        if "leaky14" in config["feature_sets"]:
            print(f"[{utc_now()}] Starting deliberate target-leakage diagnostic", flush=True)
            leaky_features = list(config["feature_sets"]["leaky14"])
            for split in folds:
                fixed_prediction, fixed_categories = fit_fixed_scale_scenario(
                    data,
                    split,
                    "leaky14",
                    leaky_features,
                    "Global",
                    config,
                    target_column,
                    best_fit_records,
                    selection_records,
                )
                predictions.extend(
                    prediction_rows(
                        data,
                        split,
                        fixed_prediction,
                        "scale_fixed_et_sensitivity",
                        "leaky14",
                        "Global",
                        id_column,
                        target_column,
                        fixed_categories,
                    )
                )

        prediction_frame = pd.DataFrame.from_records(predictions)
        if not np.isfinite(prediction_frame["predicted_effort"]).all():
            raise AssertionError("Non-finite predictions detected")
        prediction_frame.to_csv(output_dir / "oof_predictions_full_precision.csv.gz", index=False, float_format="%.15g")
        pd.DataFrame.from_records(tuning_records).to_csv(
            output_dir / "nested_tuning_candidates.csv.gz", index=False, float_format="%.15g"
        )
        pd.DataFrame.from_records(best_fit_records).to_csv(
            output_dir / "selected_parameters_by_outer_fold.csv", index=False, float_format="%.15g"
        )
        pd.DataFrame.from_records(selection_records).to_csv(
            output_dir / "mutual_information_by_outer_training_set.csv.gz", index=False, float_format="%.15g"
        )
        category_frame = pd.DataFrame.from_records(category_records)
        category_frame.to_csv(output_dir / "category_diagnostics_by_outer_fold.csv", index=False, float_format="%.15g")
        category_summary_frame = category_summary(category_frame)
        category_summary_frame.to_csv(output_dir / "category_diagnostics_summary.csv", index=False, float_format="%.15g")

        qa = verify_identical_global_predictions(prediction_frame)
        metric_frame = aggregate_metrics(
            prediction_frame,
            data,
            id_column,
            bootstrap_indices,
            output_dir,
        )
        scale_comparisons: list[pd.DataFrame] = []
        for feature_set_name in analysis_feature_sets:
            scale_comparisons.append(
                comparison_family(
                    prediction_frame,
                    "scale_nested_et",
                    feature_set_name,
                    "Global",
                    ["Global + MI", "Quantile", "Quantile + MI"],
                    data,
                    id_column,
                    bootstrap_indices,
                    fold_assignments,
                )
            )
        scale_comparison_frame = pd.concat(scale_comparisons, ignore_index=True)
        scale_comparison_frame.to_csv(
            output_dir / "scale_pairwise_inference.csv", index=False, float_format="%.15g"
        )
        fixed_scale_comparisons: list[pd.DataFrame] = []
        for feature_set_name in fixed_sensitivity_feature_sets:
            fixed_scale_comparisons.append(
                comparison_family(
                    prediction_frame,
                    "scale_fixed_et_sensitivity",
                    feature_set_name,
                    "Global",
                    ["Global + MI", "Quantile", "Quantile + MI"],
                    data,
                    id_column,
                    bootstrap_indices,
                    fold_assignments,
                )
            )
        pd.concat(fixed_scale_comparisons, ignore_index=True).to_csv(
            output_dir / "fixed_scale_pairwise_inference.csv", index=False, float_format="%.15g"
        )
        benchmark_comparison_frame = comparison_family(
            prediction_frame,
            "benchmark",
            "strict6",
            "Tuned Extra Trees",
            ["Random Forest", "XGBoost", "LightGBM", "Stacking (RF + ET + XGB)", "Training-median dummy"],
            data,
            id_column,
            bootstrap_indices,
            fold_assignments,
        )
        benchmark_comparison_frame.to_csv(
            output_dir / "benchmark_pairwise_inference.csv", index=False, float_format="%.15g"
        )
        provenance_frame = provenance_comparisons(
            prediction_frame,
            data,
            id_column,
            bootstrap_indices,
            fold_assignments,
        )
        provenance_frame.to_csv(
            output_dir / "temporal_provenance_pairwise_inference.csv", index=False, float_format="%.15g"
        )

        metric_lookup = metric_frame.set_index(["family", "feature_set", "method", "metric"])
        central_ci = metric_lookup.loc[("scale_nested_et", "strict6", "Global", "MAE")]
        benchmark_ci = metric_lookup.loc[("benchmark", "strict6", "Tuned Extra Trees", "MAE")]
        ci_identical = bool(np.array_equal(central_ci.to_numpy(), benchmark_ci.to_numpy()))
        if not ci_identical:
            raise AssertionError("Identical prediction vectors produced non-identical MAE summary rows")
        qa["identical_mae_estimate_and_ci"] = ci_identical
        qa["shared_bootstrap_file"] = "bootstrap_project_indices.npy"
        write_json(output_dir / "quality_assurance.json", qa)

        manifest.update(
            {
                "status": "complete",
                "completed_utc": utc_now(),
                "elapsed_seconds": time.time() - start,
                "outer_fold_count": len(folds),
                "prediction_rows": len(prediction_frame),
                "tuning_candidate_rows": len(tuning_records),
                "selected_fit_rows": len(best_fit_records),
                "mi_record_rows": len(selection_records),
                "quality_assurance": qa,
            }
        )
        (output_dir / "run_complete.marker").write_text("complete\n", encoding="utf-8")
        manifest["artifact_sha256"] = artifact_hashes(output_dir)
        write_json(output_dir / "run_manifest.json", manifest)
        print(f"[{utc_now()}] Run complete in {manifest['elapsed_seconds']:.1f} seconds", flush=True)
    except Exception as error:
        manifest.update(
            {
                "status": "failed",
                "failed_utc": utc_now(),
                "elapsed_seconds": time.time() - start,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        write_json(output_dir / "run_manifest.json", manifest)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="Run a short implementation check; never use its results in the paper")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.config.resolve(), arguments.output_dir.resolve(), arguments.smoke)
