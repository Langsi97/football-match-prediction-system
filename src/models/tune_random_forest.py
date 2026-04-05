"""
Optuna + MLflow tuning for Random Forest.

This module:
- loads resampled training data
- encodes the target
- creates an internal validation split from training data only
- tunes Random Forest with Optuna
- logs each trial to MLflow as a nested run
- optimizes for recall_macro
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_training_data(
    x_path: str,
    y_path: str,
) -> Tuple[pd.DataFrame, pd.Series, LabelEncoder]:
    """
    Load resampled training data and encode target labels.
    """
    X = pd.read_csv(x_path)
    y_raw = pd.read_csv(y_path)["FTR"]

    label_encoder = LabelEncoder()
    y = pd.Series(label_encoder.fit_transform(y_raw), name="FTR")

    return X, y, label_encoder


def compute_multiclass_brier_score(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray,
) -> float:
    """
    Compute multiclass Brier score.
    Lower is better.
    """
    y_true_encoded = np.zeros_like(y_proba)

    for i, cls in enumerate(classes):
        y_true_encoded[:, i] = (y_true == cls).astype(int)

    return float(np.mean(np.sum((y_proba - y_true_encoded) ** 2, axis=1)))


def build_random_forest_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Suggest Random Forest hyperparameters for Optuna.
    """
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
        "max_depth": trial.suggest_int("max_depth", 4, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
        "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample", None]),
        "random_state": 42,
        "n_jobs": -1,
    }


def objective_factory(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
):
    """
    Create an Optuna objective function bound to a fixed train/validation split.
    """
    def objective(trial: optuna.Trial) -> float:
        params = build_random_forest_params(trial)

        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            mlflow.log_params(params)

            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_valid)
            y_proba = model.predict_proba(X_valid)

            metrics = {
                "accuracy": accuracy_score(y_valid, y_pred),
                "precision_weighted": precision_score(
                    y_valid, y_pred, average="weighted", zero_division=0
                ),
                "recall_weighted": recall_score(
                    y_valid, y_pred, average="weighted", zero_division=0
                ),
                "recall_macro": recall_score(
                    y_valid, y_pred, average="macro", zero_division=0
                ),
                "f1_weighted": f1_score(
                    y_valid, y_pred, average="weighted", zero_division=0
                ),
                "brier_score": compute_multiclass_brier_score(
                    y_true=y_valid.to_numpy(),
                    y_proba=y_proba,
                    classes=model.classes_,
                ),
            }

            mlflow.log_metrics(metrics)

            return metrics["recall_macro"]

    return objective


def tune_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 40,
    valid_size: float = 0.2,
    experiment_name: str = "football-match-prediction-rf-optuna",
):
    """
    Run Optuna tuning with MLflow tracking.
    """
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=valid_size,
        random_state=42,
        stratify=y,
    )

    mlflow.set_experiment(experiment_name)

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="random_forest_recall_macro_optimization",
    )

    objective = objective_factory(
        X_train=X_train,
        X_valid=X_valid,
        y_train=y_train,
        y_valid=y_valid,
    )

    with mlflow.start_run(run_name="optuna_random_forest_parent"):
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("optimization_metric", "recall_macro")
        mlflow.log_param("validation_strategy", "train_validation_split")
        mlflow.log_param("validation_size", valid_size)

        study.optimize(objective, n_trials=n_trials)

        mlflow.log_metric("best_value_recall_macro", study.best_value)

        reports_dir = Path("artifacts/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)

        best_params_path = reports_dir / "best_random_forest_params.json"
        with best_params_path.open("w", encoding="utf-8") as f:
            json.dump(study.best_params, f, indent=2)

        trials_df = study.trials_dataframe()
        trials_csv_path = reports_dir / "optuna_random_forest_trials.csv"
        trials_df.to_csv(trials_csv_path, index=False)

        mlflow.log_artifact(str(best_params_path))
        mlflow.log_artifact(str(trials_csv_path))

    return study