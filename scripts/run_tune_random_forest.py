"""
Execution script for Random Forest tuning with Optuna + MLflow.

Inputs:
    data/processed/X_train_resampled.csv
    data/processed/y_train_resampled.csv

Outputs:
    artifacts/reports/best_random_forest_params.json
    artifacts/reports/optuna_random_forest_trials.csv
    artifacts/label_encoder.joblib
"""

from __future__ import annotations

from pathlib import Path

import joblib

from src.models.tune_random_forest import load_training_data, tune_random_forest


X_TRAIN_PATH = Path("data/processed/X_train_resampled.csv")
Y_TRAIN_PATH = Path("data/processed/y_train_resampled.csv")
LABEL_ENCODER_PATH = Path("artifacts/label_encoder.joblib")


def main() -> None:
    if not X_TRAIN_PATH.exists():
        raise FileNotFoundError(f"Missing file: {X_TRAIN_PATH}")
    if not Y_TRAIN_PATH.exists():
        raise FileNotFoundError(f"Missing file: {Y_TRAIN_PATH}")

    X_train, y_train, label_encoder = load_training_data(
        x_path=str(X_TRAIN_PATH),
        y_path=str(Y_TRAIN_PATH),
    )

    LABEL_ENCODER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)

    study = tune_random_forest(
        X=X_train,
        y=y_train,
        n_trials=40,
        valid_size=0.2,
        experiment_name="football-match-prediction-rf-optuna",
    )

    print("Tuning completed.")
    print(f"Best recall_macro: {study.best_value:.6f}")
    print("Best params:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()