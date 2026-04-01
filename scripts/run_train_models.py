"""
Execution script for model training.

Inputs:
    data/processed/X_train_processed.csv
    data/processed/y_train.csv

Outputs:
    artifacts/models/
"""

from __future__ import annotations

from pathlib import Path

from src.models.train_models import load_training_data, train_models, save_models


X_TRAIN_PATH = Path("data/processed/X_train_processed.csv")
Y_TRAIN_PATH = Path("data/processed/y_train.csv")

MODEL_OUTPUT_DIR = Path("artifacts/models")


def main() -> None:

    X_train, y_train = load_training_data(
        X_train_path=str(X_TRAIN_PATH),
        y_train_path=str(Y_TRAIN_PATH),
    )

    models = train_models(X_train, y_train)

    save_models(models, output_dir=str(MODEL_OUTPUT_DIR))

    print("Model training completed.")


if __name__ == "__main__":
    main()