"""
Execution script for model evaluation.

Inputs:
    data/processed/X_test_processed.csv
    data/processed/y_test.csv
    artifacts/models/
    artifacts/label_encoder.joblib

Output:
    artifacts/reports/model_comparison.csv
"""

from __future__ import annotations

from pathlib import Path

from src.evaluation.evaluate_models import (
    evaluate_models,
    load_models,
    load_test_data,
)


X_TEST_PATH = Path("data/processed/X_test_processed.csv")
Y_TEST_PATH = Path("data/processed/y_test.csv")
MODEL_DIR = Path("artifacts/models")
LABEL_ENCODER_PATH = Path("artifacts/label_encoder.joblib")
OUTPUT_PATH = Path("artifacts/reports/model_comparison.csv")


def main() -> None:
    X_test, y_test = load_test_data(
        X_test_path=str(X_TEST_PATH),
        y_test_path=str(Y_TEST_PATH),
        label_encoder_path=str(LABEL_ENCODER_PATH),
    )

    models = load_models(str(MODEL_DIR))
    results = evaluate_models(models=models, X_test=X_test, y_test=y_test)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    print("Model evaluation completed.")
    print(results)
    print(f"Saved report: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()