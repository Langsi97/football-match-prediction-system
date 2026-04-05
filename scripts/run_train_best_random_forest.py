"""
Run final Random Forest training using best Optuna params
"""

from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


X_PATH = Path("data/processed/X_train_resampled.csv")
Y_PATH = Path("data/processed/y_train_resampled.csv")
PARAMS_PATH = Path("artifacts/reports/best_random_forest_params.json")

MODEL_PATH = Path("artifacts/models/random_forest_best.joblib")
ENCODER_PATH = Path("artifacts/label_encoder.joblib")


def main():
    # Load data
    X = pd.read_csv(X_PATH)
    y_raw = pd.read_csv(Y_PATH)["FTR"]

    # Encode target
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)

    # Load best params
    with open(PARAMS_PATH, "r") as f:
        best_params = json.load(f)

    print("Best params:", best_params)

    # Train model
    model = RandomForestClassifier(
        **best_params,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)

    # Save artifacts
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    joblib.dump(encoder, ENCODER_PATH)

    print("✅ Model saved:", MODEL_PATH)
    print("✅ Encoder saved:", ENCODER_PATH)


if __name__ == "__main__":
    main()