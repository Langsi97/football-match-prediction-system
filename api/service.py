from __future__ import annotations

from typing import Any

from src.inference.input_schema import build_feature_ready_row
from src.inference.predict import predict_from_features


def predict_match(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Run model inference from raw API payload.
    """
    feature_df = build_feature_ready_row(payload)
    prediction_df = predict_from_features(feature_df)

    if prediction_df.empty:
        raise ValueError("Prediction pipeline returned an empty result.")

    row = prediction_df.iloc[0]

    return {
        "prediction": str(row["prediction"]),
        "prob_H": float(row.get("prob_H", 0.0)),
        "prob_D": float(row.get("prob_D", 0.0)),
        "prob_A": float(row.get("prob_A", 0.0)),
    }