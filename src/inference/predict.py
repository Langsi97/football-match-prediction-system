from pathlib import Path

import joblib
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "artifacts" / "models" / "random_forest_best.joblib"


def _load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def predict_from_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    model = _load_model()

    prediction = model.predict(feature_df)[0]
    probabilities = model.predict_proba(feature_df)[0]
    classes = list(model.classes_)

    class_prob_map = {cls: float(prob) for cls, prob in zip(classes, probabilities)}

    result_df = pd.DataFrame(
        {
            "prediction": [prediction],
            "prob_H": [class_prob_map.get(1, class_prob_map.get("H", 0.0))],
            "prob_D": [class_prob_map.get(0, class_prob_map.get("D", 0.0))],
            "prob_A": [class_prob_map.get(2, class_prob_map.get("A", 0.0))],
        }
    )

    return result_df