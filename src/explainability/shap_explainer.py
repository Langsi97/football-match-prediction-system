from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap


MODEL_PATH = Path("artifacts/models/random_forest_best.joblib")
FEATURE_CONTRACT_PATH = Path("artifacts/final_feature_contract.csv")


def load_model(model_path: Path = MODEL_PATH) -> Any:
    """
    Load the trained Random Forest model artifact.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Random Forest model artifact not found at: {model_path}"
        )
    return joblib.load(model_path)


def load_feature_contract(feature_contract_path: Path = FEATURE_CONTRACT_PATH) -> list[str]:
    """
    Load the approved feature contract used during model training.

    Expected format:
    - a CSV with one column containing feature names
    - if multiple columns exist, the first column is used
    """
    if not feature_contract_path.exists():
        raise FileNotFoundError(
            f"Feature contract not found at: {feature_contract_path}"
        )

    feature_contract_df = pd.read_csv(feature_contract_path)

    if feature_contract_df.empty:
        raise ValueError("Feature contract file is empty.")

    feature_names = feature_contract_df.iloc[:, 0].dropna().astype(str).tolist()

    if not feature_names:
        raise ValueError("No feature names found in feature contract.")

    return feature_names


def align_features_to_training_schema(
    feature_df: pd.DataFrame,
    model: Any,
    feature_contract_path: Path = FEATURE_CONTRACT_PATH,
) -> pd.DataFrame:
    """
    Align inference features to the exact training schema and column order.

    Priority:
    1. Use model.feature_names_in_ if available
    2. Fall back to saved feature contract CSV

    This prevents sklearn errors caused by feature order mismatch.
    """
    if feature_df.shape[0] != 1:
        raise ValueError("feature_df must contain exactly one row.")

    if hasattr(model, "feature_names_in_"):
        expected_features = list(model.feature_names_in_)
    else:
        expected_features = load_feature_contract(feature_contract_path)

    missing_features = [col for col in expected_features if col not in feature_df.columns]
    extra_features = [col for col in feature_df.columns if col not in expected_features]

    if missing_features:
        raise ValueError(
            f"Missing features required by the trained model: {missing_features}"
        )

    aligned_df = feature_df.reindex(columns=expected_features)

    if extra_features:
        # Extra columns are safely dropped by reindexing above.
        pass

    return aligned_df


def get_predicted_class_index(model: Any, feature_df: pd.DataFrame) -> int:
    """
    Get the predicted class index for a single-row feature dataframe.
    """
    probabilities = model.predict_proba(feature_df)[0]
    return int(np.argmax(probabilities))


def get_predicted_class_label(model: Any, predicted_class_index: int) -> str:
    """
    Return the predicted class label.
    """
    if hasattr(model, "classes_"):
        return str(model.classes_[predicted_class_index])
    return str(predicted_class_index)


def compute_tree_shap_values(model: Any, feature_df: pd.DataFrame) -> Any:
    """
    Compute SHAP values using TreeExplainer.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(feature_df)
    return shap_values


def extract_class_shap_values(
    shap_values: Any,
    predicted_class_index: int,
) -> np.ndarray:
    """
    Extract SHAP values corresponding to the predicted class.
    """
    if isinstance(shap_values, list):
        return np.asarray(shap_values[predicted_class_index][0])

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        return shap_values[0, :, predicted_class_index]

    if shap_values.ndim == 2:
        return shap_values[0]

    raise ValueError(
        f"Unsupported SHAP value format with shape: {getattr(shap_values, 'shape', 'unknown')}"
    )


def build_feature_contributions_df(
    feature_df: pd.DataFrame,
    shap_contributions: np.ndarray,
) -> pd.DataFrame:
    """
    Build a tidy dataframe of local feature contributions.
    """
    row_values = feature_df.iloc[0]

    contribution_df = pd.DataFrame(
        {
            "feature": feature_df.columns,
            "feature_value": row_values.values,
            "shap_contribution": shap_contributions,
        }
    )

    contribution_df["abs_contribution"] = contribution_df["shap_contribution"].abs()
    contribution_df["direction"] = np.where(
        contribution_df["shap_contribution"] >= 0,
        "supports_prediction",
        "opposes_prediction",
    )

    contribution_df = contribution_df.sort_values(
        by="abs_contribution",
        ascending=False,
    ).reset_index(drop=True)

    return contribution_df


def compute_shap_explanation(
    feature_df: pd.DataFrame,
    model_path: Path = MODEL_PATH,
    top_n: int = 5,
) -> dict[str, Any]:
    """
    Compute a local SHAP explanation for a single-row input.
    """
    if feature_df.empty:
        raise ValueError("feature_df is empty.")

    if feature_df.shape[0] != 1:
        raise ValueError("Only single-row SHAP explanations are supported.")

    model = load_model(model_path=model_path)
    aligned_feature_df = align_features_to_training_schema(feature_df, model=model)

    predicted_class_index = get_predicted_class_index(model, aligned_feature_df)
    predicted_class_label = get_predicted_class_label(model, predicted_class_index)

    shap_values = compute_tree_shap_values(model, aligned_feature_df)
    class_shap_values = extract_class_shap_values(
        shap_values=shap_values,
        predicted_class_index=predicted_class_index,
    )

    contribution_df = build_feature_contributions_df(
        feature_df=aligned_feature_df,
        shap_contributions=class_shap_values,
    )

    top_positive_df = (
        contribution_df[contribution_df["shap_contribution"] > 0]
        .sort_values(by="shap_contribution", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    top_negative_df = (
        contribution_df[contribution_df["shap_contribution"] < 0]
        .sort_values(by="shap_contribution", ascending=True)
        .head(top_n)
        .reset_index(drop=True)
    )

    return {
        "predicted_class_index": predicted_class_index,
        "predicted_class_label": predicted_class_label,
        "feature_contributions_df": contribution_df,
        "top_positive_df": top_positive_df,
        "top_negative_df": top_negative_df,
    }