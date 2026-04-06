from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap


MODEL_PATH = Path("artifacts/models/random_forest_best.joblib")
FEATURE_CONTRACT_PATH = Path("artifacts/final_feature_contract.csv")

# Business-facing class names
CLASS_NAME_MAP = {
    0: "Draw",
    1: "Home Win",
    2: "Away Win",
    "0": "Draw",
    "1": "Home Win",
    "2": "Away Win",
    "H": "Home Win",
    "D": "Draw",
    "A": "Away Win",
}

# External prediction output -> internal model label
# Your app prediction output uses H / D / A, while the RF model appears to use 0 / 1 / 2.
# Based on your own mapping:
# - 1 = Home Win
# - 0 = Draw
# - 2 = Away Win
EXTERNAL_TO_INTERNAL_LABEL_MAP = {
    "H": 1,
    "D": 0,
    "A": 2,
    1: 1,
    0: 0,
    2: 2,
    "1": 1,
    "0": 0,
    "2": 2,
}


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
    """
    if feature_df.shape[0] != 1:
        raise ValueError("feature_df must contain exactly one row.")

    if hasattr(model, "feature_names_in_"):
        expected_features = list(model.feature_names_in_)
    else:
        expected_features = load_feature_contract(feature_contract_path)

    missing_features = [col for col in expected_features if col not in feature_df.columns]

    if missing_features:
        raise ValueError(
            f"Missing features required by the trained model: {missing_features}"
        )

    aligned_df = feature_df.reindex(columns=expected_features)
    return aligned_df


def normalize_prediction_label(prediction_label: Any) -> str:
    """
    Convert a raw label into a human-readable class name.
    """
    prediction_str = str(prediction_label)
    return CLASS_NAME_MAP.get(prediction_label, CLASS_NAME_MAP.get(prediction_str, prediction_str))


def convert_external_prediction_to_internal_label(prediction_label: Any) -> Any:
    """
    Convert app-level prediction output into the model's internal class label.

    Example:
    - app output H -> model label 1
    - app output D -> model label 0
    - app output A -> model label 2
    """
    prediction_str = str(prediction_label)
    if prediction_label in EXTERNAL_TO_INTERNAL_LABEL_MAP:
        return EXTERNAL_TO_INTERNAL_LABEL_MAP[prediction_label]
    if prediction_str in EXTERNAL_TO_INTERNAL_LABEL_MAP:
        return EXTERNAL_TO_INTERNAL_LABEL_MAP[prediction_str]
    return prediction_label


def resolve_target_class_position(
    model: Any,
    target_prediction_label: Any | None,
    feature_df: pd.DataFrame,
) -> tuple[int, Any, str, np.ndarray, list[Any]]:
    """
    Resolve which class SHAP should explain.

    Priority:
    1. If target_prediction_label is provided from the app prediction result,
       explain that exact class.
    2. Otherwise, fall back to the model's own argmax prediction.

    Returns:
    - target_class_position
    - target_class_label
    - target_class_name
    - model_probabilities
    - class_labels
    """
    if not hasattr(model, "classes_"):
        raise ValueError("The model does not expose classes_.")

    class_labels = list(model.classes_)
    probabilities = model.predict_proba(feature_df)[0]

    if target_prediction_label is not None:
        internal_target_label = convert_external_prediction_to_internal_label(target_prediction_label)

        if internal_target_label not in class_labels:
            raise ValueError(
                f"Target prediction label '{target_prediction_label}' maps to internal label "
                f"'{internal_target_label}', but that label is not present in model.classes_={class_labels}"
            )

        target_class_position = class_labels.index(internal_target_label)
        target_class_label = internal_target_label
        target_class_name = normalize_prediction_label(target_prediction_label)

        return (
            target_class_position,
            target_class_label,
            target_class_name,
            probabilities,
            class_labels,
        )

    # Fallback: use model argmax if no explicit target class was provided
    target_class_position = int(np.argmax(probabilities))
    target_class_label = class_labels[target_class_position]
    target_class_name = normalize_prediction_label(target_class_label)

    return (
        target_class_position,
        target_class_label,
        target_class_name,
        probabilities,
        class_labels,
    )


def compute_tree_shap_values(model: Any, feature_df: pd.DataFrame) -> Any:
    """
    Compute SHAP values using TreeExplainer.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(feature_df)
    return shap_values


def extract_class_shap_values(
    shap_values: Any,
    target_class_position: int,
) -> np.ndarray:
    """
    Extract SHAP values corresponding to the requested class position.
    """
    if isinstance(shap_values, list):
        return np.asarray(shap_values[target_class_position][0])

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        return shap_values[0, :, target_class_position]

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
    target_prediction_label: Any | None = None,
) -> dict[str, Any]:
    """
    Compute a local SHAP explanation for a single-row input.

    Important:
    - If target_prediction_label is provided, SHAP explains that exact class.
    - This keeps SHAP aligned with the app's prediction output.
    """
    if feature_df.empty:
        raise ValueError("feature_df is empty.")

    if feature_df.shape[0] != 1:
        raise ValueError("Only single-row SHAP explanations are supported.")

    model = load_model(model_path=model_path)
    aligned_feature_df = align_features_to_training_schema(feature_df, model=model)

    (
        target_class_position,
        target_class_label,
        target_class_name,
        model_probabilities,
        class_labels,
    ) = resolve_target_class_position(
        model=model,
        target_prediction_label=target_prediction_label,
        feature_df=aligned_feature_df,
    )

    shap_values = compute_tree_shap_values(model, aligned_feature_df)
    class_shap_values = extract_class_shap_values(
        shap_values=shap_values,
        target_class_position=target_class_position,
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
        "target_class_position": target_class_position,
        "target_class_label": target_class_label,
        "target_class_name": target_class_name,
        "model_class_labels": class_labels,
        "model_probabilities": model_probabilities,
        "feature_contributions_df": contribution_df,
        "top_positive_df": top_positive_df,
        "top_negative_df": top_negative_df,
    }