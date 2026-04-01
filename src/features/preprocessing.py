"""
Preprocessing utilities for production ML workflows.

This module:
- separates metadata, target, and modeling features
- removes leakage columns
- keeps valid engineered rolling features
- keeps numeric features only
- imputes missing values
- scales numeric features
- returns transformed train/test matrices consistently
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_TARGET = "FTR"
DEFAULT_METADATA_COLUMNS = ["Date", "HomeTeam", "AwayTeam"]


def remove_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns that should never be used for training.

    Removed groups
    --------------
    1. Raw current-match post-match statistics
    2. Bookmaker odds
    3. Technical identifiers

    Important
    ---------
    Rolling features are intentionally kept because they are historical,
    shift-based aggregates derived from previous matches only.
    """
    leakage_columns = [
        # Raw post-match statistics from the current match
        "FTHG", "FTAG",
        "HS", "AS",
        "HST", "AST",
        "HF", "AF",
        "HC", "AC",
        "HY", "AY",
        "HR", "AR",

        # Bookmaker odds
        "B365H", "B365D", "B365A",
        "BWH", "BWD", "BWA",
        "IWH", "IWD", "IWA",
        "PSH", "PSD", "PSA",
        "WHH", "WHD", "WHA",
        "VCH", "VCD", "VCA",
        "MaxH", "MaxD", "MaxA",
        "AvgH", "AvgD", "AvgA",

        # Technical identifier
        "MatchID",
    ]

    result = df.copy()
    existing_leakage_columns = [col for col in leakage_columns if col in result.columns]

    if existing_leakage_columns:
        result = result.drop(columns=existing_leakage_columns)

    return result


def select_model_columns(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str = DEFAULT_TARGET,
    metadata_columns: List[str] | None = None,
) -> Dict[str, pd.DataFrame]:
    """
    Split train/test into metadata, target, and model features.
    """
    metadata_columns = metadata_columns or DEFAULT_METADATA_COLUMNS

    if target_column not in train_df.columns:
        raise ValueError(f"Missing target column in train data: {target_column}")
    if target_column not in test_df.columns:
        raise ValueError(f"Missing target column in test data: {target_column}")

    excluded_columns = set(metadata_columns + [target_column])

    X_train = train_df[[c for c in train_df.columns if c not in excluded_columns]].copy()
    X_test = test_df[[c for c in test_df.columns if c not in excluded_columns]].copy()

    # Remove only true leakage columns
    X_train = remove_leakage_columns(X_train)
    X_test = remove_leakage_columns(X_test)

    numeric_columns = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()

    if not numeric_columns:
        raise ValueError("No numeric columns available for preprocessing after leakage removal.")

    X_train = X_train[numeric_columns].copy()
    X_test = X_test[numeric_columns].copy()

    y_train = train_df[target_column].copy()
    y_test = test_df[target_column].copy()

    train_meta = train_df[[c for c in metadata_columns if c in train_df.columns]].copy()
    test_meta = test_df[[c for c in metadata_columns if c in test_df.columns]].copy()

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "train_meta": train_meta,
        "test_meta": test_meta,
        "feature_names": pd.Index(numeric_columns),
    }


def build_numeric_preprocessor(numeric_columns: List[str]) -> ColumnTransformer:
    """
    Build numeric preprocessing pipeline.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
        ],
        remainder="drop",
    )

    return preprocessor


def fit_transform_preprocessor(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[ColumnTransformer, pd.DataFrame, pd.DataFrame]:
    """
    Fit preprocessor on train only, then transform train and test.
    """
    numeric_columns = X_train.columns.tolist()
    preprocessor = build_numeric_preprocessor(numeric_columns)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    X_train_processed = pd.DataFrame(
        X_train_processed,
        columns=numeric_columns,
        index=X_train.index,
    )

    X_test_processed = pd.DataFrame(
        X_test_processed,
        columns=numeric_columns,
        index=X_test.index,
    )

    return preprocessor, X_train_processed, X_test_processed