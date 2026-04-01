"""
Preprocessing utilities for production ML workflows.

This module:
- separates target and metadata from modeling features
- keeps numeric features only for the current phase
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

    numeric_columns = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()

    if not numeric_columns:
        raise ValueError("No numeric columns available for preprocessing.")

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