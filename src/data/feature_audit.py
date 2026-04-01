"""
Feature audit and final dataset preparation.

Purpose
-------
- validate dataset integrity before modeling
- report duplicate rows
- report missing values
- preserve raw engineered missingness for pipeline imputation
- produce final modeling-ready dataset without global fillna

Important
---------
This module does NOT impute missing values.
Missing value handling is deferred to the preprocessing pipeline
(SimpleImputer fitted on training data only).
"""

from __future__ import annotations

import pandas as pd


def run_feature_audit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run basic dataset integrity checks before modeling.

    Parameters
    ----------
    df : pd.DataFrame
        Engineered feature dataset.

    Returns
    -------
    pd.DataFrame
        Unmodified audited dataframe.

    Raises
    ------
    ValueError
        If duplicate rows are detected or row count changes unexpectedly.
    """
    result = df.copy()
    original_row_count = len(result)

    # 1. Duplicate row check
    duplicate_count = int(result.duplicated().sum())
    if duplicate_count > 0:
        raise ValueError(f"Duplicate rows detected in dataset: {duplicate_count}")

    # 2. Missing value summary
    missing_summary = result.isna().sum().sort_values(ascending=False)
    missing_summary = missing_summary[missing_summary > 0]

    print("\n=== Feature Audit Summary ===")
    print(f"Row count    : {len(result)}")
    print(f"Column count : {result.shape[1]}")
    print(f"Duplicates   : {duplicate_count}")

    if missing_summary.empty:
        print("\nNo missing values detected.")
    else:
        print("\nColumns with missing values:")
        print(missing_summary)

    # 3. Final row integrity check
    if len(result) != original_row_count:
        raise ValueError(
            f"Row count changed during feature audit: "
            f"{original_row_count} -> {len(result)}"
        )

    return result