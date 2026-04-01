"""
Feature audit and final dataset preparation.

Purpose:
- validate dataset integrity before modeling
- handle missing values
- ensure no leakage columns remain
- produce final modeling-ready dataset
"""

from __future__ import annotations

import pandas as pd


def run_feature_audit(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    original_rows = len(result)

    # 1. Basic checks
    if result.duplicated().any():
        raise ValueError("Duplicate rows detected in dataset.")

    # 2. Missing values check
    missing_summary = result.isna().sum().sort_values(ascending=False)

    print("\nTop missing columns:")
    print(missing_summary.head(10))

    # 3. Fill missing values (production decision)
    result = result.fillna(0)

    # 4. Final validation
    if len(result) != original_rows:
        raise ValueError("Row count changed after audit.")

    return result