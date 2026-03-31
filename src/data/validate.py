"""
Schema validation utilities for season datasets.
"""

from __future__ import annotations

from typing import Dict
import pandas as pd


def get_column_report(dataframes: Dict[str, pd.DataFrame]) -> dict:
    """
    Return common and union columns across multiple DataFrames.
    """
    if not dataframes:
        raise ValueError("No dataframes provided for validation.")

    per_file_columns = {
        name: set(df.columns.tolist()) for name, df in dataframes.items()
    }

    all_column_sets = list(per_file_columns.values())
    common_columns = set.intersection(*all_column_sets)
    union_columns = set.union(*all_column_sets)

    return {
        "common_columns": sorted(common_columns),
        "union_columns": sorted(union_columns),
        "per_file_columns": {
            name: sorted(cols) for name, cols in per_file_columns.items()
        },
    }


def find_schema_differences(dataframes: Dict[str, pd.DataFrame]) -> dict:
    """
    Return missing and extra columns per file relative to the union schema.
    """
    report = get_column_report(dataframes)
    union_columns = set(report["union_columns"])

    differences = {}
    for name, cols in report["per_file_columns"].items():
        current_cols = set(cols)
        differences[name] = {
            "missing_columns": sorted(union_columns - current_cols),
            "extra_columns": sorted(current_cols - union_columns),
        }

    return differences