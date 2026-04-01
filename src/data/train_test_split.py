"""
Time-based train/test split utilities.

This module creates a deterministic chronological split using a cutoff date.
It is designed for production ML workflows where temporal leakage must be
strictly avoided.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, Any

import pandas as pd
import yaml


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def split_dataset_by_date(
    df: pd.DataFrame,
    split_date: str,
    date_column: str = "Date",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset chronologically using a cutoff date.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    split_date : str
        Cutoff date in YYYY-MM-DD format.
    date_column : str, default="Date"
        Name of the date column.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df)
    """
    if date_column not in df.columns:
        raise ValueError(f"Missing date column: {date_column}")

    result = df.copy()
    result[date_column] = pd.to_datetime(result[date_column], errors="coerce")

    if result[date_column].isna().any():
        raise ValueError(f"Invalid datetime values found in column '{date_column}'.")

    cutoff = pd.Timestamp(split_date)

    result = result.sort_values(date_column, ascending=True).reset_index(drop=True)

    train_df = result[result[date_column] < cutoff].reset_index(drop=True)
    test_df = result[result[date_column] >= cutoff].reset_index(drop=True)

    if train_df.empty:
        raise ValueError("Training dataset is empty after split.")
    if test_df.empty:
        raise ValueError("Test dataset is empty after split.")

    if train_df[date_column].max() >= cutoff:
        raise ValueError("Temporal leakage detected in training split.")
    if test_df[date_column].min() < cutoff:
        raise ValueError("Temporal leakage detected in test split.")

    return train_df, test_df