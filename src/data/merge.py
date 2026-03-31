"""
Data merging and schema standardization utilities.
"""

from __future__ import annotations

import pandas as pd


def drop_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove accidental unnamed columns, usually created by empty spreadsheet columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe without unnamed columns.
    """
    cleaned_df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", na=False)].copy()
    return cleaned_df


def standardize_season_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply basic schema standardization to one season dataframe.

    Current standardization rules:
    - drop unnamed columns

    Parameters
    ----------
    df : pd.DataFrame
        Raw season dataframe.

    Returns
    -------
    pd.DataFrame
        Standardized dataframe.
    """
    df = drop_unnamed_columns(df)
    return df


def merge_seasons(dataframes: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge multiple standardized season DataFrames into one dataset.

    Parameters
    ----------
    dataframes : list[pd.DataFrame]
        List of season dataframes.

    Returns
    -------
    pd.DataFrame
        Combined dataset.
    """
    if len(dataframes) == 0:
        raise ValueError("No dataframes provided for merging.")

    merged_df = pd.concat(dataframes, ignore_index=True)
    return merged_df