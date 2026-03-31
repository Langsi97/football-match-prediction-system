"""
Dataset enrichment utilities.

This module merges the preprocessed match dataset with the league-position
dataset while preserving the league-position row count.
"""

from __future__ import annotations

import pandas as pd


MERGE_KEYS = ["Date", "HomeTeam", "AwayTeam", "FTR"]


def prepare_merge_keys(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize merge key columns before joining datasets.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with standardized merge keys.
    """
    prepared_df = df.copy()

    prepared_df["Date"] = pd.to_datetime(prepared_df["Date"], errors="coerce")

    for col in ["HomeTeam", "AwayTeam", "FTR"]:
        prepared_df[col] = prepared_df[col].astype(str).str.strip()

    return prepared_df


def merge_with_league_positions(
    matches_df: pd.DataFrame,
    league_positions_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge preprocessed match data into the league-position dataset.

    Important:
    - league_positions_df is used as the LEFT table
    - this preserves the exact row count of league_positions_df

    Parameters
    ----------
    matches_df : pd.DataFrame
        Preprocessed match dataset.
    league_positions_df : pd.DataFrame
        League-position dataset.

    Returns
    -------
    pd.DataFrame
        Enriched dataset with the same number of rows as league_positions_df.
    """
    matches_df = prepare_merge_keys(matches_df)
    league_positions_df = prepare_merge_keys(league_positions_df)

    merged_df = pd.merge(
        league_positions_df,
        matches_df,
        on=MERGE_KEYS,
        how="left",
        suffixes=("_league", "_match"),
    )

    return merged_df


def validate_row_count(
    merged_df: pd.DataFrame,
    expected_rows: int,
) -> None:
    """
    Validate that the merged dataset kept the expected number of rows.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Merged dataframe.
    expected_rows : int
        Expected number of rows.

    Raises
    ------
    ValueError
        If row count does not match expectation.
    """
    actual_rows = merged_df.shape[0]

    if actual_rows != expected_rows:
        raise ValueError(
            f"Row count mismatch after merge. "
            f"Expected {expected_rows}, got {actual_rows}."
        )