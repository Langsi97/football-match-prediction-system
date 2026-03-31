"""
Match-level preprocessing utilities.

This module contains the first clean preprocessing steps applied
to the merged raw match dataset before feature engineering.
"""

from __future__ import annotations

import pandas as pd


def trim_columns_up_to_avg_a(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only columns up to and including 'AvgA'.

    This matches your thesis logic, where columns after the bookmaker
    average away odds were not needed for the base dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Trimmed dataframe.
    """
    if "AvgA" not in df.columns:
        raise KeyError("'AvgA' column not found in dataframe.")

    end_index = df.columns.get_loc("AvgA")
    trimmed_df = df.iloc[:, : end_index + 1].copy()
    return trimmed_df


def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that are not needed in the pre-match base dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe after removing irrelevant columns.
    """
    columns_to_drop = ["HTHG", "HTAG", "HTR"]
    existing_columns = [col for col in columns_to_drop if col in df.columns]
    return df.drop(columns=existing_columns).copy()


def preprocess_date_and_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert match date to datetime and extract hour from time.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with cleaned Date and Hour columns.
    """
    processed_df = df.copy()

    processed_df["Date"] = pd.to_datetime(
        processed_df["Date"],
        dayfirst=True,
        errors="coerce",
    )

    if "Time" in processed_df.columns:
        processed_df["Hour"] = pd.to_datetime(
            processed_df["Time"],
            format="%H:%M",
            errors="coerce",
        ).dt.hour

    return processed_df


def assign_season(date_value: pd.Timestamp) -> str | None:
    """
    Assign a season label based on match date.

    Parameters
    ----------
    date_value : pd.Timestamp
        Match date.

    Returns
    -------
    str | None
        Season label or None if outside supported ranges.
    """
    if pd.isna(date_value):
        return None

    season_ranges = [
        ("2019-2020", pd.Timestamp("2019-07-26"), pd.Timestamp("2020-05-09")),
        ("2020-2021", pd.Timestamp("2020-08-07"), pd.Timestamp("2021-05-23")),
        ("2021-2022", pd.Timestamp("2021-07-23"), pd.Timestamp("2022-05-22")),
        ("2022-2023", pd.Timestamp("2022-07-22"), pd.Timestamp("2023-06-04")),
        ("2023-2024", pd.Timestamp("2023-07-28"), pd.Timestamp("2024-06-02")),
    ]

    for season_name, start_date, end_date in season_ranges:
        if start_date <= date_value <= end_date:
            return season_name

    return None


def add_season_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a season label column from the Date column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with Season column.
    """
    processed_df = df.copy()
    processed_df["Season"] = processed_df["Date"].apply(assign_season)
    return processed_df