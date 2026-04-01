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

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    if "AvgA" not in df.columns:
        raise KeyError("'AvgA' column not found in dataframe.")

    end_index = df.columns.get_loc("AvgA")
    return df.iloc[:, : end_index + 1].copy()


def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns not needed in pre-match dataset.
    """
    columns_to_drop = ["HTHG", "HTAG", "HTR"]
    existing_columns = [col for col in columns_to_drop if col in df.columns]
    return df.drop(columns=existing_columns).copy()


def remove_bookmaker_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove all bookmaker-related columns to prevent leakage.

    Removes columns starting with:
    B365, BW, IW, PS, WH, VC, Avg, Max
    """
    bookmaker_prefixes = ("B365", "BW", "IW", "PS", "WH", "VC", "Avg", "Max")

    cols_to_drop = [col for col in df.columns if col.startswith(bookmaker_prefixes)]

    return df.drop(columns=cols_to_drop).copy()


def preprocess_date_and_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert match date to datetime and extract hour from time.
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
    Add a season label column.
    """
    processed_df = df.copy()
    processed_df["Season"] = processed_df["Date"].apply(assign_season)
    return processed_df


def run_base_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full base preprocessing pipeline.

    Order is important:
    1. Trim columns
    2. Remove bookmaker data
    3. Drop irrelevant columns
    4. Process date/time
    5. Add season
    """
    df = trim_columns_up_to_avg_a(df)
    df = remove_bookmaker_columns(df)   # 🔥 critical step
    df = drop_irrelevant_columns(df)
    df = preprocess_date_and_time(df)
    df = add_season_column(df)

    return df