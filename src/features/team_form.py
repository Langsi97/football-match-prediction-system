"""
Team form feature engineering for Belgian Pro League match data.

This module computes leakage-free form features using up to the previous
5 matches played by each team, regardless of venue.

Form definition:
- Win  = 3 points
- Draw = 1 point
- Loss = 0 points

Normalized form:
    sum(last_m_points) / (3 * m), where m <= 5

This ensures the feature is scaled between 0 and 1.

Output columns:
- HomeTeam_Form
- AwayTeam_Form
"""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd


FORM_WINDOW = 5
REQUIRED_COLUMNS = {"Date", "HomeTeam", "AwayTeam", "FTR"}


def validate_required_columns(df: pd.DataFrame, required_columns: set[str]) -> None:
    """
    Validate that the dataframe contains all required columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    required_columns : set[str]
        Required columns.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _compute_home_points(result: str) -> int:
    """
    Compute points earned by the home team from FTR.
    """
    if result == "H":
        return 3
    if result == "D":
        return 1
    return 0


def _compute_away_points(result: str) -> int:
    """
    Compute points earned by the away team from FTR.
    """
    if result == "A":
        return 3
    if result == "D":
        return 1
    return 0


def build_team_points_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand the match-level dataframe into one row per team per match
    with points earned in that match.

    Parameters
    ----------
    df : pd.DataFrame
        Match-level dataframe.

    Returns
    -------
    pd.DataFrame
        Team-level points dataframe.
    """
    validate_required_columns(df, REQUIRED_COLUMNS)

    working_df = df.copy()
    working_df["Date"] = pd.to_datetime(working_df["Date"], errors="coerce")

    if working_df["Date"].isna().any():
        raise ValueError("Date column contains invalid values after datetime conversion.")

    if "MatchID" not in working_df.columns:
        working_df["MatchID"] = range(len(working_df))

    home_df = working_df[["MatchID", "Date", "HomeTeam", "FTR"]].copy()
    home_df = home_df.rename(columns={"HomeTeam": "Team"})
    home_df["Venue"] = "Home"
    home_df["Points"] = home_df["FTR"].apply(_compute_home_points)

    away_df = working_df[["MatchID", "Date", "AwayTeam", "FTR"]].copy()
    away_df = away_df.rename(columns={"AwayTeam": "Team"})
    away_df["Venue"] = "Away"
    away_df["Points"] = away_df["FTR"].apply(_compute_away_points)

    team_points = pd.concat([home_df, away_df], axis=0, ignore_index=True)

    # Deterministic order for reproducibility.
    team_points = team_points.sort_values(
        ["Team", "Date", "MatchID"], ascending=[True, True, True]
    ).reset_index(drop=True)

    return team_points


def add_form_feature(team_points_df: pd.DataFrame, form_window: int = FORM_WINDOW) -> pd.DataFrame:
    """
    Add normalized team form based on up to the previous `form_window` matches.

    Parameters
    ----------
    team_points_df : pd.DataFrame
        Team-level points dataframe.
    form_window : int, default=5
        Maximum number of past matches used to compute form.

    Returns
    -------
    pd.DataFrame
        Team-level dataframe with Form column added.
    """
    result = team_points_df.copy()

    # Previous matches only -> shift(1)
    shifted_points = result.groupby("Team", group_keys=False)["Points"].transform(lambda s: s.shift(1))

    rolling_sum = (
        shifted_points.groupby(result["Team"])
        .transform(lambda s: s.rolling(window=form_window, min_periods=1).sum())
    )

    rolling_count = (
        shifted_points.groupby(result["Team"])
        .transform(lambda s: s.rolling(window=form_window, min_periods=1).count())
    )

    # Normalize by maximum possible points from the observed history length.
    result["Form"] = rolling_sum / (3 * rolling_count)

    # If no prior matches exist, rolling_count is NaN/0 and Form stays NaN.
    return result


def split_home_away_form(team_form_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split team form back into home and away match-level features.

    Parameters
    ----------
    team_form_df : pd.DataFrame
        Team-level dataframe containing Form.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (home_form_df, away_form_df)
    """
    home_form = (
        team_form_df.loc[team_form_df["Venue"] == "Home", ["MatchID", "Form"]]
        .rename(columns={"Form": "HomeTeam_Form"})
        .copy()
    )

    away_form = (
        team_form_df.loc[team_form_df["Venue"] == "Away", ["MatchID", "Form"]]
        .rename(columns={"Form": "AwayTeam_Form"})
        .copy()
    )

    return home_form, away_form


def merge_form_back(
    matches_df: pd.DataFrame,
    home_form_df: pd.DataFrame,
    away_form_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge home and away form features back into the match-level dataframe.

    Parameters
    ----------
    matches_df : pd.DataFrame
        Match-level dataframe.
    home_form_df : pd.DataFrame
        Home form features by MatchID.
    away_form_df : pd.DataFrame
        Away form features by MatchID.

    Returns
    -------
    pd.DataFrame
        Dataframe with HomeTeam_Form and AwayTeam_Form added.
    """
    result = matches_df.copy()

    if "MatchID" not in result.columns:
        result["MatchID"] = range(len(result))

    original_row_count = len(result)

    result = result.merge(home_form_df, on="MatchID", how="left")
    result = result.merge(away_form_df, on="MatchID", how="left")

    if len(result) != original_row_count:
        raise ValueError(
            f"Row count changed after form merge: {original_row_count} -> {len(result)}"
        )

    return result


def build_team_form_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    End-to-end builder for normalized team form features.

    Parameters
    ----------
    df : pd.DataFrame
        Match-level dataframe.

    Returns
    -------
    pd.DataFrame
        Match-level dataframe with HomeTeam_Form and AwayTeam_Form added.
    """
    working_df = df.copy()
    if "MatchID" not in working_df.columns:
        working_df["MatchID"] = range(len(working_df))

    team_points = build_team_points_table(working_df)
    team_points = add_form_feature(team_points, form_window=FORM_WINDOW)
    home_form_df, away_form_df = split_home_away_form(team_points)

    final_df = merge_form_back(working_df, home_form_df, away_form_df)

    if len(final_df) != len(df):
        raise ValueError("Final form dataset does not preserve row count.")

    return final_df