from __future__ import annotations

from io import BytesIO
from typing import BinaryIO

import pandas as pd


REQUIRED_BASE_COLUMNS = [
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "HS",
    "AS",
    "HST",
    "AST",
    "HF",
    "AF",
    "HC",
    "AC",
]

OPTIONAL_CARD_COLUMNS = ["HY", "AY", "HR", "AR"]


def read_uploaded_match_file(uploaded_file: BinaryIO) -> pd.DataFrame:
    """
    Read an uploaded CSV or Excel file into a pandas DataFrame.
    """
    filename = uploaded_file.name.lower()

    if filename.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif filename.endswith(".xlsx") or filename.endswith(".xls"):
        file_bytes = uploaded_file.read()
        df = pd.read_excel(BytesIO(file_bytes))
    else:
        raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")

    return df


def validate_uploaded_match_data(df: pd.DataFrame) -> None:
    """
    Validate that the uploaded file contains the minimum columns needed
    to compute the model features.
    """
    missing_columns = [col for col in REQUIRED_BASE_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Uploaded file is missing required columns: {missing_columns}"
        )


def standardize_uploaded_match_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize uploaded historical match data.
    """
    standardized_df = df.copy()

    validate_uploaded_match_data(standardized_df)

    standardized_df["Date"] = pd.to_datetime(
        standardized_df["Date"],
        dayfirst=True,
        errors="coerce",
    )

    if standardized_df["Date"].isna().all():
        raise ValueError(
            "The Date column could not be parsed. Please provide a valid match date column."
        )

    for col in REQUIRED_BASE_COLUMNS[3:] + [c for c in OPTIONAL_CARD_COLUMNS if c in standardized_df.columns]:
        standardized_df[col] = pd.to_numeric(standardized_df[col], errors="coerce")

    standardized_df["HomeTeam"] = standardized_df["HomeTeam"].astype(str).str.strip()
    standardized_df["AwayTeam"] = standardized_df["AwayTeam"].astype(str).str.strip()

    standardized_df = standardized_df.sort_values("Date").reset_index(drop=True)

    return standardized_df


def build_team_long_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert match-level data into a team-centric long format so rolling
    last-5 computations can be done consistently for both home and away teams.
    """
    home_df = pd.DataFrame(
        {
            "Date": df["Date"],
            "Team": df["HomeTeam"],
            "Opponent": df["AwayTeam"],
            "is_home": 1,
            "goals_scored": df["FTHG"],
            "goals_conceded": df["FTAG"],
            "shots_for": df["HS"],
            "shots_against": df["AS"],
            "shots_on_target_for": df["HST"],
            "shots_on_target_against": df["AST"],
            "fouls_for": df["HF"],
            "fouls_against": df["AF"],
            "corners_for": df["HC"],
            "corners_against": df["AC"],
            "yellow_cards_for": df["HY"] if "HY" in df.columns else 1.5,
            "yellow_cards_against": df["AY"] if "AY" in df.columns else 1.5,
            "red_cards_for": df["HR"] if "HR" in df.columns else 0.1,
            "red_cards_against": df["AR"] if "AR" in df.columns else 0.1,
            "result_points": (
                (df["FTHG"] > df["FTAG"]).astype(int) * 3
                + (df["FTHG"] == df["FTAG"]).astype(int) * 1
            ),
        }
    )

    away_df = pd.DataFrame(
        {
            "Date": df["Date"],
            "Team": df["AwayTeam"],
            "Opponent": df["HomeTeam"],
            "is_home": 0,
            "goals_scored": df["FTAG"],
            "goals_conceded": df["FTHG"],
            "shots_for": df["AS"],
            "shots_against": df["HS"],
            "shots_on_target_for": df["AST"],
            "shots_on_target_against": df["HST"],
            "fouls_for": df["AF"],
            "fouls_against": df["HF"],
            "corners_for": df["AC"],
            "corners_against": df["HC"],
            "yellow_cards_for": df["AY"] if "AY" in df.columns else 1.5,
            "yellow_cards_against": df["HY"] if "HY" in df.columns else 1.5,
            "red_cards_for": df["AR"] if "AR" in df.columns else 0.1,
            "red_cards_against": df["HR"] if "HR" in df.columns else 0.1,
            "result_points": (
                (df["FTAG"] > df["FTHG"]).astype(int) * 3
                + (df["FTAG"] == df["FTHG"]).astype(int) * 1
            ),
        }
    )

    long_df = pd.concat([home_df, away_df], axis=0, ignore_index=True)
    long_df = long_df.sort_values(["Team", "Date"]).reset_index(drop=True)

    return long_df


def _get_last_n_matches(team_history_df: pd.DataFrame, team_name: str, n: int = 5) -> pd.DataFrame:
    """
    Return the most recent n matches for a team.
    """
    team_df = team_history_df[team_history_df["Team"].str.lower() == team_name.strip().lower()].copy()

    if team_df.empty:
        raise ValueError(f"No historical matches found for team: {team_name}")

    team_df = team_df.sort_values("Date")
    last_n_df = team_df.tail(n)

    if len(last_n_df) < n:
        raise ValueError(
            f"Team '{team_name}' has only {len(last_n_df)} historical matches in the uploaded file. "
            f"At least {n} matches are required."
        )

    return last_n_df


def compute_form_from_last_matches(last_matches_df: pd.DataFrame) -> float:
    """
    Compute normalized form as points from last 5 matches divided by 15.
    """
    total_points = last_matches_df["result_points"].sum()
    return float(total_points / 15.0)


def compute_home_advantage_from_last_home_matches(
    team_history_df: pd.DataFrame,
    home_team: str,
    n: int = 5,
) -> float:
    """
    Compute adaptive home advantage using the last n home matches only.
    """
    home_only_df = team_history_df[
        (team_history_df["Team"].str.lower() == home_team.strip().lower())
        & (team_history_df["is_home"] == 1)
    ].copy()

    home_only_df = home_only_df.sort_values("Date").tail(n)

    if home_only_df.empty:
        return 0.50

    total_points = home_only_df["result_points"].sum()
    max_points = len(home_only_df) * 3

    if max_points == 0:
        return 0.50

    return float(total_points / max_points)


def compute_average_feature_block(last_matches_df: pd.DataFrame, prefix: str) -> dict:
    """
    Compute the rolling average features used by the model for one team.
    """
    return {
        f"{prefix}_goals_scored_last5": float(last_matches_df["goals_scored"].mean()),
        f"{prefix}_goals_conceded_last5": float(last_matches_df["goals_conceded"].mean()),
        f"{prefix}_shots_for_last5": float(last_matches_df["shots_for"].mean()),
        f"{prefix}_shots_against_last5": float(last_matches_df["shots_against"].mean()),
        f"{prefix}_shots_on_target_for_last5": float(last_matches_df["shots_on_target_for"].mean()),
        f"{prefix}_shots_on_target_against_last5": float(last_matches_df["shots_on_target_against"].mean()),
        f"{prefix}_fouls_for_last5": float(last_matches_df["fouls_for"].mean()),
        f"{prefix}_fouls_against_last5": float(last_matches_df["fouls_against"].mean()),
        f"{prefix}_corners_for_last5": float(last_matches_df["corners_for"].mean()),
        f"{prefix}_corners_against_last5": float(last_matches_df["corners_against"].mean()),
        f"{prefix}_yellow_cards_for_last5": float(last_matches_df["yellow_cards_for"].mean()),
        f"{prefix}_yellow_cards_against_last5": float(last_matches_df["yellow_cards_against"].mean()),
        f"{prefix}_red_cards_for_last5": float(last_matches_df["red_cards_for"].mean()),
        f"{prefix}_red_cards_against_last5": float(last_matches_df["red_cards_against"].mean()),
    }


def compute_league_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a league table from all uploaded historical matches.
    """
    home_table = pd.DataFrame(
        {
            "Team": df["HomeTeam"],
            "points": (
                (df["FTHG"] > df["FTAG"]).astype(int) * 3
                + (df["FTHG"] == df["FTAG"]).astype(int) * 1
            ),
            "goals_for": df["FTHG"],
            "goals_against": df["FTAG"],
        }
    )

    away_table = pd.DataFrame(
        {
            "Team": df["AwayTeam"],
            "points": (
                (df["FTAG"] > df["FTHG"]).astype(int) * 3
                + (df["FTAG"] == df["FTHG"]).astype(int) * 1
            ),
            "goals_for": df["FTAG"],
            "goals_against": df["FTHG"],
        }
    )

    table_df = pd.concat([home_table, away_table], axis=0, ignore_index=True)
    league_table = (
        table_df.groupby("Team", as_index=False)
        .agg(
            points=("points", "sum"),
            goals_for=("goals_for", "sum"),
            goals_against=("goals_against", "sum"),
        )
    )

    league_table["goal_difference"] = league_table["goals_for"] - league_table["goals_against"]

    league_table = league_table.sort_values(
        by=["points", "goal_difference", "goals_for"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    league_table["position"] = league_table.index + 1

    return league_table


def get_team_position(league_table: pd.DataFrame, team_name: str) -> int:
    """
    Return the current position of a team from the computed league table.
    """
    team_row = league_table[league_table["Team"].str.lower() == team_name.strip().lower()]

    if team_row.empty:
        raise ValueError(f"Team '{team_name}' is not present in the uploaded historical dataset.")

    return int(team_row.iloc[0]["position"])


def build_auto_features_from_uploaded_history(
    uploaded_df: pd.DataFrame,
    home_team: str,
    away_team: str,
    matchday: int,
    hour: float,
) -> dict:
    """
    Build the full model input feature dictionary automatically from uploaded historical match data.
    """
    standardized_df = standardize_uploaded_match_data(uploaded_df)
    long_history_df = build_team_long_history(standardized_df)

    home_last5_df = _get_last_n_matches(long_history_df, team_name=home_team, n=5)
    away_last5_df = _get_last_n_matches(long_history_df, team_name=away_team, n=5)

    league_table = compute_league_table(standardized_df)

    home_pre_position = get_team_position(league_table, home_team)
    away_pre_position = get_team_position(league_table, away_team)

    home_form = compute_form_from_last_matches(home_last5_df)
    away_form = compute_form_from_last_matches(away_last5_df)
    home_advantage = compute_home_advantage_from_last_home_matches(long_history_df, home_team, n=5)

    home_feature_block = compute_average_feature_block(home_last5_df, prefix="home")
    away_feature_block = compute_average_feature_block(away_last5_df, prefix="away")

    model_inputs = {
        "matchday": float(matchday),
        "hour": float(hour),
        "home_pre_position": float(home_pre_position),
        "away_pre_position": float(away_pre_position),
        "home_form": float(home_form),
        "away_form": float(away_form),
        "home_advantage": float(home_advantage),
        **home_feature_block,
        **away_feature_block,
    }

    return model_inputs


def build_feature_preview_df(model_inputs: dict) -> pd.DataFrame:
    """
    Create a preview dataframe of automatically generated features for display in the app.
    """
    preview_df = pd.DataFrame(
        {
            "feature": list(model_inputs.keys()),
            "value": list(model_inputs.values()),
        }
    )
    return preview_df