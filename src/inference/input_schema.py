"""
Input schema for Streamlit/manual entry.

This module defines the exact last-5-match statistics the UI will collect
from the user and maps them to model feature names.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd


UI_TO_MODEL_FEATURE_MAP: Dict[str, str] = {
    # Home team
    "home_goals_scored_last5": "Home_GoalsFor_roll5",
    "home_goals_conceded_last5": "Home_GoalsAgainst_roll5",
    "home_shots_off_target_last5": "Home_ShotsAgainst_roll5",  # optional adjust if you model exact off-target separately
    "home_shots_on_target_last5": "Home_ShotsOnTargetFor_roll5",
    "home_corners_last5": "Home_CornersFor_roll5",
    "home_cards_last5": "Home_YellowCardsFor_roll5",

    # Away team
    "away_goals_scored_last5": "Away_GoalsFor_roll5",
    "away_goals_conceded_last5": "Away_GoalsAgainst_roll5",
    "away_shots_off_target_last5": "Away_ShotsAgainst_roll5",
    "away_shots_on_target_last5": "Away_ShotsOnTargetFor_roll5",
    "away_corners_last5": "Away_CornersFor_roll5",
    "away_cards_last5": "Away_YellowCardsFor_roll5",

    # Additional engineered inputs still needed by your model
    "home_form": "HomeTeam_Form",
    "away_form": "AwayTeam_Form",
    "home_advantage": "HomeTeam_HomeAdvantage",
    "home_pre_position": "Home_pre_po",
    "away_pre_position": "Away_pre_po",
    "hour": "Hour",
}


def build_feature_ready_row(user_inputs: Dict[str, float]) -> pd.DataFrame:
    """
    Convert Streamlit form inputs into a single-row dataframe
    using model feature names.
    """
    renamed = {}

    for ui_name, model_name in UI_TO_MODEL_FEATURE_MAP.items():
        if ui_name not in user_inputs:
            raise ValueError(f"Missing required UI input: {ui_name}")
        renamed[model_name] = user_inputs[ui_name]

    return pd.DataFrame([renamed])


def required_ui_fields() -> List[str]:
    return list(UI_TO_MODEL_FEATURE_MAP.keys())
