"""
Streamlit application for Belgian Jupiler Pro League match prediction.

What this code does:
- Builds a two-page football prediction app.
- Page 1 lets the user predict a match outcome.
- The user can either:
    1. enter features manually, or
    2. upload historical match data so the app auto-computes the last-5-match features.
- The user always inputs the home and away team pre-match league positions manually,
  whether using manual entry or file upload mode.
- Sends prediction requests to a FastAPI backend instead of loading the model directly.
- Generates SHAP-based local explainability for the exact class already predicted by the app.
- Keeps the SHAP dashboard aligned with the prediction output shown to the user.
- Page 2 compares model probabilities against bookmaker odds and computes overround.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))
import os
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

from src.explainability.shap_explainer import compute_shap_explanation
from src.inference.input_schema import build_feature_ready_row

import joblib

MODEL_PATH = "artifacts/models/random_forest_best.joblib"
model = joblib.load(MODEL_PATH)

st.set_page_config(
    page_title="Jupiler Pro League Match Predictor",
    page_icon="⚽",
    layout="wide",
)

prediction = model.predict(input_df)[0]
probabilities = model.predict_proba(input_df)[0]

JPL_TEAMS = [
    "Anderlecht",
    "Antwerp",
    "Cercle Brugge",
    "Charleroi",
    "Club Brugge",
    "Dender",
    "Genk",
    "Gent",
    "Kortrijk",
    "Leuven",
    "Mechelen",
    "Mouscron",
    "St Truiden",
    "Standard",
    "Union SG",
    "Waregem",
    "Westerlo",
    "Waasland-Beveren",
]

CLASS_NAME_MAP = {
    0: "Draw",
    1: "Home Win",
    2: "Away Win",
    "0": "Draw",
    "1": "Home Win",
    "2": "Away Win",
    "H": "Home Win",
    "D": "Draw",
    "A": "Away Win",
    "Home Win": "Home Win",
    "Draw": "Draw",
    "Away Win": "Away Win",
}

DEFAULT_HOUR = 20.0
DEFAULT_HOME_ADVANTAGE = 0.50
DEFAULT_HOME_YELLOW_REGISTERED = 1.50
DEFAULT_HOME_YELLOW_CONCEDED = 1.50
DEFAULT_HOME_RED_REGISTERED = 0.10
DEFAULT_HOME_RED_CONCEDED = 0.10
DEFAULT_AWAY_YELLOW_REGISTERED = 1.50
DEFAULT_AWAY_YELLOW_CONCEDED = 1.50
DEFAULT_AWAY_RED_REGISTERED = 0.10
DEFAULT_AWAY_RED_CONCEDED = 0.10

REQUIRED_UPLOAD_COLUMNS = [
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

OPTIONAL_UPLOAD_COLUMNS_WITH_DEFAULTS = {
    "HY": DEFAULT_HOME_YELLOW_REGISTERED,
    "AY": DEFAULT_AWAY_YELLOW_REGISTERED,
    "HR": DEFAULT_HOME_RED_REGISTERED,
    "AR": DEFAULT_AWAY_RED_CONCEDED,
    "Time": None,
}

if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Match Prediction"

if "latest_prediction_ready" not in st.session_state:
    st.session_state["latest_prediction_ready"] = False

if "latest_prediction_results" not in st.session_state:
    st.session_state["latest_prediction_results"] = None

if "latest_home_team" not in st.session_state:
    st.session_state["latest_home_team"] = ""

if "latest_away_team" not in st.session_state:
    st.session_state["latest_away_team"] = ""

if "latest_shap_explanation" not in st.session_state:
    st.session_state["latest_shap_explanation"] = None

if "latest_shap_error" not in st.session_state:
    st.session_state["latest_shap_error"] = None

if "latest_auto_features_df" not in st.session_state:
    st.session_state["latest_auto_features_df"] = None


def go_to_page(page_name: str) -> None:
    st.session_state["current_page"] = page_name


def normalize_prediction_label(prediction_value) -> str:
    prediction_str = str(prediction_value)
    return CLASS_NAME_MAP.get(prediction_value, CLASS_NAME_MAP.get(prediction_str, prediction_str))


def fair_odds_from_probability(prob: float) -> float:
    if prob <= 0:
        return float("inf")
    return 1.0 / prob


def bookmaker_implied_probability(odds: float) -> float:
    if odds <= 0:
        return 0.0
    return 1.0 / odds


def explain_probability_gap(probability_gap: float) -> str:
    if probability_gap > 0.03:
        return (
            "The model assigns a meaningfully higher probability than the bookmaker market. "
            "This may indicate possible value from the model's perspective."
        )
    if probability_gap < -0.03:
        return (
            "The bookmaker market assigns a meaningfully higher probability than the model. "
            "This outcome may be overpriced by the market."
        )
    return "The model and the bookmaker market are relatively aligned on this outcome."


def explain_overround(overround: float) -> str:
    if overround > 0:
        return (
            f"The overround is positive at {overround:.2%}. "
            f"This means the bookmaker has an edge of {overround:.2%} built into the market, regardless of stake size. "
            f"The total implied probability is above 100%, so the quoted market is priced in the bookmaker's favor."
        )
    if abs(overround) < 1e-12:
        return (
            "The overround is 0.00%. This means the market is fair in pricing terms because the total implied probability is exactly 100%. "
            "There is no built-in bookmaker margin from the odds alone."
        )
    return (
        f"The overround is negative at {overround:.2%}. "
        f"This means the total implied probability is below 100%, so there may be a possible value-bet or arbitrage opportunity. "
        f"In simple terms, the quoted market may be underpriced rather than favoring the bookmaker."
    )


def explain_two_way_overround(overround: float) -> str:
    if overround > 0:
        return (
            f"The 2-outcome market has a positive overround of {overround:.2%}. "
            f"This means the bookmaker has an embedded edge of {overround:.2%} in this two-way market, regardless of the stake size."
        )
    if abs(overround) < 1e-12:
        return (
            "The 2-outcome market has an overround of 0.00%. "
            "This means the market is fair in pricing terms because the total implied probability is exactly 100%."
        )
    return (
        f"The 2-outcome market has a negative overround of {overround:.2%}. "
        f"This may indicate a possible value or arbitrage-style pricing opportunity because the total implied probability is below 100%."
    )


def validate_teams(home_team: str, away_team: str) -> None:
    if not home_team:
        raise ValueError("Please provide a Home Team.")
    if not away_team:
        raise ValueError("Please provide an Away Team.")
    if home_team == away_team:
        raise ValueError("Home Team and Away Team must be different.")


def get_prediction_results(user_inputs: dict) -> pd.DataFrame:
    """
    Call FastAPI backend instead of local model.
    """
    try:
        response = requests.post(API_URL, json=user_inputs, timeout=30)

        if response.status_code != 200:
            raise ValueError(f"API error {response.status_code}: {response.text}")

        data = response.json()

        result_df = pd.DataFrame(
            {
                "prediction": [data["prediction"]],
                "prob_H": [data["prob_H"]],
                "prob_D": [data["prob_D"]],
                "prob_A": [data["prob_A"]],
            }
        )
        return result_df

    except Exception as exc:
        raise RuntimeError(f"Failed to call prediction API: {exc}") from exc


def get_shap_results(user_inputs: dict, target_prediction_label) -> dict:
    feature_df = build_feature_ready_row(user_inputs)
    return compute_shap_explanation(
        feature_df=feature_df,
        top_n=5,
        target_prediction_label=target_prediction_label,
    )


def render_footer_navigation(show_previous: bool, show_next: bool) -> None:
    st.markdown("---")
    left, right = st.columns(2)

    with left:
        if show_previous:
            st.button(
                "⬅ Previous Page",
                use_container_width=True,
                on_click=go_to_page,
                args=("Match Prediction",),
                key=f"prev_btn_{st.session_state['current_page']}",
            )

    with right:
        if show_next:
            st.button(
                "Next Page ➡",
                use_container_width=True,
                on_click=go_to_page,
                args=("Analyse Bookmaker Bias",),
                key=f"next_btn_{st.session_state['current_page']}",
            )


def render_disclaimer() -> None:
    st.warning(
        "Disclaimer: This application is provided strictly for informational and educational purposes only. "
        "It does not constitute financial advice, betting advice, investment advice, or any guaranteed decision-support tool. "
        "Sports betting carries a serious risk of financial loss. Anyone considering sports betting should consult qualified professionals "
        "specialized in responsible gambling, financial decision-making, or related advisory services. "
        "The author accepts no responsibility or liability for any loss, damage, or consequence resulting from the use of this application "
        "as financial advice, betting advice, or as a source of income."
    )


def team_input_block(prefix: str, label: str, key_prefix: str) -> str:
    mode = st.radio(
        f"{label} input mode",
        options=["Select from list", "Type manually"],
        horizontal=True,
        key=f"{key_prefix}_{prefix}_team_mode",
    )

    if mode == "Select from list":
        return st.selectbox(
            label,
            JPL_TEAMS,
            key=f"{key_prefix}_{prefix}_team_select",
        )

    return st.text_input(
        label,
        value="",
        placeholder="Type team name exactly as it appears in your uploaded match data if using file upload mode.",
        key=f"{key_prefix}_{prefix}_team_text",
    ).strip()


def load_uploaded_match_file(uploaded_file) -> pd.DataFrame:
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif file_name.endswith(".xlsx") or file_name.endswith(".xls"):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")

    return df


def validate_and_prepare_uploaded_match_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        raise ValueError("The uploaded file is empty.")

    df = raw_df.copy()

    missing_required = [col for col in REQUIRED_UPLOAD_COLUMNS if col not in df.columns]
    if missing_required:
        raise ValueError(
            f"The uploaded file is missing required columns: {missing_required}"
        )

    for optional_col, default_value in OPTIONAL_UPLOAD_COLUMNS_WITH_DEFAULTS.items():
        if optional_col not in df.columns:
            df[optional_col] = default_value

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    if df["Date"].isna().all():
        raise ValueError("Could not parse the 'Date' column. Please check the date format.")

    df = df.dropna(subset=["Date"]).copy()

    numeric_columns = [
        "FTHG", "FTAG", "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC",
        "HY", "AY", "HR", "AR",
    ]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df[numeric_columns].isna().any().any():
        df[numeric_columns] = df[numeric_columns].fillna(0.0)

    if "Time" in df.columns:
        df["Time"] = df["Time"].fillna("").astype(str)

    df["HomeTeam"] = df["HomeTeam"].astype(str).str.strip()
    df["AwayTeam"] = df["AwayTeam"].astype(str).str.strip()

    sort_columns = ["Date"]
    if "Time" in df.columns:
        sort_columns.append("Time")

    df = df.sort_values(sort_columns).reset_index(drop=True)
    return df


def build_team_centric_history(matches_df: pd.DataFrame) -> pd.DataFrame:
    home_df = pd.DataFrame(
        {
            "Date": matches_df["Date"],
            "Time": matches_df["Time"],
            "team": matches_df["HomeTeam"],
            "opponent": matches_df["AwayTeam"],
            "is_home": 1,
            "goals_for": matches_df["FTHG"],
            "goals_against": matches_df["FTAG"],
            "shots_for": matches_df["HS"],
            "shots_against": matches_df["AS"],
            "shots_on_target_for": matches_df["HST"],
            "shots_on_target_against": matches_df["AST"],
            "fouls_for": matches_df["HF"],
            "fouls_against": matches_df["AF"],
            "corners_for": matches_df["HC"],
            "corners_against": matches_df["AC"],
            "yellow_cards_for": matches_df["HY"],
            "yellow_cards_against": matches_df["AY"],
            "red_cards_for": matches_df["HR"],
            "red_cards_against": matches_df["AR"],
        }
    )

    away_df = pd.DataFrame(
        {
            "Date": matches_df["Date"],
            "Time": matches_df["Time"],
            "team": matches_df["AwayTeam"],
            "opponent": matches_df["HomeTeam"],
            "is_home": 0,
            "goals_for": matches_df["FTAG"],
            "goals_against": matches_df["FTHG"],
            "shots_for": matches_df["AS"],
            "shots_against": matches_df["HS"],
            "shots_on_target_for": matches_df["AST"],
            "shots_on_target_against": matches_df["HST"],
            "fouls_for": matches_df["AF"],
            "fouls_against": matches_df["HF"],
            "corners_for": matches_df["AC"],
            "corners_against": matches_df["HC"],
            "yellow_cards_for": matches_df["AY"],
            "yellow_cards_against": matches_df["HY"],
            "red_cards_for": matches_df["AR"],
            "red_cards_against": matches_df["HR"],
        }
    )

    history_df = pd.concat([home_df, away_df], ignore_index=True)
    history_df["points"] = 0
    history_df.loc[history_df["goals_for"] > history_df["goals_against"], "points"] = 3
    history_df.loc[history_df["goals_for"] == history_df["goals_against"], "points"] = 1
    history_df = history_df.sort_values(["Date", "Time"]).reset_index(drop=True)
    return history_df


def get_team_last_n_matches(team_history_df: pd.DataFrame, team_name: str, n_matches: int = 5) -> pd.DataFrame:
    team_df = team_history_df[team_history_df["team"] == team_name].copy()
    team_df = team_df.sort_values(["Date", "Time"]).reset_index(drop=True)

    if len(team_df) < n_matches:
        raise ValueError(
            f"Not enough historical matches found for team '{team_name}'. "
            f"At least {n_matches} past matches are required, but only {len(team_df)} were found."
        )

    return team_df.tail(n_matches).reset_index(drop=True)


def compute_team_recent_features(team_recent_df: pd.DataFrame, prefix: str) -> dict:
    feature_map = {
        f"{prefix}_goals_scored_last5": team_recent_df["goals_for"].mean(),
        f"{prefix}_goals_conceded_last5": team_recent_df["goals_against"].mean(),
        f"{prefix}_shots_for_last5": team_recent_df["shots_for"].mean(),
        f"{prefix}_shots_against_last5": team_recent_df["shots_against"].mean(),
        f"{prefix}_shots_on_target_for_last5": team_recent_df["shots_on_target_for"].mean(),
        f"{prefix}_shots_on_target_against_last5": team_recent_df["shots_on_target_against"].mean(),
        f"{prefix}_fouls_for_last5": team_recent_df["fouls_for"].mean(),
        f"{prefix}_fouls_against_last5": team_recent_df["fouls_against"].mean(),
        f"{prefix}_corners_for_last5": team_recent_df["corners_for"].mean(),
        f"{prefix}_corners_against_last5": team_recent_df["corners_against"].mean(),
        f"{prefix}_yellow_cards_for_last5": team_recent_df["yellow_cards_for"].mean(),
        f"{prefix}_yellow_cards_against_last5": team_recent_df["yellow_cards_against"].mean(),
        f"{prefix}_red_cards_for_last5": team_recent_df["red_cards_for"].mean(),
        f"{prefix}_red_cards_against_last5": team_recent_df["red_cards_against"].mean(),
    }
    return {k: float(v) for k, v in feature_map.items()}


def auto_build_user_inputs_from_uploaded_history(
    uploaded_matches_df: pd.DataFrame,
    home_team: str,
    away_team: str,
    matchday: float,
    match_hour: float,
    home_pre_position: float,
    away_pre_position: float,
) -> tuple[dict, pd.DataFrame]:
    team_history_df = build_team_centric_history(uploaded_matches_df)

    home_recent_df = get_team_last_n_matches(team_history_df, home_team, n_matches=5)
    away_recent_df = get_team_last_n_matches(team_history_df, away_team, n_matches=5)

    home_form = float(home_recent_df["points"].sum() / 15.0)
    away_form = float(away_recent_df["points"].sum() / 15.0)

    home_features = compute_team_recent_features(home_recent_df, prefix="home")
    away_features = compute_team_recent_features(away_recent_df, prefix="away")

    user_inputs = {
        "matchday": float(matchday),
        "hour": float(match_hour),
        "home_pre_position": float(home_pre_position),
        "away_pre_position": float(away_pre_position),
        "home_form": home_form,
        "away_form": away_form,
        "home_advantage": float(DEFAULT_HOME_ADVANTAGE),
        **home_features,
        **away_features,
    }

    preview_df = pd.DataFrame(
        {
            "feature": list(user_inputs.keys()),
            "value": list(user_inputs.values()),
        }
    )

    return user_inputs, preview_df


def stat_input_block(prefix: str, title: str, key_prefix: str) -> dict:
    st.subheader(title)
    c1, c2 = st.columns(2)

    values = {}
    values[f"{prefix}_goals_scored_last5"] = c1.number_input("Goals registered", min_value=0.0, value=1.0, step=0.1, key=f"{key_prefix}_{prefix}_goals_registered")
    values[f"{prefix}_goals_conceded_last5"] = c2.number_input("Goals conceded", min_value=0.0, value=1.0, step=0.1, key=f"{key_prefix}_{prefix}_goals_conceded")
    values[f"{prefix}_shots_for_last5"] = c1.number_input("Shots registered", min_value=0.0, value=4.0, step=0.1, key=f"{key_prefix}_{prefix}_shots_registered")
    values[f"{prefix}_shots_against_last5"] = c2.number_input("Shots conceded", min_value=0.0, value=4.0, step=0.1, key=f"{key_prefix}_{prefix}_shots_conceded")
    values[f"{prefix}_shots_on_target_for_last5"] = c1.number_input("Shots on target registered", min_value=0.0, value=2.0, step=0.1, key=f"{key_prefix}_{prefix}_shots_ot_registered")
    values[f"{prefix}_shots_on_target_against_last5"] = c2.number_input("Shots on target conceded", min_value=0.0, value=2.0, step=0.1, key=f"{key_prefix}_{prefix}_shots_ot_conceded")
    values[f"{prefix}_fouls_for_last5"] = c1.number_input("Fouls registered", min_value=0.0, value=10.0, step=0.1, key=f"{key_prefix}_{prefix}_fouls_registered")
    values[f"{prefix}_fouls_against_last5"] = c2.number_input("Fouls conceded", min_value=0.0, value=10.0, step=0.1, key=f"{key_prefix}_{prefix}_fouls_conceded")
    values[f"{prefix}_corners_for_last5"] = c1.number_input("Corners registered", min_value=0.0, value=5.0, step=0.1, key=f"{key_prefix}_{prefix}_corners_registered")
    values[f"{prefix}_corners_against_last5"] = c2.number_input("Corners conceded", min_value=0.0, value=5.0, step=0.1, key=f"{key_prefix}_{prefix}_corners_conceded")
    values[f"{prefix}_yellow_cards_for_last5"] = c1.number_input("Yellow cards registered", min_value=0.0, value=DEFAULT_HOME_YELLOW_REGISTERED if prefix == "home" else DEFAULT_AWAY_YELLOW_REGISTERED, step=0.1, key=f"{key_prefix}_{prefix}_yellow_registered")
    values[f"{prefix}_yellow_cards_against_last5"] = c2.number_input("Yellow cards conceded", min_value=0.0, value=DEFAULT_HOME_YELLOW_CONCEDED if prefix == "home" else DEFAULT_AWAY_YELLOW_CONCEDED, step=0.1, key=f"{key_prefix}_{prefix}_yellow_conceded")
    values[f"{prefix}_red_cards_for_last5"] = c1.number_input("Red cards registered", min_value=0.0, value=DEFAULT_HOME_RED_REGISTERED if prefix == "home" else DEFAULT_AWAY_RED_REGISTERED, step=0.1, key=f"{key_prefix}_{prefix}_red_registered")
    values[f"{prefix}_red_cards_against_last5"] = c2.number_input("Red cards conceded", min_value=0.0, value=DEFAULT_HOME_RED_CONCEDED if prefix == "home" else DEFAULT_AWAY_RED_CONCEDED, step=0.1, key=f"{key_prefix}_{prefix}_red_conceded")
    return values


def collect_model_inputs(key_prefix: str) -> tuple[dict, str, str, str, pd.DataFrame | None]:
    st.subheader("Match Context")

    c1, c2 = st.columns(2)
    with c1:
        home_team = team_input_block("home", "Home Team", key_prefix)
    with c2:
        away_team = team_input_block("away", "Away Team", key_prefix)

    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        matchday = st.number_input("Matchday", min_value=1, max_value=40, value=10, step=1, key=f"{key_prefix}_matchday")
    with row1_col2:
        match_hour = st.number_input("Match Hour", min_value=0.0, max_value=23.0, value=float(DEFAULT_HOUR), step=1.0, key=f"{key_prefix}_match_hour")

    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        home_pre_position = st.number_input("Home pre-match position", min_value=1, max_value=20, value=6, step=1, key=f"{key_prefix}_home_pre_position")
    with row2_col2:
        away_pre_position = st.number_input("Away pre-match position", min_value=1, max_value=20, value=10, step=1, key=f"{key_prefix}_away_pre_position")

    st.markdown("---")
    st.subheader("Feature Input Mode")

    input_mode = st.radio(
        "Choose how to provide model features",
        options=["Manual feature entry", "Upload match data for better prediction"],
        horizontal=True,
        key=f"{key_prefix}_input_mode",
    )

    auto_preview_df = None

    if input_mode == "Upload match data for better prediction":
        st.info(
            "Upload historical match data so the system can automatically compute the last 5-match features used by the model. "
            "The home and away pre-match positions will still use the values you entered above."
        )

        uploaded_file = st.file_uploader(
            "Upload historical match data (CSV or Excel)",
            type=["csv", "xlsx", "xls"],
            key=f"{key_prefix}_uploaded_history_file",
        )

        if uploaded_file is None:
            st.warning("Please upload a CSV or Excel file to use automatic feature generation.")
            return {}, home_team, away_team, input_mode, auto_preview_df

        raw_uploaded_df = load_uploaded_match_file(uploaded_file)
        prepared_uploaded_df = validate_and_prepare_uploaded_match_data(raw_uploaded_df)

        st.markdown("### Uploaded Data Preview")
        st.dataframe(prepared_uploaded_df.head(10), use_container_width=True)

        user_inputs, auto_preview_df = auto_build_user_inputs_from_uploaded_history(
            uploaded_matches_df=prepared_uploaded_df,
            home_team=home_team,
            away_team=away_team,
            matchday=float(matchday),
            match_hour=float(match_hour),
            home_pre_position=float(home_pre_position),
            away_pre_position=float(away_pre_position),
        )

        return user_inputs, home_team, away_team, input_mode, auto_preview_df

    left, right = st.columns(2)

    with left:
        st.markdown("### Home Team Context")
        home_form = st.number_input("Home form (0–1)", min_value=0.0, max_value=1.0, value=0.50, step=0.01, key=f"{key_prefix}_home_form")

    with right:
        st.markdown("### Away Team Context")
        away_form = st.number_input("Away form (0–1)", min_value=0.0, max_value=1.0, value=0.50, step=0.01, key=f"{key_prefix}_away_form")

    st.markdown("---")

    left, right = st.columns(2)
    with left:
        home_inputs = stat_input_block("home", "Home Team — Last 5 Matches", key_prefix=key_prefix)
    with right:
        away_inputs = stat_input_block("away", "Away Team — Last 5 Matches", key_prefix=key_prefix)

    user_inputs = {
        "matchday": float(matchday),
        "hour": float(match_hour),
        "home_pre_position": float(home_pre_position),
        "away_pre_position": float(away_pre_position),
        "home_form": float(home_form),
        "away_form": float(away_form),
        "home_advantage": float(DEFAULT_HOME_ADVANTAGE),
        **home_inputs,
        **away_inputs,
    }

    return user_inputs, home_team, away_team, input_mode, auto_preview_df


def render_prediction_output(results: pd.DataFrame, home_team: str, away_team: str) -> None:
    row = results.iloc[0]
    raw_prediction = row["prediction"]
    friendly_prediction = normalize_prediction_label(raw_prediction)

    st.success(f"Predicted outcome for {home_team} vs {away_team}: {friendly_prediction}")

    p1, p2, p3 = st.columns(3)
    p1.metric("Home Win Probability", f"{float(row.get('prob_H', 0)):.2%}")
    p2.metric("Draw Probability", f"{float(row.get('prob_D', 0)):.2%}")
    p3.metric("Away Win Probability", f"{float(row.get('prob_A', 0)):.2%}")

    display_df = pd.DataFrame(
        {
            "Home Team": [home_team],
            "Away Team": [away_team],
            "Raw Prediction": [raw_prediction],
            "Prediction": [friendly_prediction],
            "Prob_H": [row.get("prob_H", 0)],
            "Prob_D": [row.get("prob_D", 0)],
            "Prob_A": [row.get("prob_A", 0)],
        }
    )
    st.dataframe(display_df, use_container_width=True)


def render_shap_bar_chart(contribution_df: pd.DataFrame, top_n: int = 10) -> None:
    if contribution_df.empty:
        st.info("No SHAP contributions are available for plotting.")
        return

    chart_df = contribution_df.head(top_n).copy()
    chart_df = chart_df.sort_values(by="shap_contribution", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(chart_df["feature"], chart_df["shap_contribution"])
    ax.set_xlabel("SHAP Contribution")
    ax.set_ylabel("Feature")
    ax.set_title("Top Feature Contributions for the Predicted Outcome")
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)


def render_explainability_section(shap_results: dict, prediction_results: pd.DataFrame) -> None:
    st.markdown("---")
    st.subheader("Model Explainability Dashboard")
    st.caption(
        "This section explains which input features most strongly supported or opposed the model's predicted outcome."
    )

    prediction_row = prediction_results.iloc[0]
    app_raw_prediction = prediction_row["prediction"]
    app_prediction_name = normalize_prediction_label(app_raw_prediction)

    target_class_name = shap_results.get("target_class_name", app_prediction_name)
    target_class_label = shap_results.get("target_class_label", app_raw_prediction)
    target_class_position = shap_results.get("target_class_position", "unknown")

    contribution_df = shap_results["feature_contributions_df"]
    top_positive_df = shap_results["top_positive_df"]
    top_negative_df = shap_results["top_negative_df"]

    st.info(
        f"Explanation target class: {target_class_name} "
        f"(prediction_output={app_raw_prediction}, internal_label={target_class_label}, class_position={target_class_position})"
    )

    debug_df = pd.DataFrame(
        {
            "class_name": ["Home Win", "Draw", "Away Win"],
            "predicted_probability": [
                float(prediction_row.get("prob_H", 0)),
                float(prediction_row.get("prob_D", 0)),
                float(prediction_row.get("prob_A", 0)),
            ],
        }
    )

    st.markdown("### Prediction Debug View")
    st.dataframe(debug_df, use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Top Factors Supporting the Prediction")
        if top_positive_df.empty:
            st.write("No positive contributors found.")
        else:
            st.dataframe(
                top_positive_df[["feature", "feature_value", "shap_contribution"]],
                use_container_width=True,
            )

    with c2:
        st.markdown("### Top Factors Opposing the Prediction")
        if top_negative_df.empty:
            st.write("No negative contributors found.")
        else:
            st.dataframe(
                top_negative_df[["feature", "feature_value", "shap_contribution"]],
                use_container_width=True,
            )

    st.markdown("### SHAP Contribution Chart")
    render_shap_bar_chart(contribution_df=contribution_df, top_n=10)

    st.markdown("### Full Feature Contribution Table")
    st.dataframe(
        contribution_df[
            ["feature", "feature_value", "shap_contribution", "direction", "abs_contribution"]
        ],
        use_container_width=True,
    )


st.title("⚽ Jupiler Pro League Match Prediction System")
st.caption(
    "Belgian Jupiler Pro League focused. Enter match features on page 1, then analyse bookmaker bias on page 2."
)
render_disclaimer()

selected_page = st.radio(
    "Navigation",
    options=["Match Prediction", "Analyse Bookmaker Bias"],
    index=0 if st.session_state["current_page"] == "Match Prediction" else 1,
    horizontal=True,
)

st.session_state["current_page"] = selected_page

if st.session_state["current_page"] == "Match Prediction":
    try:
        user_inputs, home_team, away_team, input_mode, auto_preview_df = collect_model_inputs("prediction_page")
    except Exception as input_error:
        st.error(f"Input preparation failed: {input_error}")
        user_inputs, home_team, away_team, input_mode, auto_preview_df = {}, "", "", "", None

    if auto_preview_df is not None and input_mode == "Upload match data for better prediction":
        st.markdown("### Auto-Generated Features Used for Prediction")
        st.dataframe(auto_preview_df, use_container_width=True)

    prediction_triggered = st.button(
        "Predict Match Outcome",
        use_container_width=True,
        key="predict_button",
    )

    if prediction_triggered:
        try:
            validate_teams(home_team, away_team)

            if not user_inputs:
                raise ValueError(
                    "Prediction inputs are not ready. Please complete the manual fields or upload a valid match data file."
                )

            results = get_prediction_results(user_inputs)

            st.session_state["latest_prediction_results"] = results
            st.session_state["latest_home_team"] = home_team
            st.session_state["latest_away_team"] = away_team
            st.session_state["latest_prediction_ready"] = True
            st.session_state["latest_auto_features_df"] = auto_preview_df

            st.session_state["latest_shap_explanation"] = None
            st.session_state["latest_shap_error"] = None

            try:
                predicted_label_for_shap = results.iloc[0]["prediction"]
                shap_results = get_shap_results(
                    user_inputs=user_inputs,
                    target_prediction_label=predicted_label_for_shap,
                )
                st.session_state["latest_shap_explanation"] = shap_results
            except Exception as shap_error:
                st.session_state["latest_shap_error"] = str(shap_error)

        except Exception as prediction_error:
            st.session_state["latest_prediction_ready"] = False
            st.session_state["latest_prediction_results"] = None
            st.session_state["latest_shap_explanation"] = None
            st.session_state["latest_shap_error"] = None
            st.session_state["latest_auto_features_df"] = None
            st.error(f"Prediction failed: {prediction_error}")

    if st.session_state.get("latest_prediction_ready", False):
        results = st.session_state.get("latest_prediction_results")
        saved_home_team = st.session_state.get("latest_home_team", "")
        saved_away_team = st.session_state.get("latest_away_team", "")
        shap_results = st.session_state.get("latest_shap_explanation")
        shap_error = st.session_state.get("latest_shap_error")
        saved_auto_features_df = st.session_state.get("latest_auto_features_df")

        if results is not None:
            render_prediction_output(results, saved_home_team, saved_away_team)

            if saved_auto_features_df is not None:
                st.markdown("### Stored Auto-Generated Features")
                st.dataframe(saved_auto_features_df, use_container_width=True)

            if shap_error:
                st.warning(
                    f"Prediction succeeded, but explainability could not be generated: {shap_error}"
                )

            if shap_results is not None and results is not None:
                render_explainability_section(shap_results, results)

    render_footer_navigation(show_previous=False, show_next=True)

elif st.session_state["current_page"] == "Analyse Bookmaker Bias":
    if not st.session_state.get("latest_prediction_ready", False):
        st.info("Please complete a prediction on the Match Prediction page first.")
    else:
        home_team = st.session_state.get("latest_home_team", "")
        away_team = st.session_state.get("latest_away_team", "")
        results = st.session_state.get("latest_prediction_results")

        st.subheader("Fixture")
        c1, c2 = st.columns(2)
        c1.text_input("Home Team", value=home_team, disabled=True, key="bias_home_team_display")
        c2.text_input("Away Team", value=away_team, disabled=True, key="bias_away_team_display")

        st.subheader("Insert Bookmaker Odds")
        b1, b2, b3 = st.columns(3)
        bookmaker_home_odds = b1.number_input("Bookmaker Home Win Odds", min_value=1.01, value=2.10, step=0.01, key="bias_home_odds")
        bookmaker_draw_odds = b2.number_input("Bookmaker Draw Odds", min_value=1.01, value=3.40, step=0.01, key="bias_draw_odds")
        bookmaker_away_odds = b3.number_input("Bookmaker Away Win Odds", min_value=1.01, value=3.60, step=0.01, key="bias_away_odds")

        if st.button("Submit Bookmaker Analysis", use_container_width=True, key="submit_bias_analysis"):
            try:
                row = results.iloc[0]

                model_prob_h = float(row.get("prob_H", 0))
                model_prob_d = float(row.get("prob_D", 0))
                model_prob_a = float(row.get("prob_A", 0))

                model_odds_h = fair_odds_from_probability(model_prob_h)
                model_odds_d = fair_odds_from_probability(model_prob_d)
                model_odds_a = fair_odds_from_probability(model_prob_a)

                bookmaker_prob_h = bookmaker_implied_probability(bookmaker_home_odds)
                bookmaker_prob_d = bookmaker_implied_probability(bookmaker_draw_odds)
                bookmaker_prob_a = bookmaker_implied_probability(bookmaker_away_odds)

                overround = bookmaker_prob_h + bookmaker_prob_d + bookmaker_prob_a - 1.0

                total_implied = bookmaker_prob_h + bookmaker_prob_d + bookmaker_prob_a
                norm_bookmaker_prob_h = bookmaker_prob_h / total_implied
                norm_bookmaker_prob_d = bookmaker_prob_d / total_implied
                norm_bookmaker_prob_a = bookmaker_prob_a / total_implied

                gap_h = model_prob_h - norm_bookmaker_prob_h
                gap_d = model_prob_d - norm_bookmaker_prob_d
                gap_a = model_prob_a - norm_bookmaker_prob_a

                comparison_df = pd.DataFrame(
                    {
                        "Outcome": ["Home", "Draw", "Away"],
                        "Model Probability": [model_prob_h, model_prob_d, model_prob_a],
                        "Model Fair Odds": [model_odds_h, model_odds_d, model_odds_a],
                        "Bookmaker Odds": [bookmaker_home_odds, bookmaker_draw_odds, bookmaker_away_odds],
                        "Bookmaker Implied Probability": [bookmaker_prob_h, bookmaker_prob_d, bookmaker_prob_a],
                        "Bookmaker Normalized Probability": [norm_bookmaker_prob_h, norm_bookmaker_prob_d, norm_bookmaker_prob_a],
                        "Probability Gap (Model - Bookmaker)": [gap_h, gap_d, gap_a],
                        "Short Explanation": [
                            explain_probability_gap(gap_h),
                            explain_probability_gap(gap_d),
                            explain_probability_gap(gap_a),
                        ],
                    }
                )

                st.success(f"Bookmaker bias analysis for {home_team} vs {away_team}")

                m1, m2, m3 = st.columns(3)
                m1.metric("Predicted Outcome", normalize_prediction_label(row["prediction"]))
                m2.metric("Bookmaker Overround", f"{overround:.2%}")
                m3.metric(
                    "Largest Model Edge",
                    comparison_df.loc[
                        comparison_df["Probability Gap (Model - Bookmaker)"].idxmax(),
                        "Outcome",
                    ],
                )

                st.subheader("Market Interpretation")
                st.info(explain_overround(overround))

                st.subheader("Model vs Bookmaker Comparison")
                st.dataframe(comparison_df, use_container_width=True)

            except Exception as e:
                st.error(f"Bias analysis failed: {e}")

        st.markdown("---")
        st.subheader("Bonus: 2-Outcome Overround Calculator")
        st.caption("Useful for markets such as Over/Under, Yes/No, or any two-way bookmaker market.")

        t1, t2 = st.columns(2)
        outcome_1_name = t1.text_input("Outcome 1 Name", value="Over 2.5 Goals", key="two_way_outcome_1_name")
        outcome_2_name = t2.text_input("Outcome 2 Name", value="Under 2.5 Goals", key="two_way_outcome_2_name")

        o1, o2 = st.columns(2)
        outcome_1_odds = o1.number_input(f"{outcome_1_name} Odds", min_value=1.01, value=1.85, step=0.01, key="two_way_outcome_1_odds")
        outcome_2_odds = o2.number_input(f"{outcome_2_name} Odds", min_value=1.01, value=1.95, step=0.01, key="two_way_outcome_2_odds")

        if st.button("Calculate 2-Outcome Overround", use_container_width=True, key="two_way_overround_button"):
            try:
                prob_1 = bookmaker_implied_probability(outcome_1_odds)
                prob_2 = bookmaker_implied_probability(outcome_2_odds)

                two_way_overround = prob_1 + prob_2 - 1.0

                total_two_way_prob = prob_1 + prob_2
                norm_prob_1 = prob_1 / total_two_way_prob
                norm_prob_2 = prob_2 / total_two_way_prob

                two_way_df = pd.DataFrame(
                    {
                        "Outcome": [outcome_1_name, outcome_2_name],
                        "Bookmaker Odds": [outcome_1_odds, outcome_2_odds],
                        "Implied Probability": [prob_1, prob_2],
                        "Normalized Probability": [norm_prob_1, norm_prob_2],
                    }
                )

                st.success("2-outcome overround analysis completed.")

                x1, x2 = st.columns(2)
                x1.metric("2-Outcome Overround", f"{two_way_overround:.2%}")
                x2.metric("Bookmaker Edge", f"{two_way_overround:.2%}")

                st.dataframe(two_way_df, use_container_width=True)
                st.info(explain_two_way_overround(two_way_overround))

            except Exception as e:
                st.error(f"2-outcome analysis failed: {e}")

    render_footer_navigation(show_previous=True, show_next=False)