import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import pandas as pd
import streamlit as st

from src.inference.input_schema import build_feature_ready_row
from src.inference.predict import predict_from_features


st.set_page_config(
    page_title="Jupiler Pro League Match Predictor",
    page_icon="⚽",
    layout="wide",
)

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


def fair_odds_from_probability(prob: float) -> float:
    if prob <= 0:
        return float("inf")
    return 1.0 / prob


def bookmaker_implied_probability(odds: float) -> float:
    if odds <= 0:
        return 0.0
    return 1.0 / odds


def explain_bias(probability_gap: float) -> str:
    if probability_gap > 0.03:
        return "The model rates this outcome higher than the bookmaker market. This may indicate possible value."
    if probability_gap < -0.03:
        return "The bookmaker market rates this outcome higher than the model. The outcome may be overpriced by the market."
    return "The model and bookmaker are relatively aligned for this outcome."


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
        placeholder="Type team name",
        key=f"{key_prefix}_{prefix}_team_text",
    ).strip()


def stat_input_block(prefix: str, title: str, key_prefix: str) -> dict:
    st.subheader(title)
    c1, c2 = st.columns(2)

    values = {}

    values[f"{prefix}_goals_scored_last5"] = c1.number_input(
        "Goals registered", min_value=0.0, value=1.0, step=0.1, key=f"{key_prefix}_{prefix}_goals_registered"
    )
    values[f"{prefix}_goals_conceded_last5"] = c2.number_input(
        "Goals conceded", min_value=0.0, value=1.0, step=0.1, key=f"{key_prefix}_{prefix}_goals_conceded"
    )
    values[f"{prefix}_shots_for_last5"] = c1.number_input(
        "Shots registered", min_value=0.0, value=4.0, step=0.1, key=f"{key_prefix}_{prefix}_shots_registered"
    )
    values[f"{prefix}_shots_against_last5"] = c2.number_input(
        "Shots conceded", min_value=0.0, value=4.0, step=0.1, key=f"{key_prefix}_{prefix}_shots_conceded"
    )
    values[f"{prefix}_shots_on_target_for_last5"] = c1.number_input(
        "Shots on target registered", min_value=0.0, value=2.0, step=0.1, key=f"{key_prefix}_{prefix}_shots_ot_registered"
    )
    values[f"{prefix}_shots_on_target_against_last5"] = c2.number_input(
        "Shots on target conceded", min_value=0.0, value=2.0, step=0.1, key=f"{key_prefix}_{prefix}_shots_ot_conceded"
    )
    values[f"{prefix}_fouls_for_last5"] = c1.number_input(
        "Fouls registered", min_value=0.0, value=10.0, step=0.1, key=f"{key_prefix}_{prefix}_fouls_registered"
    )
    values[f"{prefix}_fouls_against_last5"] = c2.number_input(
        "Fouls conceded", min_value=0.0, value=10.0, step=0.1, key=f"{key_prefix}_{prefix}_fouls_conceded"
    )
    values[f"{prefix}_corners_for_last5"] = c1.number_input(
        "Corners registered", min_value=0.0, value=5.0, step=0.1, key=f"{key_prefix}_{prefix}_corners_registered"
    )
    values[f"{prefix}_corners_against_last5"] = c2.number_input(
        "Corners conceded", min_value=0.0, value=5.0, step=0.1, key=f"{key_prefix}_{prefix}_corners_conceded"
    )

    return values


def collect_model_inputs(key_prefix: str) -> tuple[dict, str, str]:
    st.subheader("Match Context")
    c1, c2 = st.columns(2)

    with c1:
        home_team = team_input_block("home", "Home Team", key_prefix)
    with c2:
        away_team = team_input_block("away", "Away Team", key_prefix)

    c3, c4, c5, c6 = st.columns(4)
    matchday = c3.number_input("Matchday", min_value=1, max_value=40, value=10, step=1, key=f"{key_prefix}_matchday")
    home_pre_position = c4.number_input("Home pre-match position", min_value=1, max_value=20, value=6, step=1, key=f"{key_prefix}_home_pre_position")
    away_pre_position = c5.number_input("Away pre-match position", min_value=1, max_value=20, value=10, step=1, key=f"{key_prefix}_away_pre_position")
    home_form = c6.number_input("Home form (0-1)", min_value=0.0, max_value=1.0, value=0.50, step=0.01, key=f"{key_prefix}_home_form")

    away_form = st.number_input("Away form (0-1)", min_value=0.0, max_value=1.0, value=0.50, step=0.01, key=f"{key_prefix}_away_form")

    st.markdown("---")

    left, right = st.columns(2)
    with left:
        home_inputs = stat_input_block("home", "Home Team — Last 5 Matches", key_prefix=key_prefix)
    with right:
        away_inputs = stat_input_block("away", "Away Team — Last 5 Matches", key_prefix=key_prefix)

    user_inputs = {
        "matchday": float(matchday),
        "hour": float(DEFAULT_HOUR),
        "home_pre_position": float(home_pre_position),
        "away_pre_position": float(away_pre_position),
        "home_form": float(home_form),
        "away_form": float(away_form),
        "home_advantage": float(DEFAULT_HOME_ADVANTAGE),
        **home_inputs,
        **away_inputs,
        "home_yellow_cards_for_last5": float(DEFAULT_HOME_YELLOW_REGISTERED),
        "home_yellow_cards_against_last5": float(DEFAULT_HOME_YELLOW_CONCEDED),
        "home_red_cards_for_last5": float(DEFAULT_HOME_RED_REGISTERED),
        "home_red_cards_against_last5": float(DEFAULT_HOME_RED_CONCEDED),
        "away_yellow_cards_for_last5": float(DEFAULT_AWAY_YELLOW_REGISTERED),
        "away_yellow_cards_against_last5": float(DEFAULT_AWAY_YELLOW_CONCEDED),
        "away_red_cards_for_last5": float(DEFAULT_AWAY_RED_REGISTERED),
        "away_red_cards_against_last5": float(DEFAULT_AWAY_RED_CONCEDED),
    }

    return user_inputs, home_team, away_team


def get_prediction_results(user_inputs: dict) -> pd.DataFrame:
    feature_df = build_feature_ready_row(user_inputs)
    return predict_from_features(feature_df)


def validate_teams(home_team: str, away_team: str) -> None:
    if not home_team:
        raise ValueError("Please provide a Home Team.")
    if not away_team:
        raise ValueError("Please provide an Away Team.")
    if home_team == away_team:
        raise ValueError("Home Team and Away Team must be different.")


st.title("⚽ Jupiler Pro League Match Prediction System")
st.caption("Belgian Jupiler Pro League focused. Enter match features on page 1, then analyse bookmaker bias on page 2.")

tab1, tab2 = st.tabs(["Match Prediction", "Analyse Bookmaker Bias"])

with tab1:
    user_inputs, home_team, away_team = collect_model_inputs("prediction_tab")

    if st.button("Predict Match Outcome", use_container_width=True, key="predict_button"):
        try:
            validate_teams(home_team, away_team)

            results = get_prediction_results(user_inputs)
            row = results.iloc[0]

            st.session_state["latest_prediction_results"] = results
            st.session_state["latest_home_team"] = home_team
            st.session_state["latest_away_team"] = away_team
            st.session_state["latest_prediction_ready"] = True

            st.success(f"Predicted outcome for {home_team} vs {away_team}: {row['prediction']}")

            p1, p2, p3 = st.columns(3)
            p1.metric("Home Win Probability", f"{row.get('prob_H', 0):.2%}")
            p2.metric("Draw Probability", f"{row.get('prob_D', 0):.2%}")
            p3.metric("Away Win Probability", f"{row.get('prob_A', 0):.2%}")

            display_df = pd.DataFrame(
                {
                    "Home Team": [home_team],
                    "Away Team": [away_team],
                    "Prediction": [row["prediction"]],
                    "Prob_H": [row.get("prob_H", 0)],
                    "Prob_D": [row.get("prob_D", 0)],
                    "Prob_A": [row.get("prob_A", 0)],
                }
            )
            st.dataframe(display_df, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

with tab2:
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
        bookmaker_home_odds = b1.number_input(
            "Bookmaker Home Win Odds", min_value=1.01, value=2.10, step=0.01, key="bias_home_odds"
        )
        bookmaker_draw_odds = b2.number_input(
            "Bookmaker Draw Odds", min_value=1.01, value=3.40, step=0.01, key="bias_draw_odds"
        )
        bookmaker_away_odds = b3.number_input(
            "Bookmaker Away Win Odds", min_value=1.01, value=3.60, step=0.01, key="bias_away_odds"
        )

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
                        "Short Explanation": [explain_bias(gap_h), explain_bias(gap_d), explain_bias(gap_a)],
                    }
                )

                st.success(f"Bookmaker bias analysis for {home_team} vs {away_team}")

                m1, m2, m3 = st.columns(3)
                m1.metric("Predicted Outcome", row["prediction"])
                m2.metric("Bookmaker Overround", f"{overround:.2%}")
                m3.metric(
                    "Largest Model Edge",
                    comparison_df.loc[
                        comparison_df["Probability Gap (Model - Bookmaker)"].idxmax(),
                        "Outcome",
                    ],
                )

                st.subheader("Model vs Bookmaker Comparison")
                st.dataframe(comparison_df, use_container_width=True)

                st.info(
                    "Interpretation: bookmaker edge is the overround, which is the total implied probability above 100%. "
                    "A positive probability gap means the model rates that outcome higher than the bookmaker's normalized market estimate."
                )

            except Exception as e:
                st.error(f"Bias analysis failed: {e}")