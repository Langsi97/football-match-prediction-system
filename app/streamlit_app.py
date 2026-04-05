import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.explainability.shap_explainer import compute_shap_explanation
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


def go_to_page(page_name: str) -> None:
    st.session_state["current_page"] = page_name


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
    feature_df = build_feature_ready_row(user_inputs)
    return predict_from_features(feature_df)


def get_shap_results(user_inputs: dict) -> dict:
    feature_df = build_feature_ready_row(user_inputs)
    return compute_shap_explanation(feature_df=feature_df, top_n=5)


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
        placeholder="Type team name",
        key=f"{key_prefix}_{prefix}_team_text",
    ).strip()


def stat_input_block(prefix: str, title: str, key_prefix: str) -> dict:
    st.subheader(title)
    c1, c2 = st.columns(2)

    values = {}

    values[f"{prefix}_goals_scored_last5"] = c1.number_input(
        "Goals registered",
        min_value=0.0,
        value=1.0,
        step=0.1,
        key=f"{key_prefix}_{prefix}_goals_registered",
        help=(
            "Average goals scored by the team across its last 5 matches. "
            "You can get this directly from recent match results: add goals scored in the last 5 matches and divide by 5."
        ),
    )
    values[f"{prefix}_goals_conceded_last5"] = c2.number_input(
        "Goals conceded",
        min_value=0.0,
        value=1.0,
        step=0.1,
        key=f"{key_prefix}_{prefix}_goals_conceded",
        help=(
            "Average goals conceded by the team across its last 5 matches. "
            "Take goals allowed in the last 5 matches, sum them, then divide by 5."
        ),
    )

    values[f"{prefix}_shots_for_last5"] = c1.number_input(
        "Shots registered",
        min_value=0.0,
        value=4.0,
        step=0.1,
        key=f"{key_prefix}_{prefix}_shots_registered",
        help=(
            "Average total shots attempted by the team over its last 5 matches. "
            "This is often available directly on live-score or match-stat sites. Otherwise, add the last 5 values and divide by 5."
        ),
    )
    values[f"{prefix}_shots_against_last5"] = c2.number_input(
        "Shots conceded",
        min_value=0.0,
        value=4.0,
        step=0.1,
        key=f"{key_prefix}_{prefix}_shots_conceded",
        help=(
            "Average total shots allowed by the team over its last 5 matches. "
            "Use opponent shot totals from the last 5 matches, sum them, then divide by 5."
        ),
    )

    values[f"{prefix}_shots_on_target_for_last5"] = c1.number_input(
        "Shots on target registered",
        min_value=0.0,
        value=2.0,
        step=0.1,
        key=f"{key_prefix}_{prefix}_shots_ot_registered",
        help=(
            "Average shots on target made by the team across its last 5 matches. "
            "Usually available directly from match-stat sites. Otherwise, sum the last 5 shots-on-target values and divide by 5."
        ),
    )
    values[f"{prefix}_shots_on_target_against_last5"] = c2.number_input(
        "Shots on target conceded",
        min_value=0.0,
        value=2.0,
        step=0.1,
        key=f"{key_prefix}_{prefix}_shots_ot_conceded",
        help=(
            "Average shots on target allowed by the team across its last 5 matches. "
            "Use opponent shots on target from the last 5 matches, sum them, then divide by 5."
        ),
    )

    values[f"{prefix}_fouls_for_last5"] = c1.number_input(
        "Fouls registered",
        min_value=0.0,
        value=10.0,
        step=0.1,
        key=f"{key_prefix}_{prefix}_fouls_registered",
        help=(
            "Average fouls committed by the team over its last 5 matches. "
            "This is commonly shown on detailed match-stat pages. Sum the last 5 foul totals and divide by 5."
        ),
    )
    values[f"{prefix}_fouls_against_last5"] = c2.number_input(
        "Fouls conceded",
        min_value=0.0,
        value=10.0,
        step=0.1,
        key=f"{key_prefix}_{prefix}_fouls_conceded",
        help=(
            "Average fouls won by the team, or equivalently fouls committed by opponents, over the last 5 matches. "
            "Sum the last 5 values and divide by 5."
        ),
    )

    values[f"{prefix}_corners_for_last5"] = c1.number_input(
        "Corners registered",
        min_value=0.0,
        value=5.0,
        step=0.1,
        key=f"{key_prefix}_{prefix}_corners_registered",
        help=(
            "Average corners earned by the team across its last 5 matches. "
            "Usually available directly on live-score and advanced match-stat websites. Sum the last 5 values and divide by 5."
        ),
    )
    values[f"{prefix}_corners_against_last5"] = c2.number_input(
        "Corners conceded",
        min_value=0.0,
        value=5.0,
        step=0.1,
        key=f"{key_prefix}_{prefix}_corners_conceded",
        help=(
            "Average corners allowed by the team across its last 5 matches. "
            "Use opponent corners from the last 5 matches, sum them, then divide by 5."
        ),
    )

    return values


def collect_model_inputs(key_prefix: str) -> tuple[dict, str, str]:
    st.subheader("Match Context")

    c1, c2 = st.columns(2)
    with c1:
        home_team = team_input_block("home", "Home Team", key_prefix)
    with c2:
        away_team = team_input_block("away", "Away Team", key_prefix)

    matchday = st.number_input(
        "Matchday",
        min_value=1,
        max_value=40,
        value=10,
        step=1,
        key=f"{key_prefix}_matchday",
        help="The league round number for the fixture. This is usually directly available from the fixture list or competition schedule.",
    )

    left, right = st.columns(2)

    with left:
        st.markdown("### Home Team Context")
        home_pre_position = st.number_input(
            "Home pre-match position",
            min_value=1,
            max_value=20,
            value=6,
            step=1,
            key=f"{key_prefix}_home_pre_position",
            help="The team's league position before kickoff. You can usually get this directly from the live table or standings before the match starts.",
        )
        home_form = st.number_input(
            "Home form (0–1)",
            min_value=0.0,
            max_value=1.0,
            value=0.50,
            step=0.01,
            key=f"{key_prefix}_home_form",
            help=(
                "Team form = total points from the last 5 matches divided by 15. "
                "Win = 3 points, draw = 1 point, loss = 0 points. "
                "Example: 3 wins, 1 draw, 1 loss = 10 points, so form = 10 / 15 = 0.67."
            ),
        )

    with right:
        st.markdown("### Away Team Context")
        away_pre_position = st.number_input(
            "Away pre-match position",
            min_value=1,
            max_value=20,
            value=10,
            step=1,
            key=f"{key_prefix}_away_pre_position",
            help="The team's league position before kickoff. You can usually get this directly from the live table or standings before the match starts.",
        )
        away_form = st.number_input(
            "Away form (0–1)",
            min_value=0.0,
            max_value=1.0,
            value=0.50,
            step=0.01,
            key=f"{key_prefix}_away_form",
            help=(
                "Team form = total points from the last 5 matches divided by 15. "
                "Win = 3 points, draw = 1 point, loss = 0 points. "
                "Example: 2 wins, 2 draws, 1 loss = 8 points, so form = 8 / 15 = 0.53."
            ),
        )

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


def render_prediction_output(results: pd.DataFrame, home_team: str, away_team: str) -> None:
    row = results.iloc[0]

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


def render_explainability_section(shap_results: dict) -> None:
    st.markdown("---")
    st.subheader("Model Explainability Dashboard")
    st.caption(
        "This section explains which input features most strongly supported or opposed the model's predicted outcome."
    )

    predicted_class_label = shap_results["predicted_class_label"]
    contribution_df = shap_results["feature_contributions_df"]
    top_positive_df = shap_results["top_positive_df"]
    top_negative_df = shap_results["top_negative_df"]

    st.info(f"Explanation target class: {predicted_class_label}")

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
    user_inputs, home_team, away_team = collect_model_inputs("prediction_page")

    prediction_triggered = st.button(
        "Predict Match Outcome",
        use_container_width=True,
        key="predict_button",
    )

    if prediction_triggered:
        try:
            validate_teams(home_team, away_team)

            results = get_prediction_results(user_inputs)

            st.session_state["latest_prediction_results"] = results
            st.session_state["latest_home_team"] = home_team
            st.session_state["latest_away_team"] = away_team
            st.session_state["latest_prediction_ready"] = True

            st.session_state["latest_shap_explanation"] = None
            st.session_state["latest_shap_error"] = None

            try:
                shap_results = get_shap_results(user_inputs)
                st.session_state["latest_shap_explanation"] = shap_results
            except Exception as shap_error:
                st.session_state["latest_shap_error"] = str(shap_error)

        except Exception as prediction_error:
            st.session_state["latest_prediction_ready"] = False
            st.session_state["latest_prediction_results"] = None
            st.session_state["latest_shap_explanation"] = None
            st.session_state["latest_shap_error"] = None
            st.error(f"Prediction failed: {prediction_error}")

    if st.session_state.get("latest_prediction_ready", False):
        results = st.session_state.get("latest_prediction_results")
        saved_home_team = st.session_state.get("latest_home_team", "")
        saved_away_team = st.session_state.get("latest_away_team", "")
        shap_results = st.session_state.get("latest_shap_explanation")
        shap_error = st.session_state.get("latest_shap_error")

        if results is not None:
            render_prediction_output(results, saved_home_team, saved_away_team)

            if shap_error:
                st.warning(
                    f"Prediction succeeded, but explainability could not be generated: {shap_error}"
                )

            if shap_results is not None:
                render_explainability_section(shap_results)

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
        bookmaker_home_odds = b1.number_input(
            "Bookmaker Home Win Odds",
            min_value=1.01,
            value=2.10,
            step=0.01,
            key="bias_home_odds",
            help="Insert the decimal odds quoted by the bookmaker for a home win.",
        )
        bookmaker_draw_odds = b2.number_input(
            "Bookmaker Draw Odds",
            min_value=1.01,
            value=3.40,
            step=0.01,
            key="bias_draw_odds",
            help="Insert the decimal odds quoted by the bookmaker for a draw.",
        )
        bookmaker_away_odds = b3.number_input(
            "Bookmaker Away Win Odds",
            min_value=1.01,
            value=3.60,
            step=0.01,
            key="bias_away_odds",
            help="Insert the decimal odds quoted by the bookmaker for an away win.",
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
                        "Short Explanation": [
                            explain_probability_gap(gap_h),
                            explain_probability_gap(gap_d),
                            explain_probability_gap(gap_a),
                        ],
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
        outcome_1_name = t1.text_input(
            "Outcome 1 Name",
            value="Over 2.5 Goals",
            key="two_way_outcome_1_name",
            help="Example: Over 2.5 Goals",
        )
        outcome_2_name = t2.text_input(
            "Outcome 2 Name",
            value="Under 2.5 Goals",
            key="two_way_outcome_2_name",
            help="Example: Under 2.5 Goals",
        )

        o1, o2 = st.columns(2)
        outcome_1_odds = o1.number_input(
            f"{outcome_1_name} Odds",
            min_value=1.01,
            value=1.85,
            step=0.01,
            key="two_way_outcome_1_odds",
            help="Insert the decimal bookmaker odds for outcome 1.",
        )
        outcome_2_odds = o2.number_input(
            f"{outcome_2_name} Odds",
            min_value=1.01,
            value=1.95,
            step=0.01,
            key="two_way_outcome_2_odds",
            help="Insert the decimal bookmaker odds for outcome 2.",
        )

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