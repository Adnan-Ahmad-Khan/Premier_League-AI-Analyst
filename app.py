import streamlit as st
import pandas as pd
import requests
import json
import time

# --- Global Constants / Secrets ---

# IMPORTANT:
# On Streamlit Cloud, you will set this in "Secrets" as GOOGLE_API_KEY.
# Locally, you can create .streamlit/secrets.toml with:
#   GOOGLE_API_KEY = "your-real-key"
try:
    API_KEY = st.secrets["GOOGLE_API_KEY"]
except Exception:
    API_KEY = ""


GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"
MODEL_NAME = "gemini-2.5-flash-preview-09-2025"

# --- Core Functions: Mock Data and ML Simulation ---


def load_historical_data():
    all_pl_teams = [
        "Arsenal",
        "Man Utd",
        "Liverpool",
        "Chelsea",
        "Tottenham",
        "Man City",
        "Everton",
        "Newcastle Utd",
        "Aston Villa",
        "West Ham",
        "Southampton",
        "Leicester City",
        "Leeds United",
        "Blackburn Rovers",
        "Sunderland",
        "Middlesbrough",
        "Bolton Wanderers",
        "Fulham",
        "Coventry City",
        "Queens Park Rangers",
        "Sheffield Wednesday",
        "Crystal Palace",
        "Wimbledon",
        "Norwich City",
        "Ipswich Town",
        "Nottingham Forest",
        "Birmingham City",
        "Derby County",
        "West Brom",
        "Stoke City",
        "Wigan Athletic",
        "Portsmouth",
        "Burnley",
        "Wolverhampton Wanderers",
        "Reading",
        "Hull City",
        "Charlton Athletic",
        "Watford",
        "Huddersfield Town",
        "Brighton",
        "Bournemouth",
        "Cardiff City",
        "Sheffield United",
        "Bradford City",
        "Swindon Town",
        "Oldham Athletic",
        "Barnsley",
        "Blackpool",
        "Brentford",
    ]

    data = {
        "HomeTeam": all_pl_teams,
        "AwayTeam": all_pl_teams,
        "HomeGoals": [0] * len(all_pl_teams),
        "AwayGoals": [0] * len(all_pl_teams),
        "Result": ["D"] * len(all_pl_teams),
    }
    return pd.DataFrame(data)


def predict_match(home_team, away_team):
    teams = sorted([home_team, away_team])

    if teams == ["Arsenal", "Man City"]:
        return "Draw (1-1)", 0.35, 0.40, 0.25
    if teams == ["Liverpool", "Man Utd"]:
        return "Home Win (2-0)", 0.60, 0.25, 0.15
    if teams == ["Chelsea", "Tottenham"]:
        return "Draw (2-2)", 0.45, 0.35, 0.20

    return "Home Win (2-1)", 0.40, 0.30, 0.30


# --- Core Function: LLM API Interaction (with Grounding) ---


def gemini_query(prompt, system_instruction, api_key, max_retries=3):
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "tools": [{"google_search": {}}],
        "systemInstruction": {"parts": [{"text": system_instruction}]},
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{GEMINI_API_URL}?key={api_key}",
                headers=headers,
                data=json.dumps(payload),
            )
            response.raise_for_status()
            result = response.json()
            candidate = result.get("candidates", [None])[0]

            if not candidate:
                return (
                    "LLM Error: Could not generate content. Check the prompt or API key.",
                    [],
                )

            text = (
                candidate.get("content", {})
                .get("parts", [{}])[0]
                .get("text", "No response text.")
            )

            sources = []
            grounding_metadata = candidate.get("groundingMetadata", {})
            if grounding_metadata and grounding_metadata.get("groundingAttributions"):
                sources = grounding_metadata["groundingAttributions"]

            return text, sources

        except requests.exceptions.HTTPError:
            if response.status_code == 429 and attempt < max_retries - 1:
                time.sleep(2**attempt)
            else:
                return f"LLM Error (HTTP {response.status_code}): {response.text}", []
        except requests.exceptions.RequestException as e:
            return f"LLM Error (Request Failed): {e}", []

    return "LLM Error: Failed after multiple retries.", []


def get_historical_facts(team1, team2, api_key):
    """Generates a brief historical summary and famous facts about the two teams."""

    system_prompt = (
        "You are an expert Premier League historian. Your task is to provide concise, "
        "engaging, and positive historical facts about the two specified teams. "
        "Focus on their most famous achievements (trophies, legendary players, "
        "iconic matches). Keep the response under 150 words."
    )

    prompt = (
        f"Provide a brief historical summary and two interesting facts about the rivalry "
        f"and history between {team1} and {team2}. Focus on iconic moments or players."
    )

    return gemini_query(prompt, system_prompt, api_key)


# --- Streamlit UI Components ---


def display_prediction_section(api_key):
    st.header("⚽ Match Prediction Model (ML)")
    st.markdown(
        "Select two teams to see the model's predicted outcome. "
        "*(Note: The prediction logic is simulated using simplified rules.)*"
    )
    df_history = load_historical_data()
    teams = sorted(df_history["HomeTeam"].unique().tolist())

    pl_stadiums = [
        "Anfield",
        "Old Trafford",
        "Emirates Stadium",
        "Stamford Bridge",
        "Tottenham Hotspur Stadium",
        "Etihad Stadium",
        "St. James Park",
        "Villa Park",
        "London Stadium",
        "Goodison Park",
        "Molineux",
        "The City Ground",
        "Craven Cottage",
        "Vitality Stadium",
        "Amex Stadium",
    ]

    col_stadium, col_empty = st.columns([2, 1])

    with col_stadium:
        default_index_stadium = (
            pl_stadiums.index("Etihad Stadium")
            if "Etihad Stadium" in pl_stadiums
            else 0
        )
        selected_stadium = st.selectbox(
            "Select Venue", pl_stadiums, index=default_index_stadium, key="stadium_select"
        )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        default_index_home = teams.index(
            "Man City") if "Man City" in teams else 0
        home_team = st.selectbox(
            "Home Team", teams, index=default_index_home, key="home_team_select"
        )
    with col2:
        default_index_away = teams.index(
            "Arsenal") if "Arsenal" in teams else 1
        away_team = st.selectbox(
            "Away Team", teams, index=default_index_away, key="away_team_select"
        )

    st.markdown("---")

    if home_team == away_team:
        st.warning("Please select two different teams for the match.")
        return

    if st.button(
        f"Predict {home_team} vs {away_team} at {selected_stadium}", type="primary"
    ):
        with st.spinner("Calculating historical form and running prediction model..."):
            time.sleep(1)
            prediction, p_h, p_d, p_a = predict_match(home_team, away_team)

            st.subheader(
                f"🏟️ {home_team} vs {away_team} at {selected_stadium}")
            st.markdown("---")

            st.subheader("📊 Predicted Match Outcome")
            col_pred, col_prob = st.columns([1.5, 2])

            with col_pred:
                st.success(f"**Prediction:** {prediction}")

            with col_prob:
                home_win_label = f"{home_team} Win"
                away_win_label = f"{away_team} Win"

                prob_data = (
                    pd.DataFrame(
                        {
                            "Outcome": [home_win_label, "Draw", away_win_label],
                            "Probability": [p_h, p_d, p_a],
                        }
                    )
                    .set_index("Outcome")
                )
                st.bar_chart(prob_data, height=200)

        st.markdown("---")

        st.subheader("📜 Historical Rivalry Facts")
        with st.spinner(
            f"Fetching historical facts about {home_team} and {away_team} from the LLM..."
        ):
            fact_text, fact_sources = get_historical_facts(
                home_team, away_team, api_key
            )

            st.info(fact_text)

            if fact_sources:
                st.markdown(
                    "**Source(s) for facts:**",
                    help="These sources were used to ground the historical information.",
                )
                unique_sources = {}
                for s in fact_sources:
                    uri = s.get("uri")
                    if uri and uri not in unique_sources:
                        title = s.get("title", "Untitled Source")
                        unique_sources[uri] = title

                for uri, title in unique_sources.items():
                    st.markdown(f"- [{title}]({uri})", unsafe_allow_html=True)


def display_llm_chat_section(api_key):
    st.header("💬 Premier League Insight Bot (LLM)")
    st.markdown(
        "Ask the bot anything about the Premier League, current news, history, or player stats."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    system_prompt = (
        "You are a friendly, concise, and expert football analyst focusing on the English Premier League (EPL). "
        "Answer questions accurately. If asked for current/real-time information (like recent scores, standings, "
        "or transfer news), always use the provided Google Search tool results to give a grounded answer and "
        "mention the source briefly. Keep your responses professional and focused on the Premier League."
    )

    for message in st.session_state.messages:
        avatar = "🤖" if message["role"] == "assistant" else "👤"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about a team, player, or recent match..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking... (Searching the web for current data)"):
                response_text, sources = gemini_query(
                    prompt, system_prompt, api_key)
                formatted_response = response_text

                if sources:
                    formatted_response += "\n\n---\n\n**Sources:**\n"
                    unique_sources = {}
                    for s in sources:
                        uri = s.get("uri")
                        if not uri:
                            continue
                        title = s.get("title", "Untitled Source")
                        if uri not in unique_sources:
                            unique_sources[uri] = title

                    for i, (uri, title) in enumerate(unique_sources.items()):
                        formatted_response += f"{i+1}. [{title}]({uri})\n"

                st.markdown(formatted_response)

            st.session_state.messages.append(
                {"role": "assistant", "content": formatted_response}
            )


# --- Main App Function ---


def main():
    st.set_page_config(
        page_title="Premier League AI Bot",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if not API_KEY:
        st.warning(
            "GOOGLE_API_KEY not found in secrets. "
            "Set it in .streamlit/secrets.toml (locally) or in Streamlit Cloud > Settings > Secrets."
        )

    gradient_css = """
    <style>
    .stApp {
        background: linear-gradient(135deg, #1f0547 0%, #3a0e69 40%, #5d1797 100%);
        background-attachment: fixed;
    }
    h1, h2, h3 {
        color: #ffffff;
    }
    .stApp, .stMarkdown {
        color: #f0f0f0;
    }
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 10px;
    }
    .stButton>button {
        background-color: #f97316;
        color: white;
        border-radius: 8px;
        border: none;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #ea580c;
        transform: scale(1.02);
    }
    </style>
    """
    st.markdown(gradient_css, unsafe_allow_html=True)

    LOGO_PATH = "pl.png"

    logo_col, title_col = st.columns([1, 6])

    with logo_col:
        try:
            st.image(LOGO_PATH, width=80)
        except FileNotFoundError:
            st.warning("Logo image not found (expected 'pl.png').")

    with title_col:
        st.title("🏆 Premier League AI Analyst")
        st.markdown(
            "A combined tool for football analysis: an ML model for match predictions and "
            "an LLM chatbot for real-time insights and data-grounded answers."
        )

    st.divider()
    col_ml, col_llm = st.columns([1, 1])

    with col_ml:
        display_prediction_section(API_KEY)

    with col_llm:
        display_llm_chat_section(API_KEY)


if __name__ == "__main__":
    main()
