# project/__init__.py
import pandas as pd
import numpy as np

# ----  put your long helper functions here  ----
# (copy/paste get_last_matches, aggregate_team_form, generate_team_form_table, etc.)

@st.cache_data(show_spinner=False)  # Streamlit cache
def load_match_data() -> pd.DataFrame:
    season_urls = [
        "https://fbref.com/en/comps/9/2024-2025/schedule/2024-2025-Premier-League-Scores-and-Fixtures",
        "https://fbref.com/en/comps/9/2023-2024/schedule/2023-2024-Premier-League-Scores-and-Fixtures",
        "https://fbref.com/en/comps/9/2022-2023/schedule/2022-2023-Premier-League-Scores-and-Fixtures",
        "https://fbref.com/en/comps/9/2021-2022/schedule/2021-2022-Premier-League-Scores-and-Fixtures",
    ]
    df_list = []
    for url in season_urls:
        df = pd.read_html(url)[0]
        df["Season"] = url.split("/")[5]
        df_list.append(df)
    matchday_data = pd.concat(df_list, ignore_index=True)
    matchday_data = matchday_data.drop(columns=["Notes","Venue","Match Report","Day"], errors="ignore")
    return matchday_data

def predict_fixture(season:str, gw:int):
    # load, preprocess, run your trained models
    matchday_data = load_match_data()
    # --- call your existing generate_prediction_table + model predict code ---
    return preds_df        # DataFrame with columns Home, Away, home_pred, away_pred
