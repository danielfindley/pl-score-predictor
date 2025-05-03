# project/__init__.py
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
import pathlib
import time
from urllib.error import HTTPError

# -------------------------------------------------------------------
# 1. ──────────────────────────  DATA  ───────────────────────────────
# -------------------------------------------------------------------

CACHE_FILE = pathlib.Path(__file__).with_name("fbref_cache.pkl")

@st.cache_data(show_spinner="Fetching FBref…")
def load_match_data() -> pd.DataFrame:
    """Scrape FBref, with disk fallback if we hit a 429 or get 0 rows."""
    frames = []
    SEASON_URLS = [
        "https://fbref.com/en/comps/9/2024-2025/schedule/2024-2025-Premier-League-Scores-and-Fixtures",
        "https://fbref.com/en/comps/9/2023-2024/schedule/2023-2024-Premier-League-Scores-and-Fixtures",
        "https://fbref.com/en/comps/9/2022-2023/schedule/2022-2023-Premier-League-Scores-and-Fixtures",
        "https://fbref.com/en/comps/9/2021-2022/schedule/2021-2022-Premier-League-Scores-and-Fixtures",
    ]

    try:
        for url in SEASON_URLS:
            # polite pause to reduce rate‑limit risk
            time.sleep(0.8)
            df = pd.read_html(url)[0]
            df["Season"] = url.split("/")[6]          # 2024-2025 etc.
            frames.append(df)

        data = pd.concat(frames, ignore_index=True)

        # clean
        data = data[data["Wk"] != "Wk"]
        data["Wk"] = pd.to_numeric(data["Wk"], errors="coerce").dropna().astype(int)
        data = data.drop(columns=["Notes","Venue","Match Report","Day"], errors="ignore")

        # if scrape succeeded & isn’t empty, save to disk
        if not data.empty:
            CACHE_FILE.write_bytes(pd.to_pickle(data))
            return data.reset_index(drop=True)

        st.warning("Scrape returned 0 rows – using on‑disk cache.")
        raise ValueError("empty scrape")

    except (HTTPError, ValueError) as err:
        if CACHE_FILE.exists():
            st.warning(f"{err} – falling back to cached FBref ({CACHE_FILE.stat().st_mtime:%Y‑%m‑%d}).")
            return pd.read_pickle(CACHE_FILE)
        else:
            st.error("FBref unreachable and no local cache available.")
            return pd.DataFrame()


# -------------------------------------------------------------------
# 2. ────────────  HELPER FUNCTIONS (unchanged from notebook) ────────
# -------------------------------------------------------------------
def get_last_matches(data, team, n):
    """Return the last n matches for a given team."""
    return data[(data["Home"] == team) | (data["Away"] == team)].tail(n)

def aggregate_team_form(data, team):
    if data.empty:
        return pd.DataFrame(
            {"xGfor":[0],"xGagg":[0],"gf":[0],"ga":[0],
             "xG_dif":[0],"GD":[0],"teams_against":[""]},
            index=[team])
    home_games = data[data["Home"] == team]
    away_games = data[data["Away"] == team]

    total_xG_for     = home_games["xG"].sum()  + away_games["xG.1"].sum()
    total_xG_against = home_games["xG.1"].sum()+ away_games["xG"].sum()

    goals_for     = sum(int(s[0]) for s in home_games["Score"].astype(str)) \
                  + sum(int(s[2]) for s in away_games["Score"].astype(str))
    goals_against = sum(int(s[2]) for s in home_games["Score"].astype(str)) \
                  + sum(int(s[0]) for s in away_games["Score"].astype(str))

    opponents = ", ".join(home_games["Away"].tolist() + away_games["Home"].tolist())

    out = pd.DataFrame(index=[team])
    out["xGfor"]  = [total_xG_for]
    out["xGagg"]  = [total_xG_against]
    out["gf"]     = [goals_for]
    out["ga"]     = [goals_against]
    out["xG_dif"] = out["xGfor"] - out["xGagg"]
    out["GD"]     = out["gf"]    - out["ga"]
    out["teams_against"] = [opponents]
    return out

def generate_team_form_table(data, n):
    frames = [aggregate_team_form(get_last_matches(data,t,n), t) 
              for t in data["Home"].unique()]
    table = pd.concat(frames)
    table["xG_dif"] = table["xGfor"] - table["xGagg"]
    return table.sort_values("xG_dif", ascending=False)

def generate_model_data(match_data):
    match_data = match_data[match_data["Wk"] >= 6].copy()
    match_data = match_data[match_data["Score"].notna()]

    match_data["y_home_goals"] = match_data["Score"].astype(str).str[0].astype(int)
    match_data["y_away_goals"] = match_data["Score"].astype(str).str[2].astype(int)

    rows = []
    for _, row in match_data.iterrows():
        past = match_data[match_data["Wk"] < row["Wk"]]
        home_form = aggregate_team_form(get_last_matches(past,row["Home"],5), row["Home"])
        away_form = aggregate_team_form(get_last_matches(past,row["Away"],5), row["Away"])

        new = row.copy()
        for col in home_form.columns: new[f"hf5_{col}"] = home_form.iat[0, home_form.columns.get_loc(col)]
        for col in away_form.columns: new[f"af5_{col}"] = away_form.iat[0, away_form.columns.get_loc(col)]
        rows.append(new)

    df = pd.DataFrame(rows)
    df = df.drop(columns=["Score","Attendance","Referee","Season","Date","Time","Day"], errors="ignore")
    return df.reset_index(drop=True)

# -------------------------------------------------------------------
# 3. ─────────────────────  TRAIN / CACHE MODELS  ────────────────────
# -------------------------------------------------------------------
from sklearn.metrics import mean_squared_error
import numpy as np

@st.cache_resource(show_spinner="Training CatBoost (first run only)…")
def get_models():
    data  = load_match_data()
    mdata = generate_model_data(data)

    mdata_enc = pd.get_dummies(mdata, columns=["Home","Away"], dtype=int)
    features  = [c for c in mdata_enc.columns
                 if c not in ("y_home_goals","y_away_goals")
                 and "teams_against" not in c
                 and c not in ("xG","xG.1","index")]

    # --- Home model --------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        mdata_enc[features], mdata_enc["y_home_goals"], test_size=0.2, random_state=42)

    home = CatBoostRegressor(iterations=300, depth=8, learning_rate=0.05,
                             loss_function="RMSE", verbose=False)
    home.fit(X_train, y_train)

    home_rmse = np.sqrt(mean_squared_error(y_test, home.predict(X_test)))

    # --- Away model --------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        mdata_enc[features], mdata_enc["y_away_goals"], test_size=0.2, random_state=42)

    away = CatBoostRegressor(iterations=300, depth=8, learning_rate=0.05,
                             loss_function="RMSE", verbose=False)
    away.fit(X_train, y_train)

    away_rmse = np.sqrt(mean_squared_error(y_test, away.predict(X_test)))

    metrics = {"home_rmse": float(home_rmse), "away_rmse": float(away_rmse)}

    #           ↓↓↓ return metrics too
    return home, away, features, mdata_enc, metrics


# -------------------------------------------------------------------
# 4. ────────────────────  PUBLIC API FOR app.py  ────────────────────
# -------------------------------------------------------------------
def predict_fixture(season: str, gw: int) -> pd.DataFrame:
    """Return a DataFrame with Home, Away, home_pred, away_pred for the given GW."""
    data           = load_match_data()
    season_matches = data[data["Season"] == season]
    home, away, feats, template = get_models()

    # make prediction table
    preds_df = season_matches[season_matches["Wk"] == gw].copy()
    if preds_df.empty:
        return pd.DataFrame(columns=["Home","Away","home_pred","away_pred"])

    # merge 5‑game form
    past  = season_matches[(season_matches["Wk"] < gw) & (season_matches["Score"].notna())]
    form5 = generate_team_form_table(past, 5)
    preds_df = preds_df.merge(form5.add_prefix("hf5_"), left_on="Home", right_index=True)
    preds_df = preds_df.merge(form5.add_prefix("af5_"), left_on="Away", right_index=True)

    # prepare features like training set
    fe_df = preds_df.drop(columns=["Season","Date","Time","Day"], errors="ignore")
    fe_df = pd.get_dummies(fe_df, columns=["Home","Away"], dtype=int)
    fe_df = fe_df.reindex(columns=template.columns, fill_value=0)

    preds_df["home_pred"] = home.predict(fe_df[feats])
    preds_df["away_pred"] = away.predict(fe_df[feats])

    return preds_df[["Home","Away","home_pred","away_pred"]]
