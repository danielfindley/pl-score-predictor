import streamlit as st
from project import load_match_data, predict_fixture, get_models
load_match_data.clear()   # then rerun the app
get_models.clear() 
st.title("Premier League Score Predictor")

# ---- 1. Load data once and show diagnostics ------------------------
all_matches = load_match_data()

st.caption(f"📊 Loaded **{len(all_matches):,}** rows from FBref.")
if all_matches.empty:
    st.error("FBref scrape returned **0 rows** – check your internet or FBref status.")
    st.stop()
# --------------------------------------------------------------------

season = st.selectbox("Season", sorted(all_matches["Season"].unique()), index=0)

valid_gws = sorted(all_matches[all_matches["Season"] == season]["Wk"].unique())
if not valid_gws:
    st.warning(f"No fixtures found for {season}. "
               "FBref may not have published the schedule yet.")
    st.stop()

gw = st.selectbox("Gameweek", valid_gws, index=len(valid_gws)-1)

if st.button("Predict"):
    preds = predict_fixture(season, gw)
    if preds.empty:
        st.warning("No fixtures found for that game‑week.")
    else:
        st.dataframe(
            preds.style.background_gradient(
                subset=["home_pred","away_pred"], cmap="Greens"
            ).format({"home_pred":"{:.1f}", "away_pred":"{:.1f}"})
        )

        home, away, feats, template, metrics = get_models()

        st.sidebar.header("Model diagnostics")
        st.sidebar.metric("🏠 Home‑goal RMSE", f"{metrics['home_rmse']:.3f}")
        st.sidebar.metric("🚌 Away‑goal RMSE", f"{metrics['away_rmse']:.3f}")