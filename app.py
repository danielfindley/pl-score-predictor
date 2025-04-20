# app.py
import streamlit as st
from project import predict_fixture

st.set_page_config(page_title="Premier‑League Score Predictor")
st.title("Premier League Score Predictor")

season = st.selectbox("Season", ["2024-2025","2023-2024","2022-2023","2021-2022"])
gw     = st.number_input("Gameweek", min_value=1, max_value=38, value=30, step=1)

if st.button("Predict"):
    with st.spinner("Crunching the numbers…"):
        preds = predict_fixture(season, gw)
    st.subheader(f"Predicted scores – GW{gw} {season}")
    st.dataframe(
        preds.style.background_gradient(
            subset=["home_pred","away_pred"], cmap="Greens"
        ).format({"home_pred":"{:.1f}","away_pred":"{:.1f}"})
    )
