from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "latest_predictions.csv"
TRAINING_PATH = ROOT / "data" / "processed" / "flood_relief_training_data.csv"
METRICS_PATH = ROOT / "reports" / "metrics.json"
FEATURE_IMPORTANCE_PATH = ROOT / "reports" / "feature_importance.csv"

st.set_page_config(page_title="FloodRelief-AI", page_icon="🌊", layout="wide")

@st.cache_data
def load_predictions() -> pd.DataFrame:
    if not DATA_PATH.exists():
        st.error("Prediction file not found. Run: python run_pipeline.py")
        st.stop()
    return pd.read_csv(DATA_PATH, parse_dates=["date"])

@st.cache_data
def load_training() -> pd.DataFrame:
    return pd.read_csv(TRAINING_PATH, parse_dates=["date"]) if TRAINING_PATH.exists() else pd.DataFrame()

@st.cache_data
def load_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}

@st.cache_data
def load_feature_importance() -> pd.DataFrame:
    return pd.read_csv(FEATURE_IMPORTANCE_PATH) if FEATURE_IMPORTANCE_PATH.exists() else pd.DataFrame()

pred = load_predictions()
training = load_training()
metrics = load_metrics()
fi = load_feature_importance()

st.title("🌊 FloodRelief-AI: Assam Flood Relief Allocation Dashboard")
scenario = pred.get("forecast_scenario", pd.Series(["Forecast scenario not recorded"])).iloc[0]
st.caption("Real-data pipeline using CivicDataLab's Assam flood ecosystem dataset. Kit demand is estimated from real affected population: kits = ceil(affected people / 5).")
st.info(f"Forecast input: {scenario}. Replace this with live IMD/BHUVAN/CWC forecasts for operational deployment.")

with st.sidebar:
    st.header("Controls")
    districts = sorted(pred["district"].unique())
    selected_districts = st.multiselect("Districts", districts, default=districts)
    risk_options = ["Low", "Medium", "High"]
    selected_risks = st.multiselect("Risk level", risk_options, default=risk_options)
    selected_date = st.selectbox("Forecast month", sorted(pred["date"].dt.date.unique()))

filtered = pred[pred["district"].isin(selected_districts) & pred["risk_level"].isin(selected_risks)].copy()
month_df = filtered[filtered["date"].dt.date == selected_date].copy()

metric_cols = st.columns(5)
metric_cols[0].metric("Predicted kits", f"{int(month_df['predicted_kits'].sum()):,}")
metric_cols[1].metric("Shortage kits", f"{int(month_df['shortage_kits'].sum()):,}")
metric_cols[2].metric("High-risk districts", int((month_df["risk_level"] == "High").sum()))
metric_cols[3].metric("Holdout MAE", metrics.get("MAE_kits", "NA"))
metric_cols[4].metric("MAPE +ve months", f"{metrics.get('MAPE_positive_months_percent', 'NA')}%")

st.divider()
left, right = st.columns([1.2, 1])

with left:
    st.subheader(f"District risk map — {selected_date}")
    if month_df.empty:
        st.warning("No rows match the filters.")
    else:
        map_df = month_df.rename(columns={"latitude": "lat", "longitude": "lon"}).copy()
        map_df["bubble_size"] = map_df["predicted_kits"].clip(lower=1)
        st.map(map_df, latitude="lat", longitude="lon", size="bubble_size", zoom=6)
        st.caption("Bubble size represents predicted kit demand. Use the table for exact values.")

with right:
    st.subheader("Top districts by predicted need")
    top_need = month_df[["district", "predicted_kits", "shortage_kits"]].sort_values("predicted_kits", ascending=False).head(10).set_index("district")
    st.bar_chart(top_need)

st.subheader("Forecast table")
show_cols = ["date", "district", "mean_rain", "sum_rain", "inundation_pct", "predicted_kits", "lower_bound_kits", "upper_bound_kits", "current_stock_kits", "shortage_kits", "risk_level", "recommended_action"]
st.dataframe(month_df[show_cols].sort_values(["predicted_kits", "shortage_kits"], ascending=False), use_container_width=True)

if not fi.empty:
    st.subheader("Top model drivers")
    fi_show = fi.head(15).copy()
    fi_show["feature"] = fi_show["feature"].str.replace("num__", "", regex=False).str.replace("cat__", "", regex=False)
    st.bar_chart(fi_show.set_index("feature")["importance"])

if not training.empty:
    st.subheader("Historical district pattern")
    hist_district = st.selectbox("Historical district", districts)
    hist = training[training["district"] == hist_district].copy()
    hist_monthly = hist.groupby("date", as_index=False).agg(mean_rain=("mean_rain", "mean"), inundation_pct=("inundation_pct", "mean"), kits_required=("kits_required", "sum"))
    st.line_chart(hist_monthly.set_index("date")[["mean_rain", "kits_required"]])

st.divider()
st.markdown("""
### NGO operations usage
1. Sort by **predicted_kits**, **shortage_kits**, and **risk_level**.
2. Treat the lower/upper bounds as uncertainty, not exact truth.
3. For High-risk districts with shortage, pre-position supplies before the forecast month.
4. Replace the included historical P90 stress-scenario inputs with live IMD rainfall, Bhuvan inundation, and CWC river-level forecasts before field deployment.
""")
