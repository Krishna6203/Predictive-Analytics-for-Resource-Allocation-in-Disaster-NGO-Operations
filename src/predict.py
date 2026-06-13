"""Generate district-level next-month relief-kit allocation recommendations."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

try:
    from .config import FORECAST_INPUT_PATH, MODEL_PATH, PREDICTIONS_PATH
except ImportError:
    from config import FORECAST_INPUT_PATH, MODEL_PATH, PREDICTIONS_PATH


def risk_level(value: float, q40: float, q75: float) -> str:
    if value <= 0:
        return "Low"
    if value >= q75:
        return "High"
    if value >= q40:
        return "Medium"
    return "Low"


def _district_stock_simulation(districts: pd.Series, historical_peak: pd.Series) -> dict:
    stock = {}
    for district in districts.unique():
        peak = float(historical_peak.get(district, 1000))
        stock[district] = max(250, int(round(0.35 * peak)))
    return stock


def predict(input_path: Path = FORECAST_INPUT_PATH, model_path: Path = MODEL_PATH, output_path: Path = PREDICTIONS_PATH) -> pd.DataFrame:
    payload = joblib.load(model_path)
    model = payload["model"]
    features = payload["features"]
    metrics = payload.get("metrics", {})
    avg_error = float(metrics.get("MAE_kits", 250))
    df = pd.read_csv(input_path, parse_dates=["date"])
    preds = np.maximum(model.predict(df[features]), 0)
    keep_cols = ["date", "timeperiod", "district", "object_id", "latitude", "longitude", "mean_rain", "sum_rain", "inundation_pct", "sum_population", "forecast_scenario"]
    rc_out = df[keep_cols].copy()
    rc_out["predicted_kits"] = np.round(preds).astype(int)
    district = (
        rc_out.groupby(["date", "timeperiod", "district"], as_index=False)
        .agg(
            forecast_scenario=("forecast_scenario", "first"), latitude=("latitude", "first"), longitude=("longitude", "first"),
            revenue_circles=("object_id", "nunique"), mean_rain=("mean_rain", "mean"), sum_rain=("sum_rain", "sum"),
            inundation_pct=("inundation_pct", "mean"), population=("sum_population", "sum"), predicted_kits=("predicted_kits", "sum"),
        )
        .sort_values("predicted_kits", ascending=False)
    )
    district["lower_bound_kits"] = np.maximum(np.round(district["predicted_kits"] - avg_error), 0).astype(int)
    district["upper_bound_kits"] = np.round(district["predicted_kits"] + avg_error).astype(int)
    positive_preds = district.loc[district["predicted_kits"] > 0, "predicted_kits"]
    if len(positive_preds) > 0:
        q40 = positive_preds.quantile(0.40)
        q75 = positive_preds.quantile(0.75)
    else:
        q40 = q75 = 1
    district["risk_level"] = district["predicted_kits"].apply(lambda x: risk_level(x, q40, q75))
    historical_peak = district.set_index("district")["predicted_kits"]
    current_stock = _district_stock_simulation(district["district"], historical_peak)
    district["current_stock_kits"] = district["district"].map(current_stock).fillna(250).astype(int)
    district["shortage_kits"] = np.maximum(district["predicted_kits"] - district["current_stock_kits"], 0).astype(int)
    district["recommended_action"] = np.where(district["shortage_kits"] > 0, "Pre-position additional kits before the forecast month", "Monitor; current stock sufficient")
    district["forecast_note"] = "Next-month forecast input uses a historical P90 same-month stress scenario; replace with live IMD/BHUVAN/CWC forecasts for deployment."
    output_path.parent.mkdir(parents=True, exist_ok=True)
    district.to_csv(output_path, index=False)
    rc_out.to_csv(output_path.parent / "latest_predictions_revenue_circle.csv", index=False)
    return district


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(FORECAST_INPUT_PATH))
    parser.add_argument("--model", default=str(MODEL_PATH))
    parser.add_argument("--output", default=str(PREDICTIONS_PATH))
    args = parser.parse_args()
    pred_df = predict(Path(args.input), Path(args.model), Path(args.output))
    print(f"Saved district predictions: {args.output}")
    print(pred_df[["district", "predicted_kits", "shortage_kits", "risk_level"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
