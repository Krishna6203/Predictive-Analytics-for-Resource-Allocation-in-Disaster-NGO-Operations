"""Prepare real Assam flood ecosystem data for relief-kit demand forecasting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .config import DATA_PROCESSED, DATA_RAW, FORECAST_INPUT_PATH, PERSONS_PER_DRY_RATION_KIT, REAL_MASTER_PATH, SOURCE_MANIFEST_PATH, TRAINING_DATA_PATH
except ImportError:
    from config import DATA_PROCESSED, DATA_RAW, FORECAST_INPUT_PATH, PERSONS_PER_DRY_RATION_KIT, REAL_MASTER_PATH, SOURCE_MANIFEST_PATH, TRAINING_DATA_PATH

DISTRICT_CENTROIDS = {
    "BAJALI": (26.47, 91.18), "BAKSA": (26.69, 91.30), "BARPETA": (26.32, 91.01),
    "BISWANATH": (26.72, 93.15), "BONGAIGAON": (26.48, 90.56), "CACHAR": (24.83, 92.79),
    "CHARAIDEO": (27.02, 95.02), "CHIRANG": (26.66, 90.64), "DARRANG": (26.45, 92.03),
    "DHEMAJI": (27.48, 94.58), "DHUBRI": (26.02, 89.97), "DIBRUGARH": (27.47, 94.91),
    "DIMA HASAO": (25.35, 93.02), "GOALPARA": (26.17, 90.62), "GOLAGHAT": (26.52, 93.96),
    "HAILAKANDI": (24.68, 92.56), "HOJAI": (26.00, 92.86), "JORHAT": (26.75, 94.22),
    "KAMRUP": (26.15, 91.60), "KAMRUP METRO": (26.14, 91.74), "KARBI ANGLONG": (25.85, 93.44),
    "KARIMGANJ": (24.87, 92.35), "KOKRAJHAR": (26.40, 90.27), "LAKHIMPUR": (27.24, 94.10),
    "MAJULI": (27.00, 94.22), "MORIGAON": (26.25, 92.35), "NAGAON": (26.35, 92.68),
    "NALBARI": (26.44, 91.44), "SIVASAGAR": (26.98, 94.63), "SONITPUR": (26.65, 92.79),
    "SOUTH SALMARA MANCACHAR": (25.70, 89.90), "TAMULPUR": (26.64, 91.55), "TINSUKIA": (27.49, 95.36),
    "UDALGURI": (26.75, 92.10), "WEST KARBI ANGLONG": (25.80, 92.90),
}

RAW_SOURCE = "https://github.com/CivicDataLab/flood-data-ecosystem-Assam/raw/refs/heads/main/Sources/MASTER_VARIABLES.csv"
REPO_SOURCE = "https://github.com/CivicDataLab/flood-data-ecosystem-Assam"
DAMAGE_COLUMNS = {
    "Total_Animal_Washed_Away", "Total_Animal_Affected", "Population_affected_Total", "Crop_Area",
    "Male_Camp", "Female_Camp", "Children_Camp", "Total_House_Fully_Damaged", "Human_Live_Lost",
    "Human_Live_Lost_Children", "Human_Live_Lost_Female", "Human_Live_Lost_Male", "Embankments affected",
    "Roads", "Bridge", "Embankment breached", "Relief Camps", "Relief Centers", "Relief Inmates", "Relief_Camp_inmates",
}


def _timeperiod_to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.astype(str).str.replace("_", "-", regex=False) + "-01")


def _add_time_and_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["object_id", "date"]).copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["time_index"] = (df["year"] - df["year"].min()) * 12 + df["month"]
    df["is_monsoon"] = df["month"].isin([6, 7, 8, 9]).astype(int)
    g = df.groupby("object_id", group_keys=False)
    df["rain_lag_1"] = g["mean_rain"].shift(1)
    df["rain_lag_2"] = g["mean_rain"].shift(2)
    df["rain_lag_3"] = g["mean_rain"].shift(3)
    df["rain_rolling_3"] = g["mean_rain"].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    df["rain_rolling_6"] = g["mean_rain"].rolling(6, min_periods=1).mean().reset_index(level=0, drop=True)
    if "affected_population" in df.columns:
        df["affected_lag_1"] = g["affected_population"].shift(1)
        df["affected_lag_2"] = g["affected_population"].shift(2)
        df["affected_lag_3"] = g["affected_population"].shift(3)
        df["affected_rolling_3"] = g["affected_population"].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
        df["affected_rolling_6"] = g["affected_population"].rolling(6, min_periods=1).mean().reset_index(level=0, drop=True)
    if "kits_required" in df.columns:
        df["kits_lag_1"] = g["kits_required"].shift(1)
        df["kits_lag_2"] = g["kits_required"].shift(2)
        df["kits_lag_3"] = g["kits_required"].shift(3)
        df["kits_rolling_3"] = g["kits_required"].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
        df["kits_rolling_6"] = g["kits_required"].rolling(6, min_periods=1).mean().reset_index(level=0, drop=True)
    for col in [c for c in df.columns if any(k in c for k in ["lag", "rolling"] )]:
        df[col] = df[col].fillna(0)
    return df


def prepare_training_data(raw_path: Path = REAL_MASTER_PATH) -> pd.DataFrame:
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing {raw_path}. Run src/fetch_real_data.py first.")
    df = pd.read_csv(raw_path)
    df["district"] = df["district"].astype(str).str.upper().str.strip()
    df["object_id"] = df["object_id"].astype(str)
    df["date"] = _timeperiod_to_date(df["timeperiod"])
    for col in df.columns:
        if col not in ["object_id", "district", "timeperiod", "date", "dtname"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["population_density"] = np.where(df["rc_area"] > 0, df["sum_population"] / df["rc_area"], 0)
    df["affected_population"] = df["Population_affected_Total"].clip(lower=0)
    df["kits_required"] = np.ceil(df["affected_population"] / PERSONS_PER_DRY_RATION_KIT).astype(int)
    df = _add_time_and_lag_features(df)
    df["latitude"] = df["district"].map(lambda d: DISTRICT_CENTROIDS.get(d, (26.2, 92.6))[0])
    df["longitude"] = df["district"].map(lambda d: DISTRICT_CENTROIDS.get(d, (26.2, 92.6))[1])
    return df


def build_forecast_input(training_df: pd.DataFrame) -> pd.DataFrame:
    latest_date = training_df["date"].max()
    next_date = latest_date + pd.DateOffset(months=1)
    next_month = int(next_date.month)
    latest = training_df.sort_values("date").groupby("object_id", as_index=False).tail(1).copy()
    future = latest.copy()
    future["date"] = next_date
    future["timeperiod"] = f"{next_date.year}_{next_date.month:02d}"
    forecast_cols = [
        "max_rain", "mean_rain", "sum_rain", "inundation_pct", "inundation_intensity_mean",
        "inundation_intensity_sum", "riverlevel_mean", "riverlevel_min", "riverlevel_max", "mean_ndvi", "mean_ndbi",
        "water", "flooded_vegetation", "crops", "built_area", "bare_ground",
    ]
    hist_same_month = training_df[training_df["month"] == next_month]
    obj_q90 = hist_same_month.groupby("object_id")[forecast_cols].quantile(0.90)
    district_q90 = hist_same_month.groupby("district")[forecast_cols].quantile(0.90)
    global_q90 = training_df[forecast_cols].quantile(0.90)
    for idx, row in future.iterrows():
        oid, district = row["object_id"], row["district"]
        if oid in obj_q90.index:
            vals = obj_q90.loc[oid]
        elif district in district_q90.index:
            vals = district_q90.loc[district]
        else:
            vals = global_q90
        for col in forecast_cols:
            future.at[idx, col] = vals[col]
    for col in DAMAGE_COLUMNS:
        if col in future.columns:
            future[col] = 0
    future["affected_population"] = 0
    future["kits_required"] = 0
    combined = pd.concat([training_df, future], ignore_index=True).sort_values(["object_id", "date"])
    combined = _add_time_and_lag_features(combined)
    forecast = combined[combined["date"] == next_date].copy()
    forecast["latitude"] = forecast["district"].map(lambda d: DISTRICT_CENTROIDS.get(d, (26.2, 92.6))[0])
    forecast["longitude"] = forecast["district"].map(lambda d: DISTRICT_CENTROIDS.get(d, (26.2, 92.6))[1])
    forecast["forecast_scenario"] = f"Historical P90 same-month stress scenario for {next_date:%B %Y}"
    return forecast


def write_source_manifest() -> None:
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    manifest = {
        "primary_dataset": "CivicDataLab flood-data-ecosystem-Assam MASTER_VARIABLES.csv",
        "raw_url": RAW_SOURCE,
        "repository": REPO_SOURCE,
        "granularity": "Monthly revenue-circle-level records for Assam",
        "target_proxy": "kits_required = ceil(Population_affected_Total / 5)",
        "forecast_scenario": "Next-month rainfall/inundation/river-level features are historical P90 same-month values unless replaced with live forecasts.",
    }
    SOURCE_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default=str(REAL_MASTER_PATH))
    parser.add_argument("--training-output", default=str(TRAINING_DATA_PATH))
    parser.add_argument("--forecast-output", default=str(FORECAST_INPUT_PATH))
    args = parser.parse_args()
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df = prepare_training_data(Path(args.raw))
    df.to_csv(args.training_output, index=False)
    forecast = build_forecast_input(df)
    forecast.to_csv(args.forecast_output, index=False)
    write_source_manifest()
    print(f"Prepared real training data: {args.training_output} ({len(df):,} rows, {df['district'].nunique()} districts)")
    print(f"Prepared next-month forecast input: {args.forecast_output} ({len(forecast):,} revenue-circle rows)")


if __name__ == "__main__":
    main()
