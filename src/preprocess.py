"""Preprocessing utilities for real Assam flood-relief demand forecasting."""
from __future__ import annotations

import pandas as pd

TARGET_COL = "kits_required"
DATE_COL = "date"

NUMERIC_FEATURES = [
    "rc_area", "month", "year", "time_index", "is_monsoon",
    "max_rain", "mean_rain", "sum_rain",
    "rain_lag_1", "rain_lag_2", "rain_lag_3", "rain_rolling_3", "rain_rolling_6",
    "affected_lag_1", "affected_lag_2", "affected_lag_3", "affected_rolling_3", "affected_rolling_6",
    "kits_lag_1", "kits_lag_2", "kits_lag_3", "kits_rolling_3", "kits_rolling_6",
    "inundation_pct", "inundation_intensity_mean", "inundation_intensity_sum",
    "riverlevel_mean", "riverlevel_min", "riverlevel_max",
    "sum_population", "population_density", "road_length", "rail_length",
    "health_centres_count", "schools_count", "distance_from_river", "drainage_density",
    "elevation_mean", "slope_mean", "mean_ndvi", "mean_ndbi", "water",
    "flooded_vegetation", "crops", "built_area", "bare_ground", "net_sown_area_in_hac",
    "avg_electricity", "avg_tele", "rc_piped_hhds_pct", "rc_nosanitation_hhds_pct",
]
CATEGORICAL_FEATURES = ["district", "object_id"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_dataset(path: str | None = None) -> pd.DataFrame:
    if path is None:
        try:
            from .config import TRAINING_DATA_PATH
        except ImportError:
            from config import TRAINING_DATA_PATH
        path = TRAINING_DATA_PATH
    return pd.read_csv(path, parse_dates=[DATE_COL])


def time_based_split(df: pd.DataFrame, test_start_date: str = "2025-01-01") -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(DATE_COL).copy()
    train = df[df[DATE_COL] < pd.to_datetime(test_start_date)].copy()
    test = df[df[DATE_COL] >= pd.to_datetime(test_start_date)].copy()
    if train.empty or test.empty:
        split_idx = int(len(df) * 0.8)
        train = df.iloc[:split_idx].copy()
        test = df.iloc[split_idx:].copy()
    return train, test


def get_xy(df: pd.DataFrame):
    missing = [c for c in FEATURES + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df[FEATURES], df[TARGET_COL]
