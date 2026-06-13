"""Train and evaluate the real-data Assam flood-relief demand model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from .config import FEATURE_IMPORTANCE_PATH, HOLDOUT_PATH, METRICS_PATH, MODEL_PATH, MODELS_DIR, REPORTS_DIR, TRAINING_DATA_PATH
    from .preprocess import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, get_xy, load_dataset, time_based_split
except ImportError:
    from config import FEATURE_IMPORTANCE_PATH, HOLDOUT_PATH, METRICS_PATH, MODEL_PATH, MODELS_DIR, REPORTS_DIR, TRAINING_DATA_PATH
    from preprocess import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, get_xy, load_dataset, time_based_split


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true > 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def build_model(random_state: int = 42) -> Pipeline:
    numeric_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_pipe, NUMERIC_FEATURES), ("cat", categorical_pipe, CATEGORICAL_FEATURES)],
        remainder="drop",
    )
    regressor = RandomForestRegressor(n_estimators=160, max_depth=16, min_samples_leaf=2, random_state=random_state, n_jobs=-1)
    return Pipeline(steps=[("preprocess", preprocessor), ("model", regressor)])


def evaluate(y_true, y_pred) -> dict:
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    positive_mask = np.asarray(y_true) > 0
    return {
        "MAE_kits": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "RMSE_kits": round(float(rmse), 3),
        "MAPE_positive_months_percent": round(float(mape(y_true, y_pred)), 3),
        "R2_all_rows": round(float(r2_score(y_true, y_pred)), 4),
        "positive_target_rows_in_test": int(positive_mask.sum()),
        "test_rows": int(len(y_true)),
    }


def _extract_feature_importance(model: Pipeline) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocess"]
    reg = model.named_steps["model"]
    feature_names = list(preprocessor.get_feature_names_out())
    importances = getattr(reg, "feature_importances_", np.full(len(feature_names), np.nan))
    return pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)


def train(data_path: Path = TRAINING_DATA_PATH, model_path: Path = MODEL_PATH) -> tuple[Pipeline, dict, pd.DataFrame]:
    df = load_dataset(str(data_path))
    train_df, test_df = time_based_split(df)
    X_train, y_train = get_xy(train_df)
    X_test, y_test = get_xy(test_df)
    model = build_model()
    model.fit(X_train, y_train)
    test_pred = np.maximum(model.predict(X_test), 0)
    metrics = evaluate(y_test, test_pred)
    metrics["train_rows"] = int(len(train_df))
    metrics["test_start_date"] = str(test_df["date"].min().date())
    metrics["test_end_date"] = str(test_df["date"].max().date())
    metrics["target_definition"] = "kits_required = ceil(Population_affected_Total / 5)"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "numeric_features": NUMERIC_FEATURES, "categorical_features": CATEGORICAL_FEATURES, "metrics": metrics}, model_path)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    keep_cols = ["date", "timeperiod", "district", "object_id", "mean_rain", "sum_rain", "inundation_pct", "Population_affected_Total", "kits_required"]
    result_df = test_df[keep_cols].copy()
    result_df["predicted_kits"] = np.round(test_pred).astype(int)
    result_df["absolute_error"] = np.abs(result_df["kits_required"] - result_df["predicted_kits"])
    result_df.to_csv(HOLDOUT_PATH, index=False)
    _extract_feature_importance(model).to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    return model, metrics, result_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(TRAINING_DATA_PATH))
    parser.add_argument("--model", default=str(MODEL_PATH))
    args = parser.parse_args()
    _, metrics, _ = train(Path(args.data), Path(args.model))
    print("Model trained successfully on real Assam flood ecosystem data.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
