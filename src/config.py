from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

REAL_MASTER_PATH = DATA_RAW / "civicdatalab_MASTER_VARIABLES.csv"
SOURCE_MANIFEST_PATH = DATA_RAW / "source_manifest.json"
TRAINING_DATA_PATH = DATA_PROCESSED / "flood_relief_training_data.csv"
FORECAST_INPUT_PATH = DATA_PROCESSED / "forecast_input_next_30_days.csv"
MODEL_PATH = MODELS_DIR / "flood_demand_model.joblib"
METRICS_PATH = REPORTS_DIR / "metrics.json"
PREDICTIONS_PATH = DATA_PROCESSED / "latest_predictions.csv"
FEATURE_IMPORTANCE_PATH = REPORTS_DIR / "feature_importance.csv"
HOLDOUT_PATH = REPORTS_DIR / "holdout_predictions.csv"

PERSONS_PER_DRY_RATION_KIT = 5
RANDOM_SEED = 42
