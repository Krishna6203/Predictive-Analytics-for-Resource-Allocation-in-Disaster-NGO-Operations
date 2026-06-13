from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_training_data_exists_and_has_target():
    path = ROOT / "data" / "processed" / "flood_relief_training_data.csv"
    assert path.exists()
    df = pd.read_csv(path)
    assert "kits_required" in df.columns
    assert "Population_affected_Total" in df.columns
    assert len(df) > 10000


def test_predictions_exist_and_have_required_columns():
    path = ROOT / "data" / "processed" / "latest_predictions.csv"
    assert path.exists()
    df = pd.read_csv(path)
    required = {"date", "district", "predicted_kits", "risk_level", "shortage_kits", "recommended_action"}
    assert required.issubset(df.columns)
    assert (df["predicted_kits"] >= 0).all()
