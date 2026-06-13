# FloodRelief-AI: Predictive Resource Allocation for Assam Flood Operations

This project implements **Problem Statement 1.2 — Predictive Analytics for Resource Allocation in Disaster & NGO Operations**.

It uses real public Assam flood ecosystem data to forecast district-level dry-ration kit demand for flood relief planning. The model converts real `Population_affected_Total` records into an operational planning target:

```text
kits_required = ceil(Population_affected_Total / 5)
```

> Important: public datasets generally do not contain actual NGO inventory demand. This project therefore uses affected population as a real disaster-impact proxy and converts it into dry-ration kit demand using a 5-person-per-kit planning assumption.

## Real data used

Primary raw file:

```text
data/raw/civicdatalab_MASTER_VARIABLES.csv
```

Source: CivicDataLab `flood-data-ecosystem-Assam` repository, `Sources/MASTER_VARIABLES.csv`.

The downloaded dataset in this package has:

```text
Rows: 11,160
Columns: 85
Granularity: monthly revenue-circle-level records
Geography: Assam districts/revenue circles
Time range used by project: 2021-04 to 2026-05
```

The dataset combines rainfall, inundation, river-level, infrastructure, demographic, and flood-damage variables.

## What the project does

1. Prepares real monthly revenue-circle data.
2. Builds lag and rolling features for rainfall, affected population, and kit demand.
3. Trains a Random Forest regression model using a time-based train/test split.
4. Generates next-month district-level forecasts.
5. Converts predictions into allocation recommendations using a simulated stock rule.
6. Provides a Streamlit dashboard for NGO/district operations users.

## Current model result

Holdout period: `2025-01-01` to `2026-05-01`

```text
MAE: 27.944 kits
RMSE: 274.64 kits
MAPE on positive-demand months: 32.874%
R² on all rows: 0.841
Positive target rows in test: 138 / 3060
```

The MAPE is above the ideal 20% threshold because public disaster impact data is sparse and zero-heavy. The dashboard is still operationally useful as a prioritisation prototype.

## Forecast scenario

`data/processed/forecast_input_next_30_days.csv` is built from a **historical P90 same-month stress scenario**. This is not a live weather forecast. In real deployment, replace those columns with live IMD rainfall, Bhuvan inundation, and CWC river-level forecasts.

Current district forecast file:

```text
data/processed/latest_predictions.csv
```

Top current predicted districts:

```text
  district  predicted_kits  shortage_kits risk_level
    CACHAR             507            257       High
 LAKHIMPUR               6              0        Low
    BAJALI               0              0        Low
   BARPETA               0              0        Low
     BAKSA               0              0        Low
BONGAIGAON               0              0        Low
 BISWANATH               0              0        Low
   CHIRANG               0              0        Low
   DARRANG               0              0        Low
   DHEMAJI               0              0        Low
```

## How to run

```bash
cd floodrelief_ai
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate    # macOS/Linux
pip install -r requirements.txt

python run_pipeline.py
streamlit run dashboard/app.py
```

To redownload the raw public dataset:

```bash
python src/fetch_real_data.py --force
```

## Project structure

```text
floodrelief_ai/
├── data/
│   ├── raw/
│   └── processed/
├── dashboard/
│   └── app.py
├── models/
│   └── flood_demand_model.joblib
├── reports/
│   ├── accuracy_report.md
│   ├── feature_importance.csv
│   ├── holdout_predictions.csv
│   ├── metrics.json
│   ├── model_card.md
│   └── resource_allocation_playbook.md
├── src/
│   ├── fetch_real_data.py
│   ├── prepare_real_data.py
│   ├── preprocess.py
│   ├── train_model.py
│   └── predict.py
├── tests/
│   └── test_pipeline.py
├── requirements.txt
├── run_pipeline.py
└── README.md
```

## Submission notes

For an NSS/open-project demo, show:

1. The real data source and target proxy.
2. The model metrics.
3. The dashboard.
4. The allocation playbook.
5. The limitation: this forecasts planning demand, not exact NGO inventory consumption.
