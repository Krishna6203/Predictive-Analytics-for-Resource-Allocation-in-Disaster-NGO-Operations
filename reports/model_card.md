# Model Card — FloodRelief-AI

## Model

Random Forest Regressor trained on real Assam monthly revenue-circle records.

## Intended use

District-level planning support for flood-relief dry-ration kit pre-positioning.

## Target

`kits_required = ceil(Population_affected_Total / 5)`

This is a proxy for relief demand. Actual NGO inventory consumption was not available in public datasets.

## Inputs

- Rainfall: max, mean, sum, lagged and rolling rainfall
- Flood/inundation: inundation percentage and intensity
- River levels
- Population and population density
- Road/rail/infrastructure indicators
- Historical affected-population and kit-demand lags
- District and revenue-circle identity

## Evaluation

- MAE: 27.944 kits
- RMSE: 274.64 kits
- MAPE positive months: 32.874%
- R²: 0.841

## Limitations

1. Real relief-kit demand is not directly reported.
2. Dataset is monthly, so the dashboard is a 30-day/monthly planning prototype, not an hourly emergency-response engine.
3. Public disaster data is sparse and zero-heavy.
4. Forecast input currently uses a historical P90 same-month stress scenario, not a live IMD forecast.

## Safer deployment approach

Before field use, replace forecast inputs with live IMD/BHUVAN/CWC data and validate outputs with an NGO/district disaster-management official.
