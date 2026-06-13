# Accuracy Report — FloodRelief-AI Real Data Run

## Dataset

- Source: CivicDataLab Assam flood ecosystem `MASTER_VARIABLES.csv`
- Granularity: monthly revenue-circle-level Assam records
- Rows used: 11,160
- Target: `kits_required = ceil(Population_affected_Total / 5)`

## Split

The model uses a time-based split to avoid future leakage.

- Train rows: 8,100
- Test rows: 3,060
- Holdout period: 2025-01-01 to 2026-05-01

## Metrics

| Metric | Value |
|---|---:|
| MAE | 27.944 kits |
| RMSE | 274.64 kits |
| MAPE on positive-demand months | 32.874% |
| R² on all rows | 0.841 |
| Positive target rows in test | 138 / 3060 |

## Interpretation

The model performs well on the overall zero-heavy dataset, but positive flood-demand months remain difficult. This is realistic for disaster forecasting: most months have zero demand, while a small number of flood months create large spikes.

The model should be used as a prioritisation system, not as an exact procurement engine.

## Top features

```text
                      feature  importance
      num__affected_rolling_3    0.258431
          num__kits_rolling_3    0.256877
                num__max_rain    0.113771
              num__rain_lag_3    0.054974
                num__sum_rain    0.042608
               num__mean_rain    0.037792
                   num__month    0.029930
          num__inundation_pct    0.024827
num__inundation_intensity_sum    0.023813
      num__affected_rolling_6    0.019710
```
