# 1-Page Resource Allocation Playbook

## Objective

Use FloodRelief-AI to decide where to pre-position dry-ration kits before a forecast flood-risk month.

## Inputs required every cycle

1. Latest rainfall forecast or observed rainfall.
2. Inundation/flood-risk signal.
3. River-level trend.
4. District or warehouse stock count.
5. Transport availability.

## Dashboard workflow

1. Open the dashboard.
2. Filter to High and Medium risk districts.
3. Sort by `shortage_kits` and `predicted_kits`.
4. Prioritise districts with:
   - High predicted kits,
   - High shortage,
   - High inundation/rainfall signal,
   - Weak road/transport access.

## Decision rule

- **High risk + shortage > 0:** move kits before the forecast month.
- **High risk + no shortage:** keep monitoring and confirm local warehouse stock.
- **Medium risk:** prepare suppliers and local volunteers.
- **Low risk:** no action unless local field reports contradict the model.

## Uncertainty rule

Use `lower_bound_kits` and `upper_bound_kits` as planning uncertainty.

Do not treat the prediction as exact. Disaster data has sudden spikes.

## Field validation

Before acting, call at least one local contact from:

- District disaster management office,
- Local NGO worker,
- Panchayat/ward volunteer,
- Health or relief camp coordinator.

## Deployment note

This project package uses a historical P90 stress scenario for the forecast month. In real deployment, replace that file with live IMD/BHUVAN/CWC forecasts.
