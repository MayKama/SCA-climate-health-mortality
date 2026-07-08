# Climate-Attributable Risk of Sickle Cell Mortality in Africa
**Data-driven forecasts of climate-linked mortality risk to inform regional health preparedness and climate adaptation**

## Overview
This repository contains the data and modeling components for estimating and forecasting the climate-attributable risk of sickle cell mortality across Africa. The project investigates and visualizes the influence of climate variability on temporal fluctuations in sickle cell disease (SCD) mortality across African countries, enabling targeted public health interventions in the context of a warming climate.


## Repository Contents

- `Preprocessed and Merged Climate and SCA data.csv`: Final dataset combining pre-processed monthly climate variables with mortality data.
- `modeling_climate_impact_on_sickle_cell_mortality_risk_in_africa.py`: Code for training the climate-attributable mortality model and generating out-of-sample forecasts.
- `streamlit_app.py`: Dashboard interface for exploring forecast results interactively.



## Data Sources

| Dataset                              | Source/Provider                          | Description |
|--------------------------------------|------------------------------------------|-------------|
| Sickle Cell Mortality Data           | Institute for Health Metrics and Evaluation (IHME)     | yearly country-level mortality |
| Climate & Environmental Variables (Temperature, Precipitation, Aerosol Optical Depth) | Africa Data Hub | Monthly climate indicators |



## Method Summary

The modeling pipeline includes:

1. **Temporal disaggregation** (Denton-Cholette approach) to estimate monthly mortality figures from annual mortality counts reported by IHME).
2. **Feature engineering** of climate indicators using lags and rolling windows to reflect cumulative exposure effects.
3. **Gradient boosting regression (XGBoost)** to predict monthly mortality using climate features only.
4. **SHAP value decomposition** to isolate and quantify the climate-attributable signal from total model output.
5. **Time series forecasting** (SARIMA) of the climate component, producing forward-looking risk estimates at the monthly level through 2030.

The output produced is useful for estimating sickle cell mortality plausibly linked to climate variation.

##  Interpretation Caveat
Climate variables are predictive correlates and forecasts assume continuation of past climate-mortality relationships, which may not hold under severe climate change or health system shifts.

