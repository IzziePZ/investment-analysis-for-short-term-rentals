# Airbnb Pricing Prediction Project

This repository contains an Airbnb price prediction analysis using regression models. The goal is to estimate nightly rental prices from Airbnb listing features and local market indicators, with a focus on building a more accurate pricing model than a simple linear baseline.

## Project Overview

- **Data source:** A cleaned Airbnb dataset saved as `final_data_with_changes.csv`.
- **Primary analysis file:** `regression_models_notebook.ipynb`
- **Models evaluated:** Linear Regression and XGBoost Regression.
- **Objective:** Predict `price_total` based on listing characteristics, city-level features, and market economics.

## Key Findings

- The initial linear regression model had weak performance: it explained only about **36% of price variance**.
- The XGBoost model performed much better:
  - **MSE:** 79,902.57
  - **RMSE:** 282.67
  - **MAE:** 95.18
  - **R²:** 0.6538
- This means the XGBoost model explains roughly **65% of the variation** in nightly prices and has an average error of about **$95 per night**.

## Important Features

The strongest predictors in the XGBoost model include:

- `Monthly_Rent_Three_Bedroom_OCC`
- `Monthly_Rent_Three_Bedroom_CC`
- `Monthly_Average_Net_salary`
- `city_Vienna`
- `num_bedrooms`
- `attraction_index_norm`
- `Monthly_Rent_One_Bedroom_OCC`
- `max_guests`
- `is_superhost`
- `day_type_weekend`

These results suggest that local rental market levels, income, city location, listing size, host quality, and weekend demand are all important price drivers.

## Installation

1. Create and activate your Python environment. For example:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   python -m pip install --upgrade pip
   python -m pip install .
   ```

## Usage

1. Make sure `final_data_with_changes.csv` is in the repository root.
2. Launch Jupyter Notebook:
   ```bash
   jupyter notebook regression_models_notebook.ipynb
   ```
3. Run the notebook cells in order to reproduce the data loading, preprocessing, model training, and evaluation.

## Next Steps

- Validate the model using cross-validation or a separate holdout set.
- Tune XGBoost hyperparameters such as `max_depth`, `learning_rate`, and `n_estimators`.
- Add further feature engineering, including interaction terms or review metrics.
- Visualize predicted vs actual prices to identify remaining bias or outliers.
- Consider deploying the best model in a pricing recommendation workflow.

## Notes

- The dataset was originally downloaded from Kaggle.
- This analysis is intended as a proof-of-concept for Airbnb price prediction using machine learning.
- Dependency information is managed in `pyproject.toml`, and `uv.lock` is included to lock package versions.
