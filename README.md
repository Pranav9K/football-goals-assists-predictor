# ⚽ Football Player Future G/A Predictor 

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient_Boosting-green.svg)](https://xgboost.readthedocs.io/)
[![Pandas](https://img.shields.io/badge/Pandas-Data_Processing-lightgrey.svg)](https://pandas.pydata.org/)

## 📖 Project Overview
This project applies Machine Learning to predict a football (soccer) player's future **G+A (Goals + Assists)** production based on their historical performance and underlying advanced metrics. 

By utilizing data spanning three seasons (2023-2024, 2024-2025, and 2025-2026), the model forecasts a player's **G+A per 90 minutes** for the upcoming season. It transitions away from raw output and relies heavily on expected metrics (xG, xAG), progression stats, and custom-engineered consistency features to identify underlying talent rather than lucky outliers.

## 🎯 Motivation
In modern football scouting, relying solely on last year's goals and assists can be misleading due to variance and luck. This project was built to demonstrate how **predictive modeling** and **feature engineering** can uncover a player's true baseline performance, providing data-driven scouting insights similar to those used by professional clubs.

## 📊 Dataset
The raw data for this project was sourced from **Kaggle** and contains comprehensive player statistics including:
* **Historical Files:** `football-data_23-24.csv`, `football-data_24-25.csv`
* **Target/Test File:** `football-data_25-26.csv`
* **Key Metrics Extracted:** Expected Goals (xG), Expected Assisted Goals (xAG), Progressive Passes (PrgP), Progressive Carries (PrgC), Progressive Receptions (PrgR), and Minutes Played (90s).

## 🛠️ Data Pipeline & Feature Engineering
One of the core strengths of this project is the data preparation pipeline:
1. **Multi-Year Merging:** Cleaned and joined multiple seasons of player data using Pandas, handling missing values and ensuring player continuity across years.
2. **Per 90 Normalization:** Converted raw cumulative stats into *Per 90* metrics to accurately compare players regardless of total minutes played.
3. **Advanced Feature Creation:** Engineered custom time-series features to feed the model a deeper context of a player's trajectory:
   * `GA_consistency` / `xG_consistency`
   * `GA_volatility` / `xG_volatility`
   * `GA_trend` / `xG_trend`

## 🧠 Machine Learning Models
The project tests and evaluates several regression algorithms to find the most accurate predictor of future performance:
* **Random Forest Regressor:** Excellent for handling non-linear relationships and interactions between advanced metrics.
* **XGBoost Regressor:** Utilized for extreme gradient boosting and high-performance tabular data optimization.
* **Gradient Boosting Regressor (Champion Model):** The final, most accurate model chosen for full evaluation and projections, successfully minimizing prediction errors for the lowest Mean Absolute Error (MAE) on the test set.

**Target Variable:** `Target_GA_per90` (2025-2026 Season)  
**Evaluation Metric:** **Mean Absolute Error (MAE)**. The model evaluates its accuracy by projecting the predicted G+A/90 across the player's actual minutes played to find how many total Goals/Assists the model was off by on average.


# How to Run the Football Player Future G/A Predictor Locally

Follow these steps to set up your environment, train the machine learning models, and generate future player performance projections on your machine.

### Step 1: Clone the Repository
Open your terminal or command prompt and clone the project repository:
git clone https://github.com/yourusername/football-ga-predictor.git
cd football-ga-predictor

### Step 2: Install Required Python Libraries
Install all the data processing, scaling, and machine learning packages needed by running:
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter

### Step 3: Organize the Datasets
1. Inside your project folder, create a new folder named exactly: ModelData
2. Download your Kaggle files and place them inside that /ModelData folder.
Your directory structure should look like this:
  ```text
FOOTBALL-GA-PREDICTOR/
├── ModelData/
│   ├── football-data_23-24.csv
│   ├── football-data_24-25.csv
│   └── football-data_25-26.csv
├── Visualizations/
│   ├── cross_validation.png
│   ├── feature_importance_gb.png
│   ├── model_comparison_mae.png
│   ├── predicted_vs_actual.png
│   ├── residuals_gb.png
│   └── top10_predictions.png
├── player_GA_ml.ipynb
└── README.md
```

### Step 4: Launch and Run the Jupyter Notebook
Start the local Jupyter notebook environment with this command:
jupyter notebook

This will open a tab in your web browser. Click on 'player_GA_ml.ipynb', and you can run the code cells sequentially from top to bottom to clean the data, train the Poisson Regressor, and view the top player projections!
