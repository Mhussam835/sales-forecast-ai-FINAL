# 📈 Sales Forecast AI

An end-to-end Machine Learning project that forecasts weekly sales by product category using historical Superstore data, with an interactive Streamlit dashboard for exploration and predictions.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![License](https://img.shields.io/badge/License-MIT-green)

## 📋 Problem Statement

Retail businesses need to anticipate future demand to manage inventory, staffing, and marketing spend. This project builds a forecasting pipeline that predicts next-week sales per product category, backed by a full EDA uncovering actionable business insights (seasonality, discount impact, geographic concentration).

## 📊 Dataset

**Source:** [Sample Superstore Dataset](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final) (Kaggle)
- 9,994 order line-items, 2014–2017 (4 years)
- Columns: Order Date, Ship Date, Customer, Segment, City/State/Region, Category/Sub-Category, Sales, Quantity, Discount, Profit

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Data Processing | pandas, numpy |
| Visualization | matplotlib, seaborn, plotly |
| Machine Learning | scikit-learn, XGBoost, LightGBM |
| Dashboard | Streamlit |
| Model Persistence | joblib |

## 🏗️ Architecture

```
Raw CSV → Cleaning → Daily/Weekly Aggregation by Category → Feature Engineering
   (Lag, Rolling, Calendar features) → Time-based Train/Test Split →
   Model Training & Comparison → Hyperparameter Tuning → Streamlit Dashboard
```

## 📁 Project Structure

```
sales-forecast-ai/
├── data/
│   ├── raw/                  # Original untouched data
│   └── processed/            # Cleaned & feature-engineered data
├── notebooks/                # EDA & prototyping notebooks
├── src/                      # Production-ready pipeline modules
│   ├── data_loader.py
│   ├── data_cleaning.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── tune_model.py
│   ├── evaluate_model.py
│   └── utils.py
├── models/                   # Trained model + metadata
├── app/app.py                # Streamlit dashboard
├── reports/figures/          # EDA & evaluation charts
├── requirements.txt
└── README.md
```

## 📈 Key EDA Insights

1. **Strong yearly seasonality**: November–December is consistently the peak sales period across all 4 years.
2. **Discounts above 20% cause average losses** — a direct pricing-policy recommendation.
3. **Sales are geographically concentrated**: New York City alone generates ~1.5x the sales of the next city.
4. Technology, Furniture, and Office Supplies contribute relatively balanced revenue shares.

## 🤖 Modeling Results

Weekly sales aggregated per category, evaluated with a **time-based split** (no random shuffling, to avoid look-ahead bias):

| Model | MAE ($) | RMSE ($) | R² |
|---|---|---|---|
| **Linear Regression (final)** | **2,624.99** | 3,614.21 | -0.019 |
| Random Forest | 2,951.11 | 4,004.50 | -0.251 |
| Gradient Boosting | 3,431.48 | 4,486.63 | -0.571 |
| Naive Baseline (last week) | 3,745.83 | 5,022.33 | -0.969 |

All models meaningfully beat the naive baseline (~30% MAE improvement). Negative R² reflects the challenge of predicting an unprecedented holiday-season peak from only 4 years of history — documented transparently rather than hidden. See `reports/model_comparison_final.csv` for full experiment log, including hyperparameter tuning and log-transform attempts (which did **not** improve results here — an instructive real-world finding).

**Top predictive features:** Week of year, Month, and short/long-term Lag features dominate — confirming the seasonality found in EDA.

## 🚀 How to Run

```bash
git clone <your-repo-url>
cd sales-forecast-ai
pip install -r requirements.txt

# Regenerate the full pipeline
python -m src.data_loader   # or run notebooks/01-03 interactively

# Launch the dashboard
streamlit run app/app.py
```

## ☁️ Deployment


```

**Streamlit Community Cloud:**
1. Push this repo to GitHub (public or connected private repo).
2. Go to [share.streamlit.io](https://share.streamlit.io) → "New app".
3. Select your repo, branch `main`, and set the main file path to `app/app.py`.
4. Deploy — Streamlit Cloud installs `requirements.txt` automatically.

## 🔮 Future Improvements

- Extend history beyond 4 years to better capture yearly seasonality (`Lag_52` currently has only 3 prior examples)
- Try dedicated time-series models (Prophet, SARIMA) alongside tabular ML models
- Add Quantile Regression for prediction intervals instead of point estimates
- City-level forecasting once more granular/complete geographic data is available

## 📝 License

MIT
