<<<<<<< HEAD
# automotive-price-forecasting-2023-24-25
=======
# European Automotive Price Prediction (2023–2025)

## Hedonic Pricing Models vs Machine Learning — AutoScout24 Data

**Master's Thesis** — University of Warsaw, Faculty of Economic Sciences
**Author:** Elbek Majidov
**Supervisor:** Umair Ashraf Rana

---

## Research Question

> Can modern machine learning models (Random Forest, LightGBM, XGBoost) outperform traditional hedonic pricing models (OLS, Ridge, Lasso, ElasticNet) for predicting European used car prices, and how do EV and ICE pricing dynamics differ?

## Key Results

| Model Category | Best Model | RMSE (log) | MAPE | R² | MAE (EUR) |
|---|---|---|---|---|---|
| **Machine Learning** | **XGBoost** | **0.531** | **3.1%** | **0.769** | **€10,105** |
| Traditional (Hedonic) | OLS with interactions | 0.706 | 5.5% | 0.591 | €19,560 |
| Baseline | Country Mean | 1.078 | 9.0% | 0.047 | €31,062 |

**ML outperforms hedonic models by 25% in RMSE and +0.178 in R².**

### Per-Powertrain Performance (XGBoost)

| Segment | n (test) | MAPE | R² |
|---|---|---|---|
| PHEV | 3,690 | 1.0% | 0.951 |
| EV | 1,875 | 2.1% | 0.709 |
| ICE | 28,480 | 3.5% | 0.753 |

---

## Dataset

- **Source:** AutoScout24 European vehicle listings
- **Years:** 2023, 2024, 2025 (2021/2022 excluded due to incompatible schemas)
- **Size:** 228,851 cleaned records
- **Countries:** 25 European markets (DE, IT, NL, BE, FR, AT, ES, LU, ...)
- **Powertrains:** 190K ICE, 25K PHEV, 12K EV
- **Supplementary:** Rare-earth material prices (Lithium, Cobalt, Nickel, Graphite, Neodymium, Dysprosium)

## Methodology

### Data Pipeline
1. **Harmonization:** 3 CSVs with different schemas unified into common format
2. **Cleaning:** Currency conversion, price bounds (€800–€300K), outlier removal, deduplication
3. **Feature Engineering:** 24 ML features + 6 interaction terms for hedonic models

### Evaluation Strategy
- **Primary:** Stratified random split (70/15/15) by country × make × year
- **Secondary:** Leave-one-year-out cross-validation for temporal robustness
- **Metrics:** RMSE (log-price), MAPE, R², MAE (EUR)

### Models Compared
| Category | Models | Hyperparameter Tuning |
|---|---|---|
| Baselines | Global Mean, Country Mean | — |
| Hedonic (Traditional) | OLS, Ridge, Lasso, ElasticNet | GridSearchCV (cv=5) |
| Machine Learning | Random Forest, LightGBM, XGBoost | RandomizedSearchCV (cv=3) |

---

## Project Structure

```
master-thesis-9/
├── config.py                          # Central configuration
├── README.md                          # This file
├── app.py                             # Streamlit dashboard
├── data/
│   ├── raw/                           # Original CSVs
│   │   ├── autoscout24_2023.csv
│   │   ├── autoscout24_2024.csv
│   │   ├── autoscout24_full.csv       # 2025 data
│   │   └── ev_materials_prices_2021_2025.csv
│   └── processed/
│       ├── unified_dataset.parquet    # Cleaned, harmonized
│       ├── features_engineered.parquet
│       ├── feature_metadata.json
│       ├── model_results.csv
│       ├── statistical_tests.json
│       └── ev_analysis_results.json
├── notebooks/
│   ├── 01_data_harmonization.ipynb    # Raw → unified dataset
│   ├── 02_eda.ipynb                   # Exploratory analysis
│   ├── 03_feature_engineering.ipynb   # Features + train/val/test split
│   ├── 04_modeling.ipynb              # All models + GridSearchCV
│   ├── 05_ev_analysis.ipynb           # EV deep-dive + materials
│   └── 06_final_results.ipynb        # Summary, RQ answers, dashboard
├── models/
│   ├── xgboost_best.pkl               # Best model
│   ├── lightgbm_best.pkl
│   ├── random_forest_best.pkl
│   ├── preprocessor_ml.pkl
│   └── preprocessor_hedonic.pkl
└── figures/                           # All generated plots
```

## Setup & Installation

```bash
# Clone / navigate to project
cd master-thesis-9

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm xgboost
pip install scipy statsmodels pyarrow joblib streamlit

# Run notebooks in order
jupyter notebook notebooks/01_data_harmonization.ipynb

# Or run the Streamlit dashboard
streamlit run app.py
```

### Requirements
- Python 3.10+
- pandas, numpy, matplotlib, seaborn
- scikit-learn >= 1.3
- lightgbm >= 4.0
- xgboost >= 2.0
- scipy, statsmodels
- pyarrow (parquet I/O)
- streamlit (dashboard)

## Research Questions & Answers

**RQ1: Do ML models outperform hedonic pricing models?**
Yes. XGBoost reduces RMSE by 25% (0.531 vs 0.706) and improves R² by +0.178 over the best hedonic model. All three ML models (RF, LightGBM, XGBoost) significantly outperform all four hedonic variants.

**RQ2: How robust are models to temporal variation?**
Leave-one-year-out CV shows asymmetric generalization: training on 2023+2024 predicts 2025 well (R²=0.86), but predicting 2023 from 2024+2025 is harder (R²≈0). This reflects genuine market composition differences across scraping periods.

**RQ3: How do EV and ICE pricing dynamics differ?**
EVs carry a significant price premium over ICE vehicles (Cohen's d > 0.5). The model predicts EV prices with MAPE=2.1% (R²=0.71). PHEVs are the most predictable segment (R²=0.95, MAPE=1.0%).

**RQ4: Do material costs correlate with EV prices?**
Weak-to-moderate correlations with only 5 annual observations. The Battery Cost Index shows directional alignment but lacks statistical significance due to limited temporal granularity.

**RQ5: What drives car prices?**
Top features: model identity, make identity, mileage, vehicle age, power, country, and body type. For EVs specifically, electric range becomes a key predictor.

## Reproducibility

All results are reproducible with `RANDOM_STATE = 42`. Run notebooks 01–06 in sequence, or use the pre-computed artifacts in `data/processed/` and `models/`.

## License

This project is part of academic research at the University of Warsaw. Data sourced from AutoScout24 public listings.
>>>>>>> 55ea083 (Initial commit — automotive price forecasting thesis)
