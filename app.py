"""
Streamlit Dashboard — European Automotive Price Prediction
Master's Thesis: Hedonic Pricing vs Machine Learning (2023–2025)
Author: Elbek Majidov | University of Warsaw
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path

# ── Paths ──
ROOT = Path(__file__).parent
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"

st.set_page_config(
    page_title="Auto Price Prediction — Thesis Results",
    page_icon="🚗",
    layout="wide",
)

# ── Load data ──
@st.cache_data
def load_data():
    results = pd.read_csv(DATA / "model_results.csv")
    with open(DATA / "feature_metadata.json") as f:
        feat_meta = json.load(f)
    try:
        with open(DATA / "ev_analysis_results.json") as f:
            ev_res = json.load(f)
    except FileNotFoundError:
        ev_res = {}
    try:
        with open(DATA / "statistical_tests.json") as f:
            stat_tests = json.load(f)
    except FileNotFoundError:
        stat_tests = {}
    df = pd.read_parquet(DATA / "features_engineered.parquet")
    for col in ["price_eur", "mileage_km", "power_hp", "vehicle_age", "log_price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return results, feat_meta, ev_res, stat_tests, df

results, feat_meta, ev_res, stat_tests, df = load_data()

# Split results by evaluation type
val_results = results[results["evaluation"] == "val"] if "evaluation" in results.columns else results
test_results = results[results["evaluation"] == "test"] if "evaluation" in results.columns else results

# ══════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════
st.title("European Automotive Price Prediction (2023–2025)")
st.markdown("### Hedonic Pricing Models vs Machine Learning — Master's Thesis")
st.markdown("**Author:** Elbek Majidov | **Supervisor:** Umair Ashraf Rana | **University of Warsaw**")

st.divider()

# ══════════════════════════════════════════════════════════════════════════
# TAB LAYOUT
# ══════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview & Data",
    "Model Comparison",
    "EV Analysis",
    "Feature Importance",
    "Defense Guide",
])

# ══════════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Countries", f"{df['country_code'].nunique()}")
    col3.metric("Years", "2023–2025")
    n_ev = (df["powertrain"] == "EV").sum()
    col4.metric("EV Listings", f"{n_ev:,}")

    st.subheader("Records by Year")
    yr_counts = df["data_year"].value_counts().sort_index()
    st.bar_chart(yr_counts)

    st.subheader("Price Distribution by Year")
    price_stats = df.groupby("data_year")["price_eur"].agg(["count", "mean", "median", "std"]).round(0)
    price_stats.columns = ["Count", "Mean (EUR)", "Median (EUR)", "Std Dev"]
    st.dataframe(price_stats, use_container_width=True)

    st.subheader("Powertrain Mix")
    pt_mix = pd.crosstab(df["data_year"], df["powertrain"], normalize="index").round(3) * 100
    st.dataframe(pt_mix.round(1).astype(str) + "%", use_container_width=True)

    st.subheader("Top 10 Countries by Volume")
    top_c = df["country_code"].value_counts().head(10)
    st.bar_chart(top_c)

    st.info(
        "**Why 2023–2025 only?** Years 2021 and 2022 were excluded because they had "
        "fundamentally incompatible schemas: 2021 was Polish mass-market data (PLN currency, "
        "median price EUR 5K), and 2022 had 100% missing power and engine displacement. "
        "Including them would test 'can you predict Porsche prices from Opel data' — "
        "not temporal generalization."
    )

# ══════════════════════════════════════════════════════════════════════════
# TAB 2: MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Model Comparison — Held-Out Test Set")

    st.markdown(
        "The test set is a **stratified random 15% hold-out** (34,328 vehicles), "
        "never seen during training or hyperparameter tuning."
    )

    # Categorize
    def categorize(name):
        if "Mean" in str(name):
            return "Baseline"
        if any(x in str(name) for x in ["OLS", "Ridge", "Lasso", "Elastic"]):
            return "Hedonic (Traditional)"
        return "Machine Learning"

    display = test_results.copy()
    display["Category"] = display["Model"].apply(categorize)
    show_cols = [c for c in ["Category", "Model", "RMSE_log", "MAPE_%", "R²", "MAE_EUR"] if c in display.columns]
    display_sorted = display[show_cols].sort_values("RMSE_log")

    st.dataframe(
        display_sorted.style.format({
            "RMSE_log": "{:.4f}", "MAPE_%": "{:.2f}%", "R²": "{:.4f}", "MAE_EUR": "€{:,.0f}"
        }).background_gradient(subset=["R²"], cmap="Greens"),
        use_container_width=True, hide_index=True,
    )

    # Side-by-side charts
    st.subheader("Visual Comparison")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**R² Score (higher is better)**")
        chart_data = display_sorted.set_index("Model")["R²"].sort_values()
        st.bar_chart(chart_data)

    with col2:
        st.markdown("**MAPE % (lower is better)**")
        chart_data2 = display_sorted.set_index("Model")["MAPE_%"].sort_values(ascending=False)
        st.bar_chart(chart_data2)

    # Key insight
    ml_best = display[display["Category"] == "Machine Learning"].sort_values("RMSE_log").iloc[0]
    hed_best = display[display["Category"] == "Hedonic (Traditional)"].sort_values("RMSE_log").iloc[0]
    rmse_improv = (hed_best["RMSE_log"] - ml_best["RMSE_log"]) / hed_best["RMSE_log"] * 100
    r2_improv = ml_best["R²"] - hed_best["R²"]

    st.success(
        f"**Key Finding (RQ1):** {ml_best['Model']} (ML) reduces RMSE by "
        f"**{rmse_improv:.0f}%** compared to the best hedonic model, "
        f"improving R² by **+{r2_improv:.3f}** (from {hed_best['R²']:.3f} to {ml_best['R²']:.3f})."
    )

    # Per-powertrain
    st.subheader("Performance by Powertrain Segment")
    seg_data = {
        "Segment": ["PHEV", "EV", "ICE"],
        "n (test)": [3690, 1875, 28480],
        "MAPE": ["1.0%", "2.1%", "3.5%"],
        "R²": [0.951, 0.709, 0.753],
        "MAE (EUR)": ["€5,954", "€10,224", "€11,247"],
    }
    st.dataframe(pd.DataFrame(seg_data), use_container_width=True, hide_index=True)

    st.info(
        "**PHEVs are the most predictable** (R²=0.951) because they cluster tightly around "
        "premium brands with standardized configurations. EVs show strong prediction (R²=0.71) "
        "despite being only 5% of the dataset."
    )

    # LOYO
    st.subheader("Temporal Robustness — Leave-One-Year-Out")
    loyo_data = {
        "Held-Out Year": [2023, 2024, 2025],
        "RMSE (log)": [1.380, 0.793, 0.305],
        "MAPE": ["12.1%", "6.0%", "2.2%"],
        "R²": [-0.02, 0.18, 0.86],
        "Interpretation": [
            "2023 has unique luxury composition — hard to predict from other years",
            "Partial generalization — different brand/price mix",
            "Strong — 2023+2024 together predict 2025 well",
        ],
    }
    st.dataframe(pd.DataFrame(loyo_data), use_container_width=True, hide_index=True)

    st.warning(
        "**Honest assessment:** Cross-year generalization is asymmetric. Each AutoScout24 "
        "scrape captured different vehicle segments, so predicting across years tests "
        "population transfer, not just temporal forecasting. The primary stratified split "
        "is the reliable evaluation; LOYO is a supplementary robustness check."
    )

# ══════════════════════════════════════════════════════════════════════════
# TAB 3: EV ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("Electric Vehicle Pricing Dynamics")

    ev_df = df[df["powertrain"] == "EV"]
    ice_df = df[df["powertrain"] == "ICE"]
    phev_df = df[df["powertrain"] == "PHEV"]

    col1, col2, col3 = st.columns(3)
    col1.metric("EV Listings", f"{len(ev_df):,}", f"{len(ev_df)/len(df)*100:.1f}% of total")
    col2.metric("PHEV Listings", f"{len(phev_df):,}", f"{len(phev_df)/len(df)*100:.1f}% of total")
    ev_med = ev_df["price_eur"].median()
    ice_med = ice_df["price_eur"].median()
    premium = (ev_med - ice_med) / ice_med * 100 if ice_med > 0 else 0
    col3.metric("EV Premium over ICE", f"{premium:.0f}%", f"€{ev_med - ice_med:,.0f}")

    # EV by year
    st.subheader("EV Market Evolution")
    ev_yr = ev_df.groupby("data_year").agg(
        count=("price_eur", "count"),
        median_price=("price_eur", "median"),
    ).round(0)
    total_yr = df.groupby("data_year").size()
    ev_yr["market_share_%"] = (ev_yr["count"] / total_yr * 100).round(1)
    st.dataframe(ev_yr, use_container_width=True)

    # Premium by year
    st.subheader("EV Premium Over Time")
    prem_data = []
    for yr in sorted(df["data_year"].unique()):
        ev_m = ev_df[ev_df["data_year"] == yr]["price_eur"].median()
        ice_m = ice_df[ice_df["data_year"] == yr]["price_eur"].median()
        if pd.notna(ev_m) and pd.notna(ice_m) and ice_m > 0:
            prem_data.append({"Year": yr, "Premium %": (ev_m - ice_m) / ice_m * 100})
    if prem_data:
        prem_df = pd.DataFrame(prem_data).set_index("Year")
        st.bar_chart(prem_df)

    # Material costs
    st.subheader("Battery Material Costs vs EV Prices")
    try:
        mat = pd.read_parquet(DATA / "materials.parquet")
        mat["date"] = pd.to_datetime(mat["date"])
        mat["year"] = mat["date"].dt.year
        mat_annual = mat.groupby(["year", "material"])["price_usd_per_ton"].mean().reset_index()

        mat_pivot = mat_annual.pivot(index="year", columns="material", values="price_usd_per_ton")
        st.line_chart(mat_pivot)

        if ev_res and "material_correlations" in ev_res:
            st.markdown("**Spearman Correlations (material price vs EV median price):**")
            corr_df = pd.DataFrame(ev_res["material_correlations"])
            st.dataframe(corr_df.round(3), use_container_width=True, hide_index=True)

        st.warning(
            "**Caveat:** With only 5 annual observations (2021–2025), these correlations "
            "have very low statistical power. They should be interpreted as exploratory "
            "evidence of directional alignment, not confirmatory proof of causal relationships."
        )
    except Exception:
        st.info("Run notebook 05 to generate material analysis results.")

# ══════════════════════════════════════════════════════════════════════════
# TAB 4: FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("What Drives Car Prices?")

    st.markdown(
        "Feature importance from the Random Forest and LightGBM models, "
        "showing which vehicle characteristics most influence predicted price."
    )

    try:
        import joblib
        rf = joblib.load(MODELS / "random_forest_best.pkl")
        rf_imp = pd.Series(
            rf.feature_importances_, index=feat_meta["ml_features"]
        ).sort_values(ascending=False).head(15)

        st.subheader("Random Forest — Top 15 Features")
        st.bar_chart(rf_imp)

        try:
            lgb_model = joblib.load(MODELS / "lightgbm_best.pkl")
            lgb_imp = pd.Series(
                lgb_model.feature_importances_, index=feat_meta["ml_features"]
            ).sort_values(ascending=False).head(15)
            st.subheader("LightGBM — Top 15 Features")
            st.bar_chart(lgb_imp)
        except Exception:
            pass

        st.success(
            "**Key insight:** Vehicle identity (make, model) and usage (mileage, age) "
            "are the strongest price predictors. For EVs, electric range becomes a "
            "top-5 feature. Country fixed effects capture geographic market differences."
        )
    except Exception:
        st.info("Run notebook 04 to generate trained models.")

    # Feature descriptions
    st.subheader("Feature Descriptions")
    feat_desc = {
        "model_encoded": "Vehicle model (target-encoded from training set mean price)",
        "make_encoded": "Vehicle brand/manufacturer (target-encoded)",
        "log_mileage": "Log-transformed odometer reading (km)",
        "vehicle_age": "Years since first registration",
        "power_hp": "Engine/motor power in horsepower",
        "country_fe": "Country fixed effect (avg price level in that market)",
        "body_encoded": "Body type (SUV, Sedan, Estate, etc., target-encoded)",
        "transmission_encoded": "Manual=0, Semi-Auto=1, Automatic=2",
        "data_year": "Year of listing (captures market trends)",
        "electrification_score": "Fuel type score: Electric=5, PHEV=4, Diesel/Gasoline=1",
        "electric_range_km": "Battery-only range in km (0 for non-EVs)",
        "battery_cost_index": "Weighted battery material cost (normalized, EV only)",
    }
    st.dataframe(
        pd.DataFrame(feat_desc.items(), columns=["Feature", "Description"]),
        use_container_width=True, hide_index=True,
    )

# # ══════════════════════════════════════════════════════════════════════════
# # TAB 5: DEFENSE GUIDE
# # ══════════════════════════════════════════════════════════════════════════
# with tab5:
#     st.header("Thesis Defense — Interpretation Guide")
#     st.markdown("*Use this tab to walk through the key points during your defense.*")

#     st.divider()

#     st.subheader("1. What is this research about?")
#     st.markdown("""
#     This thesis compares two approaches to predicting used car prices in Europe:

#     - **Traditional approach (Hedonic Pricing):** Linear regression models that economists
#       have used since the 1960s. They assume price is a linear combination of vehicle
#       characteristics (brand, age, mileage, etc.) plus interaction terms.

#     - **Modern approach (Machine Learning):** Tree-based ensemble models (Random Forest,
#       LightGBM, XGBoost) that can automatically discover non-linear patterns and complex
#       interactions between features.

#     We test both on **228,851 real vehicle listings** from AutoScout24 across 25 European
#     countries (2023–2025), with special attention to **electric vehicles (EVs)** and the
#     relationship between **battery material costs** and EV pricing.
#     """)

#     st.subheader("2. How was the evaluation done?")
#     st.markdown("""
#     **Primary evaluation:** We split the data into 70% training, 15% validation (for
#     tuning hyperparameters), and 15% test (for final, unbiased evaluation). The split
#     is **stratified** by country, brand, and year to ensure each subset is representative.

#     **Why not a temporal split?** We investigated this thoroughly and found that each
#     year's AutoScout24 scrape captured fundamentally different vehicle populations —
#     2023 was luxury-heavy (Porsche, BMW), 2024 was mid-range, 2025 was premium German.
#     A temporal split would test "can you predict BMW prices from Porsche data," not
#     genuine temporal forecasting. We report leave-one-year-out results as a supplementary
#     robustness check.

#     **All hyperparameters** were tuned via GridSearchCV (hedonic models) or
#     RandomizedSearchCV (ML models) on the training set only. No information from the
#     test set was used at any point.
#     """)

#     st.subheader("3. What are the main results?")
#     st.markdown(f"""
#     **RQ1: ML vs Traditional Models**

#     Machine learning models dramatically outperform hedonic pricing:
#     - **XGBoost** achieves R²=0.769 (explains 77% of price variation) with only 3.1%
#       average prediction error
#     - The best hedonic model (OLS) achieves R²=0.591 with 5.5% error
#     - This is a **25% reduction in prediction error** and means ML captures non-linear
#       patterns (e.g., depreciation curves differ by brand) that linear models cannot

#     **In plain language:** If a car costs €30,000, XGBoost predicts it within ±€930 on
#     average, while OLS is off by ±€1,650.
#     """)

#     st.markdown("""
#     **RQ2: Temporal Robustness**

#     Leave-one-year-out cross-validation shows:
#     - Training on 2023+2024 predicts 2025 well (R²=0.86) — the model generalizes forward
#     - Predicting 2023 from 2024+2025 is harder (R²≈0) — because 2023 had a unique
#       luxury-heavy composition
#     - This is an **honest finding**: cross-year generalization depends on population overlap,
#       not just time
#     """)

#     st.markdown("""
#     **RQ3: EV vs ICE Pricing**

#     - EVs carry a significant price premium over ICE vehicles
#     - The model predicts EV prices with **2.1% MAPE** (R²=0.71)
#     - PHEVs are the most predictable segment: **1.0% MAPE** (R²=0.95)
#     - Key EV price drivers: brand, vehicle age, electric range, power
#     """)

#     st.markdown("""
#     **RQ4: Material Costs and EV Prices**

#     - Battery material costs (Lithium, Cobalt, Nickel, Graphite) show directional
#       alignment with EV prices
#     - However, with only 5 annual observations, correlations lack statistical significance
#     - This is reported as **exploratory evidence**, not confirmatory proof
#     - Material costs likely influence EV prices indirectly through manufacturer pricing
#       strategies with multi-year lags
#     """)

#     st.subheader("4. What are the limitations?")
#     st.markdown("""
#     Be upfront about these — it shows academic maturity:

#     1. **Data heterogeneity:** Each year's CSV came from a different AutoScout24 scrape
#        with different schemas, countries, and vehicle segments. We addressed this by
#        dropping 2021/2022 and using stratified splitting.

#     2. **Listing prices, not sale prices:** We observe asking prices, not actual
#        transaction prices. There may be a negotiation discount.

#     3. **Feature completeness:** Engine displacement is 0% available for 2023 (dropped
#        from features). Some years have gaps in specific fields.

#     4. **Material cost granularity:** Only 5 annual data points for material prices —
#        too few for robust statistical inference.

#     5. **No causal claims:** ML models find correlations and patterns, not causal
#        relationships. The feature importance tells us what the model uses, not what
#        *causes* prices.
#     """)

#     st.subheader("5. Anticipated Questions & Answers")

#     with st.expander("Q: Why not use a neural network / deep learning?"):
#         st.markdown("""
#         Tree-based models (XGBoost, LightGBM) consistently outperform deep learning on
#         tabular data — this has been shown in multiple benchmarks (Grinsztajn et al. 2022,
#         "Why do tree-based models still outperform deep learning on tabular data?").
#         Neural networks excel with images, text, and sequences, but for structured
#         tabular data with <1M rows, gradient-boosted trees are state-of-the-art.
#         """)

#     with st.expander("Q: Why not use a proper time series model?"):
#         st.markdown("""
#         This data is **repeated cross-sections** (different cars each year), not a time
#         series (same entities tracked over time). Classical time series models (ARIMA,
#         SARIMA) require tracking the same entity and would only give us 3-5 aggregate
#         data points (annual medians) — far too few for meaningful estimation. We use
#         `data_year` as a feature instead, which allows the model to learn year effects
#         within the cross-sectional framework.
#         """)

#     with st.expander("Q: Isn't R²=0.77 too low?"):
#         st.markdown("""
#         R²=0.77 means the model explains 77% of price variance. In hedonic pricing
#         literature, R² of 0.7–0.9 is standard (Pakes 2003, Reis & Santos Silva 2006).
#         The remaining 23% comes from factors we can't observe: vehicle condition details,
#         accident history, service records, optional equipment packages, and negotiation
#         dynamics. Our MAPE of 3.1% means the average prediction error is only 3.1% of
#         the actual price — highly practical for real-world applications.
#         """)

#     with st.expander("Q: How do you handle the missing data?"):
#         st.markdown("""
#         We addressed this at multiple levels:
#         1. **Schema harmonization:** Each year's CSV had different columns — we mapped
#            them to a common schema
#         2. **Derivation:** For 2025, we derived vehicle age from `registration_date`
#            (99.8% coverage) instead of relying on the sparse `production_year` field
#         3. **Feature selection:** We dropped `engine_cc` (0% available for 2023) from
#            the feature set entirely rather than imputing it
#         4. **Imputation:** Remaining gaps (<1% for most features) handled by
#            median imputation, fitted on the training set only
#         """)

#     with st.expander("Q: What is the practical value of this research?"):
#         st.markdown("""
#         1. **For car buyers:** A 3.1% accurate pricing model helps identify overpriced
#            and underpriced vehicles
#         2. **For dealers:** Automated pricing reduces manual appraisal effort
#         3. **For policy:** Understanding EV pricing dynamics and material cost
#            pass-through helps design effective EV subsidies
#         4. **For academia:** Demonstrates that ML outperforms traditional hedonic models
#            on heterogeneous, multi-country European car data — extending the existing
#            literature which mostly focuses on single-country datasets
#         """)

# Footer
st.divider()
st.caption("Master's Thesis — University of Warsaw, 2025 | Data: AutoScout24 | Models: scikit-learn, LightGBM, XGBoost")
