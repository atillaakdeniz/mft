### README.md
# 📊 Mini Forecast Tool (MFT) - Enterprise Demand Planning Platform

MFT (Mini Forecast Tool) is an **Enterprise-Grade Demand Forecasting and S&OP Platform** developed with Python and Streamlit. It is designed to optimize demand planning processes in the FMCG sector by bridging the gap between simple statistical models and modern Global Machine Learning architectures.

The software guides supply chain teams by analyzing thousands of unique SKUs, promotional fluctuations, historical stockouts, and complex time-series dynamics in seconds.

---

## 🕹️ Operation Modes (Dual-Engine Architecture)

MFT features two distinct forecasting engines tailored to different user needs:

### 1. 🛠️ Basic Mode
Designed for fast, linear, and classical supply chain approaches. It trains independent models for each SKU.
* **Models:** Naive, Moving Average, Holt-Winters, Hardcoded SARIMA $(1,1,1)$, XGBoost, LightGBM, CatBoost, Croston, SBA, TSB.
* **Metrics:** Selects the best model based on standard MAPE (Mean Absolute Percentage Error).
* **Output:** Generates direct point forecasts.

### 2. 🚀 Advanced Mode (Enterprise Edition)
Delivers advanced time-series engineering and probabilistic forecasting infrastructure used by global FMCG giants.
* **Advanced Preprocessing:** Z-Score based **Winsorization (Capping)** for promotional spikes; automatic **Stockout Imputation** via rolling means.
* **Global ML Architecture:** Utilizes a **Global LightGBM** engine that pools all SKUs into a single dataset (Shared Learning).
* **Smart Statistical Engine:** Stepwise **AutoARIMA** with AIC/BIC optimization.
* **Probabilistic Forecasting (Confidence Intervals):** Generates **P10 (Pessimistic), P50 (Target), and P90 (Optimistic)** scenarios for safety stock and risk management.
* **Forecast Value Added (FVA):** Measures the value added by advanced ML models compared to a simple Naive baseline.

---

## 🛠️ Architecture & Data Pipeline

```text
[Raw Data Upload] ➡️ [SKU Mapping & Consolidation] ➡️ [Outlier & Stockout Treatment]
         ⬇️
[Global Feature Engineering Matrix] ➡️ [Model Tournament & Cross-Validation] ➡️ [Interactive Planner Override]

```

1. **Data Consolidation:** Automatically merges historical conjugate SKUs or code changes via mapping templates.
2. **Feature Engineering:** Extracts time-series dynamics for ML models (Lags, Rolling Stats, Momentum, Cyclical Seasonality).
3. **Walk-Forward Validation:** Uses expanding window cross-validation to prevent future data leakage.
4. **FMCG-Safe Metrics:** Uses volume-weighted **WMAPE** and **Forecast Bias** instead of standard MAPE to accurately reflect business impact.

---

## 📦 Installation & Usage

### 1. Installation

Python 3.9+ is required. Run the following command in your terminal:

```bash
pip install streamlit pandas numpy plotly xgboost lightgbm catboost pmdarima scikit-learn openpyxl xlsxwriter

```

### 2. Running the App

```bash
python -m streamlit run app.py

```

### 3. Usage Guide

1. **Data Upload:** Download the "Sales Data Template" from the main screen, populate it, and upload. Use the SKU Mapping template if you need to consolidate old/new product codes.
2. **Select Mode:** Choose between **Basic** or **Advanced** mode from the sidebar.
3. **Run Forecast:** Set your forecast horizon, select the engines, and click "Run Forecast".
4. **Planner Override:** In Advanced Mode, the results grid is editable. You can manually adjust the `Final (P50)` forecast column before exporting the final plan to Excel.

---

## 📜 License & Commercial Use

This software is open-source and entirely **free** for experimental and individual use.

💼 **For commercial use, enterprise rights, please contact:**
📧 atillakdeniz@icloud.com

```

```
