# 📊 MFT OS v5.2 - Mini Forecast Tool

MFT (Mini Forecast Tool) is an **Forecasting Tool** built with Python and Streamlit. Designed specifically for the FMCG (Fast-Moving Consumer Goods) sector, it bridges the gap between traditional statistical methods and modern Global Machine Learning architectures.

MFT processes thousands of SKUs, handles promotional fluctuations, manages historical stockouts, and models complex time-series dynamics in seconds with strict data leakage prevention and ML governance.

---

## 🚀 Key Enterprise Features (v5 Architecture)

* **100% Leakage-Free Pipeline:** Expanding window limits for winsorization (outlier capping) and stockout imputation ensure that future data never leaks into past training sets.
* **Global Machine Learning (Shared Learning):** Utilizes a pooled **Global LightGBM** engine across all SKUs, capturing cross-product seasonality, momentum, and holiday effects simultaneously.
* **Strict ML Governance & Audit Trail:** Features an interactive "Planner Override" grid. The system automatically tracks any manual adjustments made by demand planners against the baseline ML forecast and exports a detailed delta/audit log.
* **Dynamic Feature Registry:** Single-source-of-truth feature engineering pipeline. Automatically generates, registers, and drops NaNs for lag safety (Lag_1 to Lag_12, Rolling Stats, Cyclical Seasonality).
* **Explainable AI (XAI):** Integrated **SHAP (SHapley Additive exPlanations)** dashboard to visualize top forecast drivers and provide model transparency to stakeholders.
* **Conformal Prediction (Uncertainty Intervals):** Generates robust **P10 (Pessimistic) and P90 (Optimistic)** bounds around the P50 forecast for optimal safety stock and risk management.
* **Parallel Processing:** Multi-threaded local model evaluation (AutoARIMA, Croston, TSB) utilizing `joblib` for maximum CPU efficiency.
* **Robust Segmentation:** Advanced multi-dimensional segmentation (ADI, CV2, Zero-Ratio) to accurately route Smooth, Volatile, Intermittent, and Lumpy SKUs to their ideal algorithmic engines.

---

## 🛠️ Architecture & Data Pipeline

```text
[Raw Data Upload] ➡️ [Monthly Alignment & SKU Consolidation] ➡️ [Leakage-Free Outlier & Stockout Treatment]
         ⬇️
[Feature Registry & Lag Safety Check] ➡️ [Parallel Walk-Forward Validation] ➡️ [Governance & Audit Delta]

```

---

## 📦 Installation & Setup

**Prerequisites:** Python 3.9 or higher.

**1. Clone the repository and navigate to the directory:**

```bash
git clone [https://github.com/atillaakdeniz/mft.git](https://github.com/atillaakdeniz/mft.git)
cd mft

```

**2. Install the required dependencies:**
Create a virtual environment (recommended) and install the packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt

```

*(Dependencies include: `streamlit, pandas, numpy, holidays, shap, joblib, plotly, lightgbm, pmdarima, scikit-learn, openpyxl, xlsxwriter`)*

**3. Run the application:**

```bash
python -m streamlit run app.py

```

The platform will automatically open in your default web browser at `http://localhost:8501`.

---

## 💡 Usage Guide

1. **Data Upload:** Click the download button in the app to get the standard `Sales Data Template`. Format your historical data (`SKU Code`, `SKU Description`, `Date`, `Sales`) and upload it.
2. **Configure Parameters:** Set your forecast horizon (e.g., 6 months) and toggle advanced treatments (Winsorization, Imputation, Conformal Intervals) from the sidebar.
3. **Select Engines:** Choose the forecasting engines to participate in the tournament (Global LightGBM, AutoARIMA, Intermittent models).
4. **Execute Pipeline:** Click **Execute Certified Pipeline**. The system will process features, train the global model, run parallel validations, and output the results.
5. **Planner Override & Export:** Review the interactive grid. Make any manual market-driven adjustments directly in the UI. When you click **Export Plan & Audit Trail**, the system downloads a multi-sheet Excel file containing both your final S&OP plan and the governance audit log.

```

```

## 📜 License & Commercial Use

This project is licensed under a custom Business Source License.
Free for personal, academic, and non-profit use.

**Commercial use requires written permission.**
💼 **For commercial use, enterprise rights, please contact:**
📧 atillakdeniz@icloud.com

© 2025 Atilla AKDENİZ
This project is provided as-is. No support SLA guaranteed.


*Created by a demand planner, for supply chain resilience.*
