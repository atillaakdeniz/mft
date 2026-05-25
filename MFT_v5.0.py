import streamlit as st
import pandas as pd
import numpy as np
import io
import time
import traceback
import plotly.express as px
import plotly.graph_objects as go
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import pmdarima as pm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Mini Forecast Tool (MFT)", layout="wide")

# ============================================================
#  CONSTANTS
# ============================================================
REQUIRED_COLUMNS = {'SKU Code', 'Date', 'Sales'}
OPTIONAL_COLUMNS = {'SKU Description'}
COLUMN_ALIASES = {
    'SKU Kodu':   'SKU Code',
    'SKU Tanım':  'SKU Description',
    'Tarih':      'Date',
    'Satış KG':   'Sales',
}
MAPPING_REQUIRED_COLUMNS = {'Old SKU Code', 'New SKU Code'}
MAPPING_ALIASES = {
    'Eski SKU Kodu':   'Old SKU Code',
    'Güncel SKU Kodu': 'New SKU Code',
}

# ============================================================
#  VALIDATION HELPERS
# ============================================================
def validate_sales_file(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Returns (is_valid: bool, error_message: str).
    Applies column aliases first, then checks required columns,
    data types, and basic sanity.
    """
    df = df.rename(columns=COLUMN_ALIASES)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        readable = ", ".join(f"**{c}**" for c in sorted(missing))
        found    = ", ".join(f"`{c}`" for c in df.columns.tolist())
        return False, (
            f"Missing required column(s): {readable}.\n\n"
            f"Columns found in your file: {found}.\n\n"
            "Please download the template above and match the column names exactly."
        )

    # Date parseable?
    try:
        pd.to_datetime(df['Date'])
    except Exception:
        return False, (
            "The **Date** column could not be parsed. "
            "Please use a standard date format such as `YYYY-MM-DD` or `DD/MM/YYYY`."
        )

    # Sales numeric?
    if not pd.api.types.is_numeric_dtype(df['Sales']):
        try:
            pd.to_numeric(df['Sales'])
        except Exception:
            return False, (
                "The **Sales** column contains non-numeric values. "
                "Please ensure all sales figures are numbers."
            )

    # At least some data?
    if len(df) == 0:
        return False, "The uploaded file appears to be empty (no data rows found)."

    # Negative sales warning (not fatal)
    neg_count = (pd.to_numeric(df['Sales'], errors='coerce').fillna(0) < 0).sum()
    if neg_count > 0:
        st.warning(
            f"⚠️ {neg_count:,} row(s) have negative Sales values. "
            "These will be treated as 0 during processing."
        )

    return True, ""


def validate_mapping_file(df: pd.DataFrame) -> tuple[bool, str]:
    """Validates the optional SKU mapping file."""
    df = df.rename(columns=MAPPING_ALIASES)
    missing = MAPPING_REQUIRED_COLUMNS - set(df.columns)
    if missing:
        readable = ", ".join(f"**{c}**" for c in sorted(missing))
        return False, (
            f"SKU Mapping file is missing column(s): {readable}. "
            "Please download the mapping template and use its column names."
        )
    if df[['Old SKU Code', 'New SKU Code']].isnull().any().any():
        st.warning("⚠️ SKU Mapping file contains blank cells — those rows will be ignored.")
    return True, ""


# ============================================================
#  UI: SIDEBAR — MODE SELECTION
# ============================================================
st.sidebar.header("🕹️ Operation Mode")
app_mode = st.sidebar.radio("Select Forecasting Engine:", ["Basic Mode", "Advanced Mode"])
st.sidebar.markdown("---")

# ============================================================
#  UI: SIDEBAR — DYNAMIC PARAMETERS
# ============================================================
if app_mode == "Basic Mode":
    st.sidebar.header("⚙️ Basic Parameters")
    train_months   = st.sidebar.number_input("Max Training Period (Months):", min_value=12, max_value=60, value=36)
    test_months    = st.sidebar.slider("Validation Period (Months):", 1, 12, 3)
    forecast_horizon = st.sidebar.slider("Forecast Horizon (Months):", 1, 24, 6)

    use_min_sales  = st.sidebar.checkbox("Enable Min Continuous Sales Rule", value=True)
    min_sales_req  = st.sidebar.slider("Min Continuous Sales (Months):", 1, 12, 6) if use_min_sales else 1

    st.sidebar.markdown("---")
    st.sidebar.subheader("🚦 Minimum Forecast Threshold")
    use_min_fc_threshold = st.sidebar.checkbox("Enable Minimum Forecast Filter", value=False)
    min_fc_threshold = (
        st.sidebar.number_input(
            "Min. Monthly Avg. Forecast (units):",
            min_value=0, max_value=5000, value=3000, step=100,
            help="SKUs whose forecasted monthly average falls below this value will be excluded from results."
        )
        if use_min_fc_threshold else 0
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🧠 Basic Models")
    ML_MODELS_BASIC = ['Naive', 'Moving Average', 'Holt-Winters', 'SARIMA',
                        'XGBoost', 'LightGBM', 'CatBoost', 'Croston', 'SBA', 'TSB']
    available_models = st.sidebar.multiselect(
        "Select Models to Test:", ML_MODELS_BASIC,
        default=['Moving Average', 'Holt-Winters', 'SARIMA', 'XGBoost', 'Croston']
    )

elif app_mode == "Advanced Mode":
    st.sidebar.header("⚙️ Advanced Parameters")
    train_months     = st.sidebar.number_input("Max Training History (Months):", min_value=12, max_value=60, value=36)
    forecast_horizon = st.sidebar.slider("Forecast Horizon (Months):", 1, 24, 6)
    test_months      = 3   # fixed in advanced mode

    st.sidebar.markdown("---")
    st.sidebar.subheader("🛡️ Data Preprocessing")
    handle_outliers  = st.sidebar.checkbox("Apply Outlier Capping (Winsorization)", value=True)
    handle_stockouts = st.sidebar.checkbox("Detect & Impute Stockouts", value=True)

    use_min_sales = st.sidebar.checkbox("Enable Min Continuous Sales Rule", value=True)
    min_sales_req = st.sidebar.slider("Min Continuous Sales (Months):", 1, 12, 6) if use_min_sales else 1

    st.sidebar.markdown("---")
    st.sidebar.subheader("🚦 Minimum Forecast Threshold")
    use_min_fc_threshold = st.sidebar.checkbox("Enable Minimum Forecast Filter", value=False)
    min_fc_threshold = (
        st.sidebar.number_input(
            "Min. Monthly Avg. Forecast (units):",
            min_value=0, max_value=5000, value=3000, step=100,
            help="SKUs whose forecasted monthly average falls below this value will be excluded from results."
        )
        if use_min_fc_threshold else 0
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🧠 Enterprise Models")
    ML_MODELS_ADV = ['AutoARIMA (Optimized)', 'Global LightGBM (ML)',
                     'Croston (Intermittent)', 'Ensemble (LGBM + ARIMA)']
    available_models = st.sidebar.multiselect(
        "Select Forecasting Engines:", ML_MODELS_ADV,
        default=['AutoARIMA (Optimized)', 'Global LightGBM (ML)', 'Ensemble (LGBM + ARIMA)']
    )
    calc_intervals = st.sidebar.checkbox("Generate Confidence Intervals (P10/P90)", value=True)

# ============================================================
#  UI: SIDEBAR — SKU CONSOLIDATION
# ============================================================
st.sidebar.markdown("---")
st.sidebar.subheader("🔄 SKU Consolidation")


def get_mapping_template() -> bytes:
    output = io.BytesIO()
    pd.DataFrame(columns=['Old SKU Code', 'New SKU Code']).to_excel(
        pd.ExcelWriter(output, engine='xlsxwriter'),  # noqa: SIM115
        index=False, sheet_name='SKU_Mapping'
    )
    # Use context manager properly
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        pd.DataFrame(columns=['Old SKU Code', 'New SKU Code']).to_excel(
            writer, index=False, sheet_name='SKU_Mapping'
        )
    return output.getvalue()


st.sidebar.download_button(
    label="Download SKU Mapping Template",
    data=get_mapping_template(),
    file_name="SKU_Mapping_Template.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
mapping_file = st.sidebar.file_uploader("Upload SKU Mapping Excel", type=['xlsx', 'xls'])

# ============================================================
#  UI: SIDEBAR — LICENSE & AUTHOR
# ============================================================
st.sidebar.markdown("---")
st.sidebar.markdown("### 📜 License & Info")
st.sidebar.info(
    "🐍 **Built with Python**\n\n"
    "This software is **open-source** and freely available for experimental use.\n\n"
    "💼 **For commercial use**, please contact:\n"
    "📧 [atillakdeniz@icloud.com](mailto:atillakdeniz@icloud.com)"
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='text-align:center;color:#888;font-size:12px;' title='a humble demand planner'>"
    "Created by <b>Atilla AKDENİZ</b></div>",
    unsafe_allow_html=True
)

# ============================================================
#  MAIN TITLE
# ============================================================
st.title("📊 Mini Forecast Tool (MFT)")
if app_mode == "Advanced Mode":
    st.caption("🚀 Running in Enterprise Demand Planning Mode")
else:
    st.caption("🛠️ Running in Basic Statistical Mode")

# ============================================================
#  SALES TEMPLATE DOWNLOAD
# ============================================================
def get_sales_template() -> bytes:
    output = io.BytesIO()
    template_df = pd.DataFrame({
        'SKU Code':        ['SKU-001', 'SKU-001', 'SKU-002', 'SKU-002'],
        'SKU Description': ['Sample Product A', 'Sample Product A', 'Sample Product B', 'Sample Product B'],
        'Date':            ['2023-01-01', '2023-02-01', '2023-01-01', '2023-02-01'],
        'Sales':           [150, 180, 0, 55],
    })
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        template_df.to_excel(writer, index=False, sheet_name='Sales_Data')
    return output.getvalue()


st.download_button(
    label="📥 Download Sales Data Template",
    data=get_sales_template(),
    file_name="MFT_Sales_Template.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# ============================================================
#  FORECASTING HELPER FUNCTIONS
# ============================================================

def get_demand_segment(ts: np.ndarray) -> str:
    non_zero = ts[ts > 0]
    if len(non_zero) == 0:
        return "Dead"
    adi        = len(ts) / len(non_zero)
    cv2        = (np.std(non_zero) / np.mean(non_zero)) ** 2 if len(non_zero) > 1 else 0
    volatility = np.std(ts) / np.mean(ts) if np.mean(ts) > 0 else 0
    if   adi <= 1.32 and cv2 <= 0.49: return "Smooth"
    elif adi <= 1.32 and cv2 >  0.49: return "Highly Volatile" if volatility > 1.0 else "Erratic"
    elif adi >  1.32 and cv2 <= 0.49: return "Intermittent"
    else:                              return "Lumpy"


def winsorize_series(ts: np.ndarray, limit: float = 3.0) -> np.ndarray:
    mean, std = np.mean(ts), np.std(ts)
    if std == 0:
        return ts
    z = (ts - mean) / std
    ts_capped = np.where(z >  limit, mean + limit * std,  ts)
    ts_capped = np.where(z < -limit, max(0, mean - limit * std), ts_capped)
    return ts_capped


def impute_stockouts(ts: np.ndarray, segment: str) -> np.ndarray:
    if segment in ("Smooth", "Erratic") and len(ts) >= 3:
        s = pd.Series(ts)
        rolling_mean = s.replace(0, np.nan).rolling(window=3, min_periods=1, center=True).mean()
        return np.where(ts == 0, rolling_mean.fillna(0), ts)
    return ts


def create_adv_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Month']      = df['Datetime'].dt.month
    df['Month_Sin']  = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_Cos']  = np.cos(2 * np.pi * df['Month'] / 12)
    df = df.sort_values(['SKU Code', 'Datetime'])
    for lag in [1, 2, 3, 12]:
        df[f'Lag_{lag}'] = df.groupby('SKU Code')['Sales'].shift(lag)
    df['Rolling_Mean_3'] = df.groupby('SKU Code')['Lag_1'].transform(
        lambda x: x.rolling(3, min_periods=1).mean()
    )
    df['Rolling_Std_3'] = df.groupby('SKU Code')['Lag_1'].transform(
        lambda x: x.rolling(3, min_periods=1).std().fillna(0)
    )
    df['Momentum'] = (df['Lag_1'] - df['Lag_2']) / (df['Lag_2'].abs() + 1e-5)
    return df


def calculate_adv_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.maximum(np.array(y_pred, dtype=float), 0)
    sum_true = np.sum(y_true)
    if sum_true == 0:
        wmape = 100.0 if np.sum(y_pred) > 0 else 0.0
        bias  = 100.0 if np.sum(y_pred) > 0 else 0.0
    else:
        wmape = np.sum(np.abs(y_true - y_pred)) / sum_true * 100
        bias  = (np.sum(y_pred) - sum_true) / sum_true * 100
    return min(wmape, 999.0), bias


def mape_loss_basic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-9)))


def croston_forecast(ts: np.ndarray, horizon: int) -> np.ndarray:
    n, alpha   = len(ts), 0.1
    last_q     = float(ts[0]) if ts[0] > 0 else 1.0
    last_p     = 1.0
    q_interval = 1
    for val in ts:
        if val > 0:
            last_q     = alpha * val + (1 - alpha) * last_q
            last_p     = alpha * q_interval + (1 - alpha) * last_p
            q_interval = 1
        else:
            q_interval += 1
    return np.repeat(last_q / last_p, horizon)


def safe_model_run(fn, fallback, error_log: list, sku: str, model: str):
    """
    Wraps a model callable. Returns fn() on success, fallback on failure.
    Appends a structured dict to error_log on failure.
    """
    try:
        return fn()
    except Exception as exc:
        error_log.append({
            'SKU Code':   sku,
            'Model':      model,
            'Error':      type(exc).__name__,
            'Detail':     str(exc)[:200],
        })
        return fallback


def to_excel_download(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Forecast_Results')
    return output.getvalue()


# ============================================================
#  FILE UPLOAD & VALIDATION
# ============================================================
uploaded_file = st.file_uploader("Upload Sales Data (Excel format)", type=['xlsx', 'xls'])

if uploaded_file is not None:

    # ── Load raw file ──────────────────────────────────────
    try:
        df_raw = pd.read_excel(uploaded_file)
    except Exception as exc:
        st.error(
            f"❌ Could not read the uploaded file: `{exc}`\n\n"
            "Please make sure it is a valid `.xlsx` or `.xls` file."
        )
        st.stop()

    # ── Validate columns ───────────────────────────────────
    is_valid, err_msg = validate_sales_file(df_raw.copy())
    if not is_valid:
        st.error(f"❌ **Invalid file format:**\n\n{err_msg}")
        st.stop()

    # ── Apply aliases & coerce types ───────────────────────
    df_raw = df_raw.rename(columns=COLUMN_ALIASES)
    if 'SKU Description' not in df_raw.columns:
        df_raw['SKU Description'] = "Unknown"

    df_raw['Datetime'] = pd.to_datetime(df_raw['Date'])
    df_raw['Sales']    = pd.to_numeric(df_raw['Sales'], errors='coerce').fillna(0).clip(lower=0)

    raw_row_count  = len(df_raw)
    global_max_date = df_raw['Datetime'].max()

    desc_map = (
        df_raw[['SKU Code', 'SKU Description']]
        .drop_duplicates(subset=['SKU Code'])
        .set_index('SKU Code')['SKU Description']
        .to_dict()
    )

    # ── SKU Consolidation (optional) ───────────────────────
    df = df_raw.copy()
    if mapping_file is not None:
        try:
            map_df = pd.read_excel(mapping_file)
        except Exception as exc:
            st.warning(f"⚠️ Could not read the mapping file (`{exc}`). Consolidation skipped.")
            map_df = None

        if map_df is not None:
            map_valid, map_err = validate_mapping_file(map_df.copy())
            if not map_valid:
                st.warning(f"⚠️ SKU Mapping file issue: {map_err}. Consolidation skipped.")
            else:
                map_df = map_df.rename(columns=MAPPING_ALIASES).dropna(subset=['Old SKU Code', 'New SKU Code'])
                mapping_dict = dict(zip(map_df['Old SKU Code'], map_df['New SKU Code']))
                df['SKU Code'] = df['SKU Code'].replace(mapping_dict)
                st.success(f"✅ SKUs consolidated — {len(mapping_dict):,} mapping(s) applied.")

    df = df.groupby(['SKU Code', 'Datetime'])['Sales'].sum().reset_index()
    df['SKU Description'] = df['SKU Code'].map(desc_map).fillna("Consolidated / Unknown")
    df = df.sort_values(['SKU Code', 'Datetime'])

    # ── SKU selector ───────────────────────────────────────
    ui_choices  = np.sort((df['SKU Code'].astype(str) + " | " + df['SKU Description'].astype(str)).unique())
    selected_ui = st.multiselect("🎯 Select SKUs to Forecast [Leave blank for ALL]:", ui_choices)

    # ── Guard: model list empty ────────────────────────────
    if not available_models:
        st.warning("⚠️ No models selected in the sidebar. Please select at least one model.")
        st.stop()

    if st.button("🚀 Run Forecast"):

        timer_placeholder = st.empty()
        start_time        = time.time()

        if selected_ui:
            selected_codes = [x.split(" | ")[0] for x in selected_ui]
            df = df[df['SKU Code'].isin(selected_codes)].copy()

        # Guard: selection resulted in empty frame
        if df.empty:
            st.error("❌ The selected SKUs have no data after filtering. Please revise your selection.")
            st.stop()

        progress_bar = st.progress(0)
        status_text  = st.empty()

        # ── Standardize monthly timelines ──────────────────
        all_skus = df['SKU Code'].unique()
        standardized_frames = []
        for sku in all_skus:
            grp = (
                df[df['SKU Code'] == sku]
                .set_index('Datetime')
                .resample('MS')['Sales'].sum()
                .fillna(0)
            )
            full_idx = pd.date_range(start=grp.index.min(), end=global_max_date, freq='MS')
            grp = grp.reindex(full_idx, fill_value=0).reset_index().rename(columns={'index': 'Datetime'})
            grp['SKU Code'] = sku
            standardized_frames.append(grp)

        df = pd.concat(standardized_frames, ignore_index=True)

        results, failed_skus, model_errors = [], [], []
        total_skus = len(all_skus)

        # ======================================================
        #  BASIC MODE
        # ======================================================
        if app_mode == "Basic Mode":
            future_dates = [global_max_date + pd.DateOffset(months=i) for i in range(1, forecast_horizon + 1)]

            for sku_idx, sku in enumerate(all_skus):
                elapsed    = int(time.time() - start_time)
                mins, secs = divmod(elapsed, 60)
                timer_placeholder.markdown(f"⏱️ **Elapsed Time:** {mins} min {secs} sec")

                group    = df[df['SKU Code'] == sku]
                sku_desc = desc_map.get(sku, 'Unknown')
                ts       = group['Sales'].values

                # ── Continuous sales rule ──────────────────
                if use_min_sales:
                    ts_series = pd.Series(ts)
                    if (ts_series > 0).rolling(window=min_sales_req).sum().max() < min_sales_req:
                        failed_skus.append({
                            'SKU Code': sku, 'SKU Description': sku_desc,
                            'Reason': 'Failed Continuous Sales Rule'
                        })
                        progress_bar.progress((sku_idx + 1) / total_skus)
                        continue

                # ── Training data window ───────────────────
                ts_data = ts[-(train_months + test_months):]
                if len(ts_data) <= test_months + 2:
                    failed_skus.append({
                        'SKU Code': sku, 'SKU Description': sku_desc,
                        'Reason': 'Insufficient Historical Data'
                    })
                    progress_bar.progress((sku_idx + 1) / total_skus)
                    continue

                train_y = ts_data[:-test_months]
                test_y  = ts_data[-test_months:]

                best_mape       = float('inf')
                best_model_name = "Not Found"
                best_forecast   = np.zeros(forecast_horizon)

                for m_name in available_models:
                    try:
                        preds   = np.zeros(test_months)
                        f_preds = np.zeros(forecast_horizon)

                        if m_name == 'Naive':
                            preds   = np.repeat(train_y[-1], test_months)
                            f_preds = np.repeat(train_y[-1], forecast_horizon)

                        elif m_name == 'Moving Average':
                            ma      = pd.Series(train_y).rolling(3, min_periods=1).mean().iloc[-1]
                            preds   = np.repeat(ma, test_months)
                            f_preds = np.repeat(ma, forecast_horizon)

                        elif m_name == 'Holt-Winters':
                            hw      = ExponentialSmoothing(train_y, trend='add').fit()
                            preds   = hw.forecast(test_months)
                            f_preds = hw.forecast(forecast_horizon)

                        elif m_name == 'SARIMA':
                            sarima  = SARIMAX(train_y, order=(1, 1, 1)).fit(disp=False)
                            preds   = sarima.forecast(steps=test_months)
                            s_full  = SARIMAX(ts_data, order=(1, 1, 1)).fit(disp=False)
                            f_preds = s_full.forecast(steps=forecast_horizon)

                        elif m_name in ('XGBoost', 'LightGBM', 'CatBoost'):
                            # Build lag features so tree models learn demand patterns,
                            # not just a monotone time index.
                            n_lags = 3

                            def _build_lag_features(series, n_lags=n_lags):
                                rows = []
                                for i in range(n_lags, len(series)):
                                    rows.append(list(series[i - n_lags:i]) + [series[i]])
                                lag_cols = [f'lag_{j+1}' for j in range(n_lags)]
                                return pd.DataFrame(rows, columns=lag_cols + ['target'])

                            feat_all = _build_lag_features(ts_data)
                            split    = len(train_y) - n_lags
                            if split <= 0:
                                raise ValueError("Not enough data for lag features.")

                            X_tr = feat_all.iloc[:split, :-1].values
                            y_tr = feat_all.iloc[:split, -1].values
                            X_te = feat_all.iloc[split:split + test_months, :-1].values

                            if   m_name == 'XGBoost':  mdl = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, verbosity=0)
                            elif m_name == 'LightGBM': mdl = lgb.LGBMRegressor(n_estimators=100, verbose=-1)
                            else:                      mdl = CatBoostRegressor(iterations=100, verbose=0)

                            mdl.fit(X_tr, y_tr)
                            preds = mdl.predict(X_te)

                            # Retrain on full window, then forecast autoregressively
                            feat_full = _build_lag_features(ts_data)
                            mdl.fit(feat_full.iloc[:, :-1].values, feat_full.iloc[:, -1].values)
                            buf          = list(ts_data[-n_lags:])
                            f_preds_list = []
                            for _ in range(forecast_horizon):
                                x_in  = np.array(buf[-n_lags:]).reshape(1, -1)
                                y_hat = float(mdl.predict(x_in)[0])
                                f_preds_list.append(max(0.0, y_hat))
                                buf.append(y_hat)
                            f_preds = np.array(f_preds_list)

                        elif m_name in ('Croston', 'SBA', 'TSB'):
                            preds   = croston_forecast(train_y, test_months)
                            f_preds = croston_forecast(ts_data, forecast_horizon)

                        else:
                            continue   # unknown model name — skip silently

                    except Exception as exc:
                        model_errors.append({
                            'SKU Code': sku,
                            'Model':    m_name,
                            'Error':    type(exc).__name__,
                            'Detail':   str(exc)[:200],
                        })
                        continue   # move on to next model; don't crash the run

                    if len(preds) == 0:
                        continue
                    mape = mape_loss_basic(test_y, preds) * 100
                    if mape < best_mape:
                        best_mape       = mape
                        best_model_name = m_name
                        best_forecast   = f_preds

                best_forecast = np.round(np.maximum(best_forecast, 0)).astype(int)

                if np.all(best_forecast == 0) or best_model_name == "Not Found":
                    failed_skus.append({
                        'SKU Code': sku, 'SKU Description': sku_desc,
                        'Reason': 'All models yielded Zero or Failed'
                    })
                    progress_bar.progress((sku_idx + 1) / total_skus)
                    continue

                # ── Minimum forecast threshold check ──────────
                if use_min_fc_threshold and min_fc_threshold > 0:
                    fc_monthly_avg = float(np.mean(best_forecast))
                    if fc_monthly_avg < min_fc_threshold:
                        failed_skus.append({
                            'SKU Code': sku, 'SKU Description': sku_desc,
                            'Reason': f'Below Min. Forecast Threshold (avg {fc_monthly_avg:,.0f} < {min_fc_threshold:,})'
                        })
                        progress_bar.progress((sku_idx + 1) / total_skus)
                        continue

                row_data = {
                    'SKU Code':             sku,
                    'SKU Description':      sku_desc,
                    'Selected Model':       best_model_name,
                    'Validation MAPE (%)':  round(best_mape, 2),
                    'V. Accuracy (%)':      round(max(0, 100 - best_mape), 2),
                    'Volume':               float(ts.sum()),
                }
                for i, f_date in enumerate(future_dates):
                    col = f"{f_date.year}-{f_date.month:02d} Forecast"
                    row_data[col] = int(best_forecast[i]) if i < len(best_forecast) else 0

                results.append(row_data)
                progress_bar.progress((sku_idx + 1) / total_skus)


        # ======================================================
        #  ADVANCED MODE
        # ======================================================
        elif app_mode == "Advanced Mode":
            status_text.text("⚙️ Preprocessing Data (Outliers, Stockouts, Segmentation)...")
            segments: dict[str, str] = {}
            for sku in all_skus:
                mask    = df['SKU Code'] == sku
                ts      = df.loc[mask, 'Sales'].values
                segment = get_demand_segment(ts)
                segments[sku] = segment
                if handle_outliers:  ts = winsorize_series(ts)
                if handle_stockouts: ts = impute_stockouts(ts, segment)
                df.loc[mask, 'Sales'] = ts

            status_text.text("🧠 Training Global LightGBM Model across all SKUs...")

            df_feat   = create_adv_features(df).dropna()
            cutoff    = global_max_date - pd.DateOffset(months=3)
            train_df  = df_feat[df_feat['Datetime'] <= cutoff]
            test_df   = df_feat[df_feat['Datetime'] >  cutoff]
            FEATURES  = ['Lag_1','Lag_2','Lag_3','Lag_12',
                         'Rolling_Mean_3','Rolling_Std_3','Momentum',
                         'Month_Sin','Month_Cos']

            global_lgb  = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, verbose=-1)
            lgb_p10 = lgb_p90 = None

            if not train_df.empty:
                global_lgb.fit(train_df[FEATURES], train_df['Sales'])
                if calc_intervals:
                    lgb_p10 = lgb.LGBMRegressor(objective='quantile', alpha=0.1, n_estimators=100, verbose=-1)
                    lgb_p90 = lgb.LGBMRegressor(objective='quantile', alpha=0.9, n_estimators=100, verbose=-1)
                    lgb_p10.fit(train_df[FEATURES], train_df['Sales'])
                    lgb_p90.fit(train_df[FEATURES], train_df['Sales'])
            else:
                st.warning("⚠️ Not enough training data to fit the Global LightGBM model. LightGBM engine will be skipped.")

            status_text.text("⏳ Running Local Models & Compiling Forecasts...")
            future_dates = [global_max_date + pd.DateOffset(months=i) for i in range(1, forecast_horizon + 1)]

            for sku_idx, sku in enumerate(all_skus):
                elapsed    = int(time.time() - start_time)
                mins, secs = divmod(elapsed, 60)
                timer_placeholder.markdown(f"⏱️ **Elapsed Time:** {mins} min {secs} sec")

                sku_df   = df[df['SKU Code'] == sku].copy()
                ts       = sku_df['Sales'].values
                sku_desc = desc_map.get(sku, 'Unknown')

                if use_min_sales:
                    ts_s = pd.Series(ts)
                    if (ts_s > 0).rolling(window=min_sales_req).sum().max() < min_sales_req:
                        failed_skus.append({
                            'SKU Code': sku, 'SKU Description': sku_desc,
                            'Reason': 'Failed Continuous Sales Rule'
                        })
                        progress_bar.progress((sku_idx + 1) / total_skus)
                        continue

                if len(ts) < 15:
                    failed_skus.append({
                        'SKU Code': sku, 'SKU Description': sku_desc,
                        'Reason': 'Insufficient Historical Data (< 15 months)'
                    })
                    progress_bar.progress((sku_idx + 1) / total_skus)
                    continue

                train_y = ts[:-3]
                test_y  = ts[-3:]

                best_wmape      = float('inf')
                best_model_name = "Naive (Fallback)"
                final_forecast  = np.repeat(float(ts[-1]) if len(ts) > 0 else 0.0, forecast_horizon)

                metrics_dict:   dict[str, tuple[float, float]] = {}
                forecasts_dict: dict[str, np.ndarray]          = {}

                # Naive baseline (always computed for FVA)
                preds_naive    = np.repeat(float(train_y[-1]), 3)
                f_preds_naive  = np.repeat(float(ts[-1]), forecast_horizon)
                metrics_dict['Naive'] = calculate_adv_metrics(test_y, preds_naive)

                # ── Global LightGBM ────────────────────────
                if 'Global LightGBM (ML)' in available_models and not train_df.empty:
                    sku_test_feats = test_df[test_df['SKU Code'] == sku][FEATURES]
                    if not sku_test_feats.empty:
                        def _lgb_run():
                            preds_lgb = global_lgb.predict(sku_test_feats)
                            metrics_dict['Global LightGBM (ML)'] = calculate_adv_metrics(test_y, preds_lgb)
                            temp_df = sku_df.copy()
                            for fi in range(forecast_horizon):
                                t_feat   = create_adv_features(temp_df).iloc[-1:]
                                valid_f  = [c for c in FEATURES if c in t_feat.columns]
                                if len(valid_f) < len(FEATURES):
                                    raise ValueError("Missing lag feature columns during autoregression.")
                                nxt      = max(0.0, float(global_lgb.predict(t_feat[valid_f])[0]))
                                new_row  = pd.DataFrame({'SKU Code': [sku], 'Datetime': [future_dates[fi]], 'Sales': [nxt]})
                                temp_df  = pd.concat([temp_df, new_row], ignore_index=True)
                            forecasts_dict['Global LightGBM (ML)'] = temp_df['Sales'].iloc[-forecast_horizon:].values
                        safe_model_run(_lgb_run, None, model_errors, sku, 'Global LightGBM (ML)')

                # ── AutoARIMA ──────────────────────────────
                if 'AutoARIMA (Optimized)' in available_models and segments.get(sku) in ("Smooth","Highly Volatile","Erratic"):
                    def _arima_run():
                        arima = pm.auto_arima(
                            train_y, seasonal=True, m=12, stepwise=True,
                            suppress_warnings=True, error_action="ignore",
                            max_p=2, max_q=2
                        )
                        preds_arima = arima.predict(n_periods=3)
                        metrics_dict['AutoARIMA (Optimized)'] = calculate_adv_metrics(test_y, preds_arima)
                        arima_full = pm.auto_arima(
                            ts, seasonal=True, m=12, stepwise=True,
                            suppress_warnings=True, error_action="ignore"
                        )
                        forecasts_dict['AutoARIMA (Optimized)'] = arima_full.predict(n_periods=forecast_horizon)
                    safe_model_run(_arima_run, None, model_errors, sku, 'AutoARIMA (Optimized)')

                # ── Croston ────────────────────────────────
                if 'Croston (Intermittent)' in available_models and segments.get(sku) in ("Intermittent","Lumpy"):
                    def _croston_run():
                        preds_c = croston_forecast(train_y, 3)
                        metrics_dict['Croston (Intermittent)'] = calculate_adv_metrics(test_y, preds_c)
                        forecasts_dict['Croston (Intermittent)'] = croston_forecast(ts, forecast_horizon)
                    safe_model_run(_croston_run, None, model_errors, sku, 'Croston (Intermittent)')

                # ── Ensemble ───────────────────────────────
                if (
                    'Ensemble (LGBM + ARIMA)' in available_models
                    and 'Global LightGBM (ML)' in forecasts_dict
                    and 'AutoARIMA (Optimized)' in forecasts_dict
                ):
                    def _ensemble_run():
                        ens_preds = (
                            global_lgb.predict(sku_test_feats)
                            + np.array(metrics_dict.get('AutoARIMA (Optimized)', (0,))[0])
                        ) / 2
                        # Re-derive component preds for ensemble validation metric
                        lgb_val  = global_lgb.predict(sku_test_feats) if not sku_test_feats.empty else preds_naive
                        arima_val_key = 'AutoARIMA (Optimized)'
                        # Use stored metric average as proxy
                        ens_fc   = (forecasts_dict['Global LightGBM (ML)'] + forecasts_dict['AutoARIMA (Optimized)']) / 2
                        # Validation: average val predictions
                        ens_val  = (lgb_val + croston_forecast(train_y, 3)) / 2 if 'Croston (Intermittent)' in forecasts_dict else lgb_val
                        metrics_dict['Ensemble (LGBM + ARIMA)'] = calculate_adv_metrics(test_y, ens_val)
                        forecasts_dict['Ensemble (LGBM + ARIMA)'] = ens_fc
                    safe_model_run(_ensemble_run, None, model_errors, sku, 'Ensemble (LGBM + ARIMA)')

                # ── Pick winner ────────────────────────────
                for m_name, mets in metrics_dict.items():
                    if m_name in available_models and mets[0] < best_wmape:
                        best_wmape      = mets[0]
                        best_model_name = m_name
                        final_forecast  = forecasts_dict.get(m_name, f_preds_naive)

                fva_score  = metrics_dict['Naive'][0] - best_wmape
                p10_vals   = np.zeros(forecast_horizon)
                p90_vals   = np.zeros(forecast_horizon)

                if calc_intervals and lgb_p10 is not None and 'Global LightGBM (ML)' in forecasts_dict:
                    # Use the last horizon rows of the autoregression temp_df features
                    temp_feat_df = create_adv_features(
                        pd.concat([sku_df, pd.DataFrame({
                            'SKU Code':  [sku] * forecast_horizon,
                            'Datetime':  future_dates,
                            'Sales':     final_forecast.tolist()
                        })], ignore_index=True)
                    ).iloc[-forecast_horizon:]
                    valid_cols = [c for c in FEATURES if c in temp_feat_df.columns]
                    if valid_cols:
                        def _intervals():
                            p10_vals[:] = np.maximum(0, lgb_p10.predict(temp_feat_df[valid_cols]))
                            p90_vals[:] = np.maximum(final_forecast, lgb_p90.predict(temp_feat_df[valid_cols]))
                        safe_model_run(_intervals, None, model_errors, sku, 'P10/P90 Intervals')

                final_forecast = np.round(np.maximum(final_forecast, 0)).astype(int)

                if np.all(final_forecast == 0):
                    failed_skus.append({
                        'SKU Code': sku, 'SKU Description': sku_desc,
                        'Reason': 'All Forecasts yielded Zero'
                    })
                    progress_bar.progress((sku_idx + 1) / total_skus)
                    continue

                # ── Minimum forecast threshold check ──────────
                if use_min_fc_threshold and min_fc_threshold > 0:
                    fc_monthly_avg = float(np.mean(final_forecast))
                    if fc_monthly_avg < min_fc_threshold:
                        failed_skus.append({
                            'SKU Code': sku, 'SKU Description': sku_desc,
                            'Reason': f'Below Min. Forecast Threshold (avg {fc_monthly_avg:,.0f} < {min_fc_threshold:,})'
                        })
                        progress_bar.progress((sku_idx + 1) / total_skus)
                        continue

                row_data = {
                    'SKU Code':       sku,
                    'SKU Description': sku_desc,
                    'Demand Segment': segments[sku],
                    'Winning Engine': best_model_name,
                    'FVA (%)':        round(fva_score, 1),
                    'Val. WMAPE (%)': round(best_wmape, 1),
                    'Val. Bias (%)':  round(metrics_dict[best_model_name][1], 1),
                    'Volume':         float(ts.sum()),
                }
                for i, f_date in enumerate(future_dates):
                    mon = f"{f_date.year}-{f_date.month:02d}"
                    row_data[f"{mon} P10"]        = int(p10_vals[i]) if calc_intervals else 0
                    row_data[f"{mon} Final (P50)"] = int(final_forecast[i])
                    row_data[f"{mon} P90"]         = int(p90_vals[i]) if calc_intervals else 0

                results.append(row_data)
                progress_bar.progress((sku_idx + 1) / total_skus)


        # ======================================================
        #  COMMON RESULTS RENDER
        # ======================================================
        final_elapsed = int(time.time() - start_time)
        f_mins, f_secs = divmod(final_elapsed, 60)
        timer_placeholder.success(f"✅ **Process Completed in:** {f_mins} min {f_secs} sec")
        status_text.empty()

        res_df   = pd.DataFrame(results)
        fail_df  = pd.DataFrame(failed_skus)
        error_df = pd.DataFrame(model_errors)

        st.markdown("---")
        st.markdown("### 📝 Execution Summary")
        st.write(f"- **Total Rows Analyzed:** {raw_row_count:,}")
        st.write(f"- **Total Unique SKUs Checked:** {total_skus:,}")
        st.write(f"- **Successfully Forecasted SKUs:** {len(results):,}")

        if not fail_df.empty:
            st.write(f"- **SKUs Excluded from Forecast:** {len(failed_skus):,}")
            st.markdown("#### Exclusion Reasons Summary:")

            # Group all threshold exclusions into one summary line
            threshold_mask  = fail_df['Reason'].str.startswith('Below Min. Forecast Threshold')
            threshold_count = int(threshold_mask.sum())
            other_fail_df   = fail_df[~threshold_mask]

            for reason, count in other_fail_df['Reason'].value_counts().items():
                st.markdown(f"  - **{count} SKUs**: *{reason}*")

            if threshold_count > 0:
                st.markdown(
                    f"  - **{threshold_count} SKUs**: "
                    f"*Below Min. Forecast Threshold (< {min_fc_threshold:,} monthly avg) — "
                    f"see exclusion log below for details*"
                )

        # Model-level error log (collapsible, only shown if errors occurred)
        if not error_df.empty:
            with st.expander(f"⚠️ {len(error_df)} model-level error(s) were caught and skipped — click to inspect"):
                st.dataframe(error_df, use_container_width=True)

        if not res_df.empty:
            st.markdown("---")

            # ── BASIC results ──────────────────────────────
            if app_mode == "Basic Mode":
                res_df['Weighted Accuracy'] = res_df['V. Accuracy (%)'] * res_df['Volume']
                portfolio_accuracy = res_df['Weighted Accuracy'].sum() / (res_df['Volume'].sum() + 1e-5)
                st.success(f"🎯 **Portfolio Overall Validation Accuracy:** {round(portfolio_accuracy, 2)}%")

                model_stats = res_df.groupby('Selected Model').apply(
                    lambda x: pd.Series({
                        'SKU_Count':    len(x),
                        'Model_WMAPE': (x['Validation MAPE (%)'] * x['Volume']).sum() / (x['Volume'].sum() + 1e-5)
                    })
                ).reset_index()
                model_stats['V. Accuracy (%)'] = (100 - model_stats['Model_WMAPE']).clip(lower=0).round(2)

                fig_model = px.pie(
                    model_stats, values='SKU_Count', names='Selected Model',
                    custom_data=['V. Accuracy (%)'],
                    title="Model Contribution to Portfolio", hole=0.3
                )
                fig_model.update_traces(hovertemplate="<b>%{label}</b><br>%{value} SKUs (%{percent})<br>Model Accuracy: %{customdata[0]}%")
                st.plotly_chart(fig_model, use_container_width=True)

                display_df = res_df.drop(columns=['Volume', 'Weighted Accuracy']).copy()
                st.download_button(
                    "📥 Download Results as Excel",
                    data=to_excel_download(display_df),
                    file_name="Basic_Forecast_Results.xlsx"
                )

                for col in ['Validation MAPE (%)', 'V. Accuracy (%)']:
                    display_df[col] = display_df[col].fillna(0).astype(str).str.replace('.', ',')
                for col in [c for c in display_df.columns if 'Forecast' in c]:
                    display_df[col] = display_df[col].fillna(0).apply(lambda x: f"{int(float(x)):,}".replace(',', '.'))
                st.dataframe(display_df)

            # ── ADVANCED results ───────────────────────────
            elif app_mode == "Advanced Mode":
                if not calc_intervals:
                    res_df = res_df.loc[:, ~res_df.columns.str.contains('P10|P90')]

                st.markdown("### 📊 Interactive Planner View (Editable)")
                st.info("💡 **Planner Override:** You can manually edit the Final (P50) forecast columns below before exporting to ERP.")

                display_cols = [c for c in res_df.columns if c != 'Volume']
                edited_df    = st.data_editor(res_df[display_cols], num_rows="dynamic", use_container_width=True)

                st.download_button(
                    "📥 Export Final Plan to Excel",
                    data=to_excel_download(edited_df),
                    file_name="Enterprise_Demand_Plan.xlsx"
                )

                st.markdown("---")
                st.markdown("### 📈 Pipeline Diagnostics")
                col1, col2, col3 = st.columns(3)
                port_wmape   = (res_df['Val. WMAPE (%)'] * res_df['Volume']).sum() / (res_df['Volume'].sum() + 1e-5)
                positive_fva = len(res_df[res_df['FVA (%)'] > 0])

                col1.metric("Total SKUs Processed",  f"{len(res_df):,}")
                col2.metric("Portfolio WMAPE",        f"{round(port_wmape, 1)}%")
                col3.metric("SKUs with Positive FVA", f"{positive_fva} ({round(positive_fva / max(len(res_df), 1) * 100)}%)")

                col_a, col_b = st.columns(2)
                fig_eng = px.pie(res_df, names='Winning Engine', title='Engine Allocation Strategy',
                                 hole=0.3, color_discrete_sequence=px.colors.qualitative.Set2)
                col_a.plotly_chart(fig_eng, use_container_width=True)
                fig_seg = px.pie(res_df, names='Demand Segment', title='Enterprise Demand Segmentation',
                                 hole=0.3, color_discrete_sequence=px.colors.qualitative.Pastel)
                col_b.plotly_chart(fig_seg, use_container_width=True)

        else:
            st.warning("⚠️ No results were generated for the selected criteria. Check the exclusion log below.")

        if not fail_df.empty:
            st.markdown("---")
            with st.expander("🔍 Show Detailed Exclusion Log (Collapsible)"):
                log_text = "--- FORECAST EXCLUSION LOG ---\n"
                for _, r in fail_df.iterrows():
                    log_text += f"{r['SKU Code']} | {r['SKU Description']} -> {r['Reason']}\n"
                st.code(log_text, language='text')
