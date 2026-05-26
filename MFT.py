import streamlit as st
import pandas as pd
import numpy as np
import io
import time
import warnings
import holidays
import shap

from joblib import Parallel, delayed
import plotly.express as px
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import pmdarima as pm

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings('ignore')
st.set_page_config(page_title="Mini Forecast Tool (MFT)", layout="wide")

# ==========================================
#    1. DATA LOADING & ALIGNMENT (v5.2)
# ==========================================
@st.cache_data(show_spinner=False)
def load_and_align_data(df_raw):
    df_raw = df_raw.rename(columns={'SKU Kodu': 'SKU Code', 'SKU Tanım': 'SKU Description',
                                    'Tarih': 'Date', 'Satış KG': 'Sales'})
    if 'SKU Description' not in df_raw.columns:
        df_raw['SKU Description'] = "Unknown"
    df_raw['Datetime'] = pd.to_datetime(df_raw['Date'])

    desc_map = (df_raw[['SKU Code', 'SKU Description']]
                .drop_duplicates(subset=['SKU Code'])
                .set_index('SKU Code')['SKU Description'].to_dict())

    df_grouped = df_raw.groupby(['SKU Code', 'Datetime'])['Sales'].sum().reset_index()
    global_max_date = df_grouped['Datetime'].max()
    all_skus = df_grouped['SKU Code'].unique()

    aligned_frames = []
    for sku in all_skus:
        group = (df_grouped[df_grouped['SKU Code'] == sku]
                 .set_index('Datetime')
                 .resample('MS')['Sales'].sum().fillna(0))
        full_idx = pd.date_range(start=group.index.min(), end=global_max_date, freq='MS')
        group = group.reindex(full_idx, fill_value=0).reset_index().rename(columns={'index': 'Datetime'})
        group['SKU Code'] = sku
        aligned_frames.append(group)

    df_aligned = pd.concat(aligned_frames, ignore_index=True)
    df_aligned['SKU Description'] = df_aligned['SKU Code'].map(desc_map).fillna("Unknown")
    return df_aligned, desc_map, global_max_date

# ==========================================
#    2. LEAKAGE-FREE TREATMENTS & SEGMENTATION (v5.2)
# ==========================================
def get_robust_segment(ts):
    if len(ts) == 0:
        return "Dead"
    non_zero = ts[ts > 0]
    if len(non_zero) == 0:
        return "Dead"
    adi = len(ts) / len(non_zero)
    cv2 = (np.std(non_zero) / np.mean(non_zero)) ** 2 if len(non_zero) > 1 else 0
    zero_ratio = len(ts[ts == 0]) / len(ts)
    if zero_ratio >= 0.4:
        return "Highly Intermittent" if cv2 > 0.5 else "Intermittent"
    elif adi <= 1.32 and cv2 <= 0.49:
        return "Smooth"
    elif adi <= 1.32 and cv2 > 0.49:
        return "Volatile"
    else:
        return "Lumpy"

def expanding_winsorize(ts, limit=3):
    ts_s = pd.Series(ts)
    if len(ts_s) < 3:
        return ts
    roll_mean = ts_s.expanding().mean().shift(1).fillna(ts_s.iloc[0])
    roll_std = ts_s.expanding().std().shift(1).fillna(0)
    z_scores = np.where(roll_std > 0, (ts_s - roll_mean) / roll_std, 0)
    capped = np.where(z_scores > limit, roll_mean + limit * roll_std, ts)
    return np.where(z_scores < -limit, np.maximum(0, roll_mean - limit * roll_std), capped)

def expanding_impute(ts, segment):
    if segment in ["Smooth", "Volatile"] and len(ts) >= 3:
        ts_series = pd.Series(ts)
        roll_mean = ts_series.replace(0, np.nan).expanding().mean().shift(1)
        return np.where((ts == 0) & roll_mean.notna(), roll_mean, ts)
    return ts

# ==========================================
#    3. SINGLE SOURCE FEATURE PIPELINE (v5.2)
# ==========================================
def generate_features(df):
    df = df.copy()
    df['Quarter'] = df['Datetime'].dt.quarter
    df['Month'] = df['Datetime'].dt.month
    df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)

    tr_holidays = holidays.Turkey()
    workdays, is_holiday = [], []
    for dt in df['Datetime']:
        start = dt.replace(day=1).date()
        end = (dt + pd.offsets.MonthEnd(0)).date()
        workdays.append(np.busday_count(start, end))
        is_holiday.append(int(dt.date() in tr_holidays))

    df['WorkingDays'] = workdays
    df['IsHoliday'] = is_holiday
    df['Ramadan_Flag'] = (df['Datetime'].dt.month.isin([3, 4])).astype(int)
    df['BlackFriday'] = (df['Datetime'].dt.month == 11).astype(int)

    df = df.sort_values(['SKU Code', 'Datetime'])
    for lag in [1, 2, 3, 6, 12]:
        df[f'Lag_{lag}'] = df.groupby('SKU Code')['Sales'].shift(lag)

    df['Rolling_Mean_3'] = df.groupby('SKU Code')['Lag_1'].transform(
        lambda x: x.rolling(3, min_periods=1).mean())
    df['Rolling_Std_3'] = df.groupby('SKU Code')['Lag_1'].transform(
        lambda x: x.rolling(3, min_periods=1).std().fillna(0))
    df['Momentum'] = (df['Lag_1'] - df['Lag_2']) / (df['Lag_2'].abs() + 1e-6)

    base_cols = ['SKU Code', 'SKU Description', 'Datetime', 'Sales', 'Demand Segment']
    feature_registry = [col for col in df.columns if col not in base_cols]
    return df, feature_registry

# ==========================================
#    4. METRICS & INTERMITTENT MODELS (v5.2)
# ==========================================
def calculate_ts_cv_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.maximum(0, np.array(y_pred))
    sum_true = np.sum(y_true)
    if sum_true == 0:
        return (100.0 if np.sum(y_pred) > 0 else 0.0), (100.0 if np.sum(y_pred) > 0 else 0.0)
    wmape = min(999.0, np.sum(np.abs(y_true - y_pred)) / sum_true * 100)
    bias = (np.sum(y_pred) - sum_true) / sum_true * 100
    return wmape, bias

def mape_loss_basic(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-9))

def propagate_uncertainty(preds, residuals):
    residuals = residuals[~np.isnan(residuals)]
    if len(residuals) == 0:
        sigma = np.nanstd(preds) * 0.1 if len(preds) > 0 else 1.0
    else:
        sigma = np.nanstd(residuals)
        if np.isnan(sigma) or sigma == 0:
            sigma = np.nanstd(preds) * 0.1 if len(preds) > 0 else 1.0
    lowers, uppers = [], []
    for h, pred in enumerate(preds, start=1):
        dynamic_sigma = sigma * np.sqrt(h)
        low = max(0, pred - 1.645 * dynamic_sigma)
        high = pred + 1.645 * dynamic_sigma
        if np.isnan(low):
            low = 0.0
        if np.isnan(high):
            high = 0.0
        lowers.append(low)
        uppers.append(high)
    return np.array(lowers), np.array(uppers)

def calculate_shap_importance(model, X_sample, feature_names):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    importance = np.abs(shap_values).mean(axis=0)
    return pd.DataFrame({'Feature': feature_names,
                         'Importance': importance}).sort_values('Importance', ascending=False)

def croston_classic(ts, horizon=1, alpha=0.1):
    ts = np.array(ts)
    demand = ts[ts > 0]
    if len(demand) == 0:
        return np.zeros(horizon)
    intervals, last = [], 0
    for i, val in enumerate(ts):
        if val > 0:
            intervals.append(i - last if last != 0 else 1)
            last = i
    z, p = demand[0], intervals[0]
    for d, inter in zip(demand[1:], intervals[1:]):
        z = z + alpha * (d - z)
        p = p + alpha * (inter - p)
    return np.repeat(z / max(p, 1.0), horizon)

def tsb_forecast(ts, horizon=1, alpha_p=0.2, alpha_d=0.2):
    ts = np.array(ts)
    if len(ts) == 0:
        return np.zeros(horizon)
    p, z = (1 if ts[0] > 0 else 0), (ts[0] if ts[0] > 0 else 0)
    for val in ts:
        occurrence = 1 if val > 0 else 0
        p = alpha_p * occurrence + (1 - alpha_p) * p
        if occurrence:
            z = alpha_d * val + (1 - alpha_d) * z
    return np.repeat(p * z, horizon)

def to_excel_download(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Forecast_Results')
    return output.getvalue()

# ==========================================
#    5. PARALLEL WORKER (ADVANCED)
# ==========================================
def process_single_sku(sku, df, desc_map, segments, test_df, global_lgb,
                       available_models, forecast_horizon, feature_registry,
                       future_dates, calc_intervals, use_min_sales, min_sales_req,
                       val_months):
    try:
        sku_df = df[df['SKU Code'] == sku].copy()
        ts = sku_df['Sales'].values
        sku_desc = desc_map.get(sku, 'Unknown')

        if use_min_sales:
            if (pd.Series(ts) > 0).rolling(window=min_sales_req).sum().max() < min_sales_req:
                return {'status': 'failed',
                        'data': {'SKU Code': sku, 'SKU Description': sku_desc,
                                 'Reason': f'Failed Continuous Sales Rule ({min_sales_req}m)'}}

        if len(ts) < (15 if val_months <= 3 else val_months + 12):
            return {'status': 'failed',
                    'data': {'SKU Code': sku, 'SKU Description': sku_desc,
                             'Reason': f'Insufficient History (<{val_months+12}m)'}}

        train_y, test_y = ts[:-val_months], ts[-val_months:]
        best_wmape, best_model_name = float('inf'), "Naive (Baseline)"
        final_forecast = np.repeat(train_y[-1] if len(train_y) > 0 else 0, forecast_horizon)
        metrics_dict, forecasts_dict, residuals_dict = {}, {}, {}

        preds_naive = np.repeat(train_y[-1], val_months)
        metrics_dict['Naive'] = calculate_ts_cv_metrics(test_y, preds_naive)
        residuals_dict['Naive'] = test_y - preds_naive

        # Global LightGBM
        if 'Global LightGBM (ML)' in available_models and not test_df[test_df['SKU Code'] == sku].empty:
            sku_test_feat = test_df[test_df['SKU Code'] == sku][feature_registry]
            if len(sku_test_feat) == val_months:
                preds_lgb = global_lgb.predict(sku_test_feat)
                metrics_dict['Global LightGBM (ML)'] = calculate_ts_cv_metrics(test_y, preds_lgb)
                residuals_dict['Global LightGBM (ML)'] = test_y - preds_lgb

                temp_df = sku_df.copy()
                for i in range(forecast_horizon):
                    temp_df, _ = generate_features(temp_df)
                    temp_feat = temp_df.iloc[-1:][feature_registry]
                    next_pred = max(0, global_lgb.predict(temp_feat)[0])
                    next_row = pd.DataFrame({'SKU Code': [sku],
                                             'Datetime': [future_dates[i]],
                                             'Sales': [next_pred]})
                    temp_df = pd.concat([temp_df, next_row], ignore_index=True)
                forecasts_dict['Global LightGBM (ML)'] = temp_df['Sales'].iloc[-forecast_horizon:].values

        # AutoARIMA
        if 'AutoARIMA (Optimized)' in available_models and segments[sku] in ["Smooth", "Volatile"]:
            try:
                arima = pm.auto_arima(train_y, stepwise=True, max_p=1, max_q=1, max_order=2,
                                      seasonal=False, suppress_warnings=True, error_action="ignore")
                preds_arima = arima.predict(n_periods=val_months)
                metrics_dict['AutoARIMA (Optimized)'] = calculate_ts_cv_metrics(test_y, preds_arima)
                residuals_dict['AutoARIMA (Optimized)'] = test_y - preds_arima

                arima_full = pm.auto_arima(ts, stepwise=True, max_p=1, max_q=1, max_order=2,
                                           seasonal=False, suppress_warnings=True, error_action="ignore")
                forecasts_dict['AutoARIMA (Optimized)'] = arima_full.predict(n_periods=forecast_horizon)
            except:
                pass

        # Croston & TSB for intermittent segments
        if segments[sku] in ["Intermittent", "Highly Intermittent", "Lumpy"]:
            if 'Croston' in available_models:
                p_cr = croston_classic(train_y, val_months)
                metrics_dict['Croston'] = calculate_ts_cv_metrics(test_y, p_cr)
                residuals_dict['Croston'] = test_y - p_cr
                forecasts_dict['Croston'] = croston_classic(ts, forecast_horizon)
            if 'TSB' in available_models:
                p_tsb = tsb_forecast(train_y, val_months)
                metrics_dict['TSB'] = calculate_ts_cv_metrics(test_y, p_tsb)
                residuals_dict['TSB'] = test_y - p_tsb
                forecasts_dict['TSB'] = tsb_forecast(ts, forecast_horizon)

        # Ensemble (LGBM + ARIMA)
        if ('Ensemble (LGBM + ARIMA)' in available_models and
            'Global LightGBM (ML)' in forecasts_dict and
            'AutoARIMA (Optimized)' in forecasts_dict):
            preds_ens = (preds_lgb + preds_arima) / 2
            metrics_dict['Ensemble (LGBM + ARIMA)'] = calculate_ts_cv_metrics(test_y, preds_ens)
            forecasts_dict['Ensemble (LGBM + ARIMA)'] = (
                forecasts_dict['Global LightGBM (ML)'] + forecasts_dict['AutoARIMA (Optimized)']
            ) / 2

        # Select best model
        for m_name, mets in metrics_dict.items():
            if mets[0] < best_wmape:
                best_wmape = mets[0]
                best_model_name = m_name
                final_forecast = forecasts_dict.get(m_name,
                    np.repeat(train_y[-1] if len(train_y) > 0 else 0, forecast_horizon))

        fva_score = metrics_dict['Naive'][0] - best_wmape
        final_forecast = np.round(np.maximum(0, final_forecast)).astype(int)

        if np.all(final_forecast == 0):
            return {'status': 'failed',
                    'data': {'SKU Code': sku, 'SKU Description': sku_desc,
                             'Reason': 'Forecast is Absolute Zero'}}

        p10_vals, p90_vals = np.zeros(forecast_horizon), np.zeros(forecast_horizon)
        if calc_intervals and best_model_name in residuals_dict:
            p10_vals, p90_vals = propagate_uncertainty(final_forecast,
                                                       residuals_dict[best_model_name])

        row_data = {
            'SKU Code': sku,
            'SKU Description': sku_desc,
            'Demand Segment': segments[sku],
            'Winning Engine': best_model_name,
            'FVA (%)': round(fva_score, 1),
            'Val. WMAPE (%)': round(best_wmape, 1),
            'Val. Bias (%)': round(metrics_dict[best_model_name][1], 1),
            'Volume': ts.sum()
        }
        for i, f_date in enumerate(future_dates):
            mon_str = f"{f_date.year}-{f_date.month:02d}"
            row_data[f"{mon_str}_Original"] = final_forecast[i]
            row_data[f"{mon_str} Final (P50)"] = final_forecast[i]
            if calc_intervals:
                row_data[f"{mon_str} P10"] = int(p10_vals[i])
                row_data[f"{mon_str} P90"] = int(p90_vals[i])

        return {'status': 'success', 'data': row_data}
    except Exception as e:
        return {'status': 'failed',
                'data': {'SKU Code': sku, 'SKU Description': 'Unknown',
                         'Reason': f'System Crash: {str(e)}'}}

# ==========================================
#    6. UI & ORCHESTRATION
# ==========================================
if 'pipeline_run' not in st.session_state:
    st.session_state.pipeline_run = False
if 'res_df' not in st.session_state:
    st.session_state.res_df = pd.DataFrame()
if 'fail_df' not in st.session_state:
    st.session_state.fail_df = pd.DataFrame()
if 'filtered_df' not in st.session_state:
    st.session_state.filtered_df = pd.DataFrame()
if 'train_df' not in st.session_state:
    st.session_state.train_df = pd.DataFrame()
if 'global_lgb' not in st.session_state:
    st.session_state.global_lgb = None
if 'feature_registry' not in st.session_state:
    st.session_state.feature_registry = []
if 'raw_row_count' not in st.session_state:
    st.session_state.raw_row_count = 0

st.sidebar.header("🕹️ Operation Mode")
app_mode = st.sidebar.radio("Select Engine:", ["Basic Mode", "Advanced Enterprise Mode"], index=1)
st.sidebar.markdown("---")

if app_mode == "Basic Mode":
    st.sidebar.header("⚙️ Basic Parameters")
    train_months = st.sidebar.number_input("Max Training Period (Months):", min_value=12, max_value=60, value=36)
    test_months = st.sidebar.slider("Validation Period (Months):", 1, 12, 3)
    forecast_horizon = st.sidebar.slider("Forecast Horizon (Months):", 1, 24, 6)

    use_min_sales = st.sidebar.checkbox("Enable Min Continuous Sales Rule", value=True)
    min_sales_req = st.sidebar.slider("Min Continuous Sales (Months):", 1, 12, 6) if use_min_sales else 1

    ml_models_basic = ['Naive', 'Moving Average', 'Holt-Winters', 'SARIMA',
                       'XGBoost', 'LightGBM', 'CatBoost', 'Croston', 'TSB']
    available_models = st.sidebar.multiselect("Select Models to Test:", ml_models_basic,
                                             default=['Moving Average', 'Holt-Winters', 'SARIMA', 'Croston', 'TSB'])

elif app_mode == "Advanced Enterprise Mode":
    st.sidebar.header("⚙️ Enterprise Parameters")
    forecast_horizon = st.sidebar.slider("Forecast Horizon (Months):", 1, 24, 6)

    use_min_sales = st.sidebar.checkbox("Enable Min Continuous Sales Rule", value=True)
    min_sales_req = st.sidebar.slider("Min Continuous Sales (Months):", 1, 12, 6) if use_min_sales else 1

    handle_outliers = st.sidebar.checkbox("Winsorization (Expanding Window)", value=True)
    handle_stockouts = st.sidebar.checkbox("Impute Stockouts (Leakage-Free)", value=True)

    ml_models_adv = ['AutoARIMA (Optimized)', 'Global LightGBM (ML)', 'Croston', 'TSB',
                     'Ensemble (LGBM + ARIMA)']
    available_models = st.sidebar.multiselect("Select Engines:", ml_models_adv,
                                             default=['Global LightGBM (ML)', 'TSB',
                                                      'AutoARIMA (Optimized)'])
    calc_intervals = st.sidebar.checkbox("Generate Conformal Intervals", value=True)
    val_months = st.sidebar.slider("Validation Period (Months):", 1, 6, 3)

# --- Forecast Threshold ---
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Forecast Threshold")
forecast_threshold = st.sidebar.number_input(
    "Minimum total forecast (sum across horizon) to keep SKU:",
    min_value=0, max_value=5000, value=0, step=100,
    help="SKUs whose total forecast is below this number will be filtered out separately."
)

# --- SKU CONSOLIDATION ---
st.sidebar.markdown("---")
st.sidebar.subheader("🔄 SKU Consolidation")
def get_mapping_template():
    output = io.BytesIO()
    pd.DataFrame(columns=['Old SKU Code', 'New SKU Code']).to_excel(output, index=False, sheet_name='SKU_Mapping')
    return output.getvalue()
st.sidebar.download_button("Download SKU Mapping Template", data=get_mapping_template(),
                           file_name="SKU_Mapping_Template.xlsx")
mapping_file = st.sidebar.file_uploader("Upload SKU Mapping Excel", type=['xlsx', 'xls'])

# --- LICENSE & INFO ---
st.sidebar.markdown("---")
st.sidebar.info("🐍 **Built with Python**\n\nThis software is **open-source** and freely available for experimental use.\n\n💼 **For commercial use**, please contact:\n📧 [atillakdeniz@icloud.com](mailto:atillakdeniz@icloud.com)")
st.sidebar.markdown("<div style='text-align: center; color: #888; font-size: 12px;' title='a humble demand planner'>Created by <b>Atilla AKDENİZ</b></div>", unsafe_allow_html=True)

st.title("📊 Mini Forecast Tool (MFT)")
st.caption("🚀 Unified Dual-Engine Architecture (v5.2 + v5.0 UI)")

# --- SALES TEMPLATE DOWNLOAD ---
def get_sales_template():
    output = io.BytesIO()
    template_df = pd.DataFrame({
        'SKU Code': ['SKU-001', 'SKU-001', 'SKU-002', 'SKU-002'],
        'SKU Description': ['Sample Product A', 'Sample Product A', 'Sample Product B', 'Sample Product B'],
        'Date': ['2023-01-01', '2023-02-01', '2023-01-01', '2023-02-01'],
        'Sales': [150, 180, 0, 55]
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

uploaded_file = st.file_uploader("Upload Sales Data (Excel format)", type=['xlsx', 'xls'])

if uploaded_file is not None:
    df_raw = pd.read_excel(uploaded_file)
    raw_row_count = len(df_raw)
    st.session_state.raw_row_count = raw_row_count

    if mapping_file is not None:
        map_df = pd.read_excel(mapping_file).rename(columns={'Eski SKU Kodu': 'Old SKU Code',
                                                             'Güncel SKU Kodu': 'New SKU Code'})
        if 'Old SKU Code' in map_df.columns and 'New SKU Code' in map_df.columns:
            mapping_dict = dict(zip(map_df['Old SKU Code'], map_df['New SKU Code']))
            df_raw = df_raw.rename(columns={'SKU Kodu': 'SKU Code'})
            df_raw['SKU Code'] = df_raw['SKU Code'].replace(mapping_dict)
            st.success("SKUs successfully consolidated!")

    df_aligned, desc_map, global_max_date = load_and_align_data(df_raw)

    # PyArrow fix
    sku_options = df_aligned[['SKU Code', 'SKU Description']].drop_duplicates()
    sku_list = (sku_options['SKU Code'].astype(str) + " | " +
                sku_options['SKU Description'].astype(str)).sort_values().tolist()
    selected_ui = st.multiselect("🎯 Select SKUs to Forecast [Leave blank for ALL]:", sku_list)

    if st.button("🚀 Execute Forecast"):
        st.session_state.pipeline_run = True
        start_time = time.time()
        status_text = st.empty()

        df = df_aligned.copy()
        if len(selected_ui) > 0:
            selected_codes = [x.split(" | ")[0] for x in selected_ui]
            df = df[df['SKU Code'].isin(selected_codes)]

        all_skus = df['SKU Code'].unique()
        future_dates = [global_max_date + pd.DateOffset(months=i) for i in range(1, forecast_horizon + 1)]

        # ==========================================
        #          BASIC MODE
        # ==========================================
        if app_mode == "Basic Mode":
            status_text.info("⏳ Executing Basic Statistical Pipeline...")
            results, failed_skus = [], []
            progress_bar = st.progress(0)
            timer_placeholder = st.empty()

            for idx, sku in enumerate(all_skus):
                elapsed = int(time.time() - start_time)
                mins, secs = divmod(elapsed, 60)
                timer_placeholder.markdown(f"⏱️ **Elapsed Time:** {mins} min {secs} sec")

                sku_desc = desc_map.get(sku, 'Unknown')
                ts = df[df['SKU Code'] == sku]['Sales'].values

                if use_min_sales:
                    if (pd.Series(ts) > 0).rolling(window=min_sales_req).sum().max() < min_sales_req:
                        failed_skus.append({'SKU Code': sku, 'SKU Description': sku_desc,
                                            'Reason': f'Failed Continuous Sales Rule ({min_sales_req}m)'})
                        progress_bar.progress((idx + 1) / len(all_skus))
                        continue

                ts_data = ts[-(train_months + test_months):]
                if len(ts_data) <= test_months + 2:
                    failed_skus.append({'SKU Code': sku, 'SKU Description': sku_desc,
                                        'Reason': 'Insufficient Historical Data'})
                    progress_bar.progress((idx + 1) / len(all_skus))
                    continue

                train_y, test_y = ts_data[:-test_months], ts_data[-test_months:]
                best_mape, best_model_name = float('inf'), "Not Found"
                best_forecast = np.zeros(forecast_horizon)

                for m_name in available_models:
                    try:
                        preds = np.zeros(test_months)
                        f_preds = np.zeros(forecast_horizon)

                        if m_name == 'Naive':
                            preds = np.repeat(train_y[-1], test_months)
                            f_preds = np.repeat(train_y[-1], forecast_horizon)
                        elif m_name == 'Moving Average':
                            ma = pd.Series(train_y).rolling(3).mean().iloc[-1]
                            preds = np.repeat(ma, test_months)
                            f_preds = np.repeat(ma, forecast_horizon)
                        elif m_name == 'Holt-Winters':
                            hw = ExponentialSmoothing(train_y, trend='add').fit()
                            preds = hw.forecast(test_months)
                            f_preds = hw.forecast(forecast_horizon)
                        elif m_name == 'SARIMA':
                            sarima = SARIMAX(train_y, order=(1,1,1)).fit(disp=False)
                            preds = sarima.forecast(steps=test_months)
                            sarima_full = SARIMAX(ts_data, order=(1,1,1)).fit(disp=False)
                            f_preds = sarima_full.forecast(steps=forecast_horizon)
                        elif m_name in ['XGBoost', 'LightGBM', 'CatBoost']:
                            idx_train = np.arange(len(train_y)).reshape(-1, 1)
                            idx_test = np.arange(len(train_y), len(train_y)+test_months).reshape(-1, 1)
                            idx_full = np.arange(len(ts_data)).reshape(-1, 1)
                            idx_future = np.arange(len(ts_data), len(ts_data)+forecast_horizon).reshape(-1, 1)
                            if m_name == 'XGBoost':
                                model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=50)
                            elif m_name == 'LightGBM':
                                model = lgb.LGBMRegressor(n_estimators=50, verbose=-1)
                            else:
                                model = CatBoostRegressor(iterations=50, verbose=0)
                            model.fit(idx_train, train_y)
                            preds = model.predict(idx_test)
                            model.fit(idx_full, ts_data)
                            f_preds = model.predict(idx_future)
                        elif m_name == 'Croston':
                            preds = croston_classic(train_y, test_months)
                            f_preds = croston_classic(ts_data, forecast_horizon)
                        elif m_name == 'TSB':
                            preds = tsb_forecast(train_y, test_months)
                            f_preds = tsb_forecast(ts_data, forecast_horizon)

                        mape = mape_loss_basic(test_y, preds) * 100
                        if mape < best_mape:
                            best_mape = mape
                            best_model_name = m_name
                            best_forecast = f_preds
                    except:
                        continue

                best_forecast = np.round(np.maximum(0, best_forecast)).astype(int)

                if np.all(best_forecast == 0) or best_model_name == "Not Found":
                    failed_skus.append({'SKU Code': sku, 'SKU Description': sku_desc,
                                        'Reason': 'All models yielded Zero or Failed'})
                    progress_bar.progress((idx + 1) / len(all_skus))
                    continue

                segment = get_robust_segment(ts)

                row_data = {
                    'SKU Code': sku,
                    'SKU Description': sku_desc,
                    'Demand Segment': segment,
                    'Selected Model': best_model_name,
                    'Validation MAPE (%)': round(best_mape, 2),
                    'V. Accuracy (%)': round(max(0, 100 - best_mape), 2),
                    'Volume': ts.sum()
                }
                for i, f_date in enumerate(future_dates):
                    row_data[f"{f_date.year}-{f_date.month:02d} Forecast"] = best_forecast[i] if i < len(best_forecast) else 0
                results.append(row_data)
                progress_bar.progress((idx + 1) / len(all_skus))

            st.session_state.res_df = pd.DataFrame(results)
            st.session_state.fail_df = pd.DataFrame(failed_skus)
            status_text.success(f"✅ Pipeline Completed in {int(time.time() - start_time)} seconds.")
            timer_placeholder.empty()

        # ==========================================
        #          ADVANCED MODE
        # ==========================================
        elif app_mode == "Advanced Enterprise Mode":
            status_text.info("⚙️ Phase 1: Robust Segmentation & Leakage-Safe Treatments...")
            segments = {}
            for sku in all_skus:
                mask = df['SKU Code'] == sku
                ts = df.loc[mask, 'Sales'].values
                segment = get_robust_segment(ts)
                segments[sku] = segment
                if handle_outliers:
                    ts = expanding_winsorize(ts)
                if handle_stockouts:
                    ts = expanding_impute(ts, segment)
                df.loc[mask, 'Sales'] = ts

            status_text.info("🧠 Phase 2: Feature Generation & Lag Safety Check...")
            df_feat, feature_registry = generate_features(df)
            st.session_state.feature_registry = feature_registry
            df_feat = df_feat.dropna(subset=feature_registry)

            train_df = df_feat[df_feat['Datetime'] <= global_max_date - pd.DateOffset(months=val_months)]
            test_df = df_feat[df_feat['Datetime'] > global_max_date - pd.DateOffset(months=val_months)]

            global_lgb = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, verbose=-1)
            if not train_df.empty:
                global_lgb.fit(train_df[feature_registry], train_df['Sales'])

            st.session_state.train_df = train_df
            st.session_state.global_lgb = global_lgb

            status_text.info("⏳ Phase 3: Parallel CV Execution...")
            parallel_results = Parallel(n_jobs=-1, backend='threading')(
                delayed(process_single_sku)(sku, df, desc_map, segments, test_df, global_lgb,
                                           available_models, forecast_horizon, feature_registry,
                                           future_dates, calc_intervals, use_min_sales, min_sales_req,
                                           val_months)
                for sku in all_skus
            )

            results, failed_skus = [], []
            for res in parallel_results:
                if res['status'] == 'success':
                    results.append(res['data'])
                else:
                    failed_skus.append(res['data'])

            st.session_state.res_df = pd.DataFrame(results)
            st.session_state.fail_df = pd.DataFrame(failed_skus)
            status_text.success(f"✅ Pipeline Completed in {int(time.time() - start_time)} seconds.")

# --- POST-PROCESSING: FORECAST THRESHOLD FILTER ---
if st.session_state.pipeline_run:
    res_df = st.session_state.res_df.copy()
    fail_df = st.session_state.fail_df.copy()
    filtered_df = pd.DataFrame()

    if forecast_threshold > 0 and not res_df.empty:
        if app_mode == "Basic Mode":
            forecast_cols = [c for c in res_df.columns if "Forecast" in c]
        else:
            forecast_cols = [c for c in res_df.columns if "Final (P50)" in c]

        if forecast_cols:
            total_forecast = res_df[forecast_cols].sum(axis=1)
            above_threshold = total_forecast >= forecast_threshold
            below_threshold = ~above_threshold

            if below_threshold.any():
                filtered_df = res_df.loc[below_threshold, ['SKU Code', 'SKU Description']].copy()
                filtered_df['Reason'] = f'Total forecast below {forecast_threshold}'
                res_df = res_df.loc[above_threshold]

    st.session_state.res_df = res_df
    st.session_state.fail_df = fail_df
    st.session_state.filtered_df = filtered_df

# --- RESULTS RENDERING ---
if st.session_state.pipeline_run and not st.session_state.res_df.empty:
    res_df = st.session_state.res_df.copy()
    fail_df = st.session_state.fail_df.copy()
    filtered_df = st.session_state.filtered_df.copy()
    total_skus = len(res_df) + len(fail_df) + len(filtered_df)

    st.markdown("---")
    st.markdown("### 📝 Execution Summary")
    st.write(f"- **Total Rows Analyzed:** {st.session_state.raw_row_count:,}")
    st.write(f"- **Total Unique SKUs Checked:** {total_skus:,}")
    st.write(f"- **Successfully Forecasted SKUs:** {len(res_df):,}")
    if not filtered_df.empty:
        st.info(f"🔍 **Filtered Out (Below Threshold):** {len(filtered_df)} SKUs")
    if not fail_df.empty:
        st.write(f"- **SKUs Excluded from Forecast:** {len(fail_df):,}")
        reason_counts = fail_df['Reason'].value_counts()
        for reason, count in reason_counts.items():
            st.markdown(f"- **{count} SKUs**: *{reason}*")
    st.markdown("---")

    # ==========================================
    #   COMMON PORTFOLIO METRICS & GRAPHS
    # ==========================================
    if app_mode == "Basic Mode":
        res_df['Weighted Error'] = res_df['Validation MAPE (%)'] / 100 * res_df['Volume']
        total_volume = res_df['Volume'].sum()
        if total_volume > 0:
            port_wmape = res_df['Weighted Error'].sum() / total_volume * 100
            port_accuracy = max(0, 100 - port_wmape)
        else:
            port_wmape = port_accuracy = 0.0
        forecast_cols = [c for c in res_df.columns if "Forecast" in c]
        total_forecast_volume = res_df[forecast_cols].sum().sum()
    else:
        total_volume = res_df['Volume'].sum()
        if total_volume > 0:
            port_wmape = (res_df['Val. WMAPE (%)'] * res_df['Volume']).sum() / total_volume
            port_bias = (res_df['Val. Bias (%)'] * res_df['Volume']).sum() / total_volume
        else:
            port_wmape = port_bias = 0.0
        forecast_cols = [c for c in res_df.columns if "Final (P50)" in c]
        total_forecast_volume = res_df[forecast_cols].sum().sum()

    col1, col2, col3 = st.columns(3)
    if app_mode == "Basic Mode":
        col1.metric("Portfolio WMAPE", f"{round(port_wmape, 1)}%")
        col2.metric("Portfolio Accuracy", f"{round(port_accuracy, 1)}%")
        col3.metric("Total Forecast Volume", f"{total_forecast_volume:,.0f}")
    else:
        col1.metric("Portfolio WMAPE", f"{round(port_wmape, 1)}%")
        col2.metric("Portfolio Bias", f"{round(port_bias, 1)}%")
        col3.metric("Total Forecast Volume", f"{total_forecast_volume:,.0f}")

    # Graphs
    st.markdown("### 📊 Demand Segmentation")
    seg_fig = px.pie(res_df, names='Demand Segment', title='SKU Demand Segments',
                     hole=0.3, color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(seg_fig, use_container_width=True)

    if app_mode == "Basic Mode":
        st.markdown("### 🧠 Model Contribution to Portfolio")
        model_stats = res_df.groupby('Selected Model').apply(
            lambda x: pd.Series({
                'SKU_Count': len(x),
                'Model_WMAPE': (x['Validation MAPE (%)'] * x['Volume']).sum() / (x['Volume'].sum() + 1e-5)
            })
        ).reset_index()
        model_stats['V. Accuracy (%)'] = (100 - model_stats['Model_WMAPE']).clip(lower=0).round(2)
        fig_model = px.pie(model_stats, values='SKU_Count', names='Selected Model',
                           custom_data=['V. Accuracy (%)'],
                           title="Model Selection Distribution", hole=0.3,
                           color_discrete_sequence=px.colors.qualitative.Set2)
        fig_model.update_traces(hovertemplate="<b>%{label}</b><br>%{value} SKUs (%{percent})<br>Model Accuracy: %{customdata[0]}%")
        st.plotly_chart(fig_model, use_container_width=True)
    else:
        st.markdown("### 🧠 Engine Allocation Strategy")
        fig_eng = px.pie(res_df, names='Winning Engine', title='Winning Engine Distribution',
                         hole=0.3, color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig_eng, use_container_width=True)

    # --- MODE SPECIFIC DETAILS ---
    if app_mode == "Basic Mode":
        st.markdown("### 📋 Basic Forecast Results")
        display_df = res_df.drop(columns=['Volume', 'Weighted Error'], errors='ignore').copy()
        st.download_button("📥 Download Results as Excel", data=to_excel_download(display_df),
                           file_name="Basic_Forecast_Results.xlsx")
        st.dataframe(display_df, use_container_width=True)

    elif app_mode == "Advanced Enterprise Mode":
        st.markdown("### 📊 ML Governance & Planner Override")
        with st.expander("🛠️ Show Pipeline Metadata (Feature Registry & Setup)"):
            st.write(f"**Total Features Used:** {len(st.session_state.feature_registry)}")
            st.code(", ".join(st.session_state.feature_registry))
            st.write(f"**Validation Strategy:** Walk-Forward TimeSeriesSplit (Last {val_months} Periods)")

        display_cols = [c for c in res_df.columns if not c.endswith('_Original') and c != 'Volume']
        edited_df = st.data_editor(res_df[display_cols], num_rows="fixed", use_container_width=True)

        future_dates_str = [c for c in res_df.columns if "Final (P50)" in c]
        audit_data = []
        for col in future_dates_str:
            orig_col = col.replace(" Final (P50)", "_Original")
            if orig_col in res_df.columns:
                diff = edited_df[col] - res_df[orig_col]
                for idx, val in diff[diff != 0].items():
                    audit_data.append({
                        'SKU Code': edited_df.loc[idx, 'SKU Code'],
                        'Period': col.split()[0],
                        'System Forecast': res_df.loc[idx, orig_col],
                        'Planner Override': edited_df.loc[idx, col],
                        'Delta': val
                    })
        if audit_data:
            st.warning(f"🚨 **Governance Alert:** {len(audit_data)} manual overrides detected.")
            audit_df = pd.DataFrame(audit_data)
            st.dataframe(audit_df, use_container_width=True)
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
                edited_df.to_excel(writer, index=False, sheet_name='Final_Plan')
                audit_df.to_excel(writer, index=False, sheet_name='Audit_Trail')
            st.download_button("📥 Export Plan & Audit Trail", data=buf.getvalue(),
                               file_name="Enterprise_Plan_with_Audit.xlsx")
        else:
            st.download_button("📥 Export Final Plan", data=to_excel_download(edited_df),
                               file_name="Enterprise_Plan.xlsx")

        if not st.session_state.train_df.empty and st.session_state.global_lgb is not None:
            st.markdown("### 🧠 Explainable AI: Forecast Drivers")
            importance_df = calculate_shap_importance(
                st.session_state.global_lgb,
                st.session_state.train_df[st.session_state.feature_registry].head(500),
                st.session_state.feature_registry
            )
            fig_shap = px.bar(importance_df.head(10), x='Importance', y='Feature',
                              orientation='h', title='Global ML Top Features')
            fig_shap.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_shap, use_container_width=True)

            with st.expander("📖 Feature Descriptions"):
                feat_desc = {
                    'Lag_1': 'Sales 1 month ago',
                    'Lag_2': 'Sales 2 months ago',
                    'Lag_3': 'Sales 3 months ago',
                    'Lag_6': 'Sales 6 months ago',
                    'Lag_12': 'Sales 12 months ago',
                    'Rolling_Mean_3': '3-month rolling average (using Lag_1)',
                    'Rolling_Std_3': '3-month rolling standard deviation (using Lag_1)',
                    'Momentum': 'Momentum indicator: (Lag_1 - Lag_2) / (Lag_2 + 1e-6)',
                    'Month_Sin': 'Sine transform of month (cyclic encoding)',
                    'Month_Cos': 'Cosine transform of month (cyclic encoding)',
                    'Quarter': 'Quarter of the year (1-4)',
                    'Month': 'Month of the year (1-12)',
                    'WorkingDays': 'Number of working days in the month (Turkey calendar)',
                    'IsHoliday': 'Indicator if the month contains a national holiday',
                    'Ramadan_Flag': 'Indicator if month is March or April (Ramadan effect)',
                    'BlackFriday': 'Indicator for November (Black Friday effect)'
                }
                desc_df = pd.DataFrame(
                    [(k, v) for k, v in feat_desc.items() if k in st.session_state.feature_registry],
                    columns=['Feature', 'Description']
                )
                st.table(desc_df)

    # ==========================================
    #   GROUPED EXCLUSION LOG
    # ==========================================
    all_excluded = pd.concat([
        fail_df.assign(Status='Excluded'),
        filtered_df.assign(Status='Filtered Out')
    ], ignore_index=True) if not filtered_df.empty else fail_df

    if not all_excluded.empty:
        st.markdown("---")
        st.markdown("### 🔍 Detailed Exclusion Log")
        for reason, group in all_excluded.groupby('Reason'):
            with st.expander(f"**{reason}** ({len(group)} SKUs)"):
                st.dataframe(group[['SKU Code', 'SKU Description']], use_container_width=True)

else:
    if st.session_state.pipeline_run and st.session_state.res_df.empty:
        st.warning("No results generated for the selected criteria.")
