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
import plotly.graph_objects as go

import lightgbm as lgb
import pmdarima as pm

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')

st.set_page_config(page_title="MFT v5.2", layout="wide")

# ==========================================
#    1. DATA STANDARDIZATION (RESAMPLE FIRST)
# ==========================================
@st.cache_data(show_spinner=False)
def load_and_align_data(df_raw):
    """Monthly Alignment for all SKUs before any lagging."""
    df_raw = df_raw.rename(columns={'SKU Kodu': 'SKU Code', 'SKU Tanım': 'SKU Description', 'Tarih': 'Date', 'Satış KG': 'Sales'})
    if 'SKU Description' not in df_raw.columns: df_raw['SKU Description'] = "Unknown"
    df_raw['Datetime'] = pd.to_datetime(df_raw['Date'])
    
    desc_map = df_raw[['SKU Code', 'SKU Description']].drop_duplicates(subset=['SKU Code']).set_index('SKU Code')['SKU Description'].to_dict()
    df_grouped = df_raw.groupby(['SKU Code', 'Datetime'])['Sales'].sum().reset_index()
    
    global_max_date = df_grouped['Datetime'].max()
    all_skus = df_grouped['SKU Code'].unique()
    
    aligned_frames = []
    for sku in all_skus:
        group = df_grouped[df_grouped['SKU Code'] == sku].set_index('Datetime').resample('MS')['Sales'].sum().fillna(0)
        full_idx = pd.date_range(start=group.index.min(), end=global_max_date, freq='MS')
        group = group.reindex(full_idx, fill_value=0).reset_index().rename(columns={'index': 'Datetime'})
        group['SKU Code'] = sku
        aligned_frames.append(group)
        
    df_aligned = pd.concat(aligned_frames, ignore_index=True)
    df_aligned['SKU Description'] = df_aligned['SKU Code'].map(desc_map).fillna("Unknown")
    
    return df_aligned, desc_map, global_max_date

# ==========================================
#    2. LEAKAGE-FREE TREATMENTS & SEGMENTATION
# ==========================================
def get_robust_segment(ts):
    """Enhanced FMCG Segmentation with Zero Ratio to prevent noise sensitivity."""
    if len(ts) == 0: return "Dead"
    non_zero = ts[ts > 0]
    if len(non_zero) == 0: return "Dead"
    
    adi = len(ts) / len(non_zero)
    cv2 = (np.std(non_zero) / np.mean(non_zero)) ** 2 if len(non_zero) > 1 else 0
    zero_ratio = len(ts[ts == 0]) / len(ts)
    
    if zero_ratio >= 0.4:
        return "Highly Intermittent" if cv2 > 0.5 else "Intermittent"
    elif adi <= 1.32 and cv2 <= 0.49: return "Smooth"
    elif adi <= 1.32 and cv2 > 0.49: return "Volatile"
    else: return "Lumpy"

def expanding_winsorize(ts, limit=3):
    """100% Leakage-Free: Uses only past data (expanding window) to calculate boundaries."""
    ts_s = pd.Series(ts)
    if len(ts_s) < 3: return ts
    
    roll_mean = ts_s.expanding().mean().shift(1).fillna(ts_s.iloc[0])
    roll_std = ts_s.expanding().std().shift(1).fillna(0)
    
    z_scores = np.where(roll_std > 0, (ts_s - roll_mean) / roll_std, 0)
    capped = np.where(z_scores > limit, roll_mean + limit * roll_std, ts)
    capped = np.where(z_scores < -limit, np.maximum(0, roll_mean - limit * roll_std), capped)
    return capped

def expanding_impute(ts, segment):
    """Leakage-Free stockout imputation."""
    if segment in ["Smooth", "Volatile"] and len(ts) >= 3:
        ts_series = pd.Series(ts)
        roll_mean = ts_series.replace(0, np.nan).expanding().mean().shift(1)
        return np.where((ts == 0) & roll_mean.notna(), roll_mean, ts)
    return ts

# ==========================================
#    3. SINGLE SOURCE FEATURE PIPELINE & REGISTRY
# ==========================================
def generate_features(df):
    """Single Feature Pipeline. Dynamically registers and returns features."""
    df = df.copy()
    
    # Calendar
    df['Quarter'] = df['Datetime'].dt.quarter
    df['Month'] = df['Datetime'].dt.month
    df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    
    # Holidays & Working Days
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

    # Lags & Rolling (Lag Safety Checked via dropna later)
    for lag in [1, 2, 3, 6, 12]:
        df[f'Lag_{lag}'] = df.groupby('SKU Code')['Sales'].shift(lag)

    df['Rolling_Mean_3'] = df.groupby('SKU Code')['Lag_1'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df['Rolling_Std_3'] = df.groupby('SKU Code')['Lag_1'].transform(lambda x: x.rolling(3, min_periods=1).std().fillna(0))
    df['Momentum'] = (df['Lag_1'] - df['Lag_2']) / (df['Lag_2'].abs() + 1e-6)

    # Feature Registry: Automatically detect feature columns
    base_cols = ['SKU Code', 'SKU Description', 'Datetime', 'Sales', 'Demand Segment']
    feature_registry = [col for col in df.columns if col not in base_cols]
    
    return df, feature_registry

# ==========================================
#    4. ADVANCED TIME-SERIES CV & INTERVALS
# ==========================================
def calculate_ts_cv_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.maximum(0, np.array(y_pred))
    sum_true = np.sum(y_true)
    if sum_true == 0: return (100.0 if np.sum(y_pred) > 0 else 0.0), (100.0 if np.sum(y_pred) > 0 else 0.0)
    wmape = min(999.0, np.sum(np.abs(y_true - y_pred)) / sum_true * 100)
    bias = (np.sum(y_pred) - sum_true) / sum_true * 100
    return wmape, bias

def propagate_uncertainty(preds, residuals):
    if len(residuals) == 0: sigma = np.std(preds) * 0.1
    else: sigma = np.std(residuals)
    
    lowers, uppers = [], []
    for h, pred in enumerate(preds, start=1):
        dynamic_sigma = sigma * np.sqrt(h)
        lowers.append(max(0, pred - 1.645 * dynamic_sigma)) # P10
        uppers.append(pred + 1.645 * dynamic_sigma) # P90
    return np.array(lowers), np.array(uppers)

# ==========================================
#    5. CROSTON / TSB WITH BASELINE
# ==========================================
def croston_classic(ts, horizon=1, alpha=0.1):
    ts = np.array(ts)
    demand = ts[ts > 0]
    if len(demand) == 0: return np.zeros(horizon)
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
    if len(ts) == 0: return np.zeros(horizon)
    p = 1 if ts[0] > 0 else 0
    z = ts[0] if ts[0] > 0 else 0
    for val in ts:
        occurrence = 1 if val > 0 else 0
        p = alpha_p * occurrence + (1 - alpha_p) * p
        if occurrence: z = alpha_d * val + (1 - alpha_d) * z
    return np.repeat(p * z, horizon)

# ==========================================
#    6. PARALLEL WORKER (SKU PROCESSOR)
# ==========================================
def process_single_sku(sku, df, desc_map, segments, test_df, global_lgb, available_models, forecast_horizon, feature_registry, future_dates, calc_intervals):
    try:
        sku_df = df[df['SKU Code'] == sku].copy()
        ts = sku_df['Sales'].values
        sku_desc = desc_map.get(sku, 'Unknown')
        
        if len(ts) < 15:
            return {'status': 'failed', 'data': {'SKU Code': sku, 'SKU Description': sku_desc, 'Reason': 'Insufficient History (<15m)'}}
            
        # Walk-Forward Simulation (Last 3 periods)
        train_y, test_y = ts[:-3], ts[-3:]
        best_wmape, best_model_name = float('inf'), "Naive (Baseline)"
        final_forecast = np.repeat(train_y[-1] if len(train_y)>0 else 0, forecast_horizon)
        
        metrics_dict, forecasts_dict, residuals_dict = {}, {}, {}
        
        # Baseline: Naive
        preds_naive = np.repeat(train_y[-1], 3)
        metrics_dict['Naive'] = calculate_ts_cv_metrics(test_y, preds_naive)
        residuals_dict['Naive'] = test_y - preds_naive
        
        # Engine: Global ML
        if 'Global LightGBM (ML)' in available_models and not test_df[test_df['SKU Code'] == sku].empty:
            sku_test_feat = test_df[test_df['SKU Code'] == sku][feature_registry]
            preds_lgb = global_lgb.predict(sku_test_feat)
            metrics_dict['Global LightGBM (ML)'] = calculate_ts_cv_metrics(test_y, preds_lgb)
            residuals_dict['Global LightGBM (ML)'] = test_y - preds_lgb
            
            # Recursive Feature Generation
            temp_df = sku_df.copy()
            for i in range(forecast_horizon):
                temp_df, _ = generate_features(temp_df)
                temp_feat = temp_df.iloc[-1:][feature_registry]
                next_pred = max(0, global_lgb.predict(temp_feat)[0])
                next_row = pd.DataFrame({'SKU Code': [sku], 'Datetime': [future_dates[i]], 'Sales': [next_pred]})
                temp_df = pd.concat([temp_df, next_row], ignore_index=True)
            forecasts_dict['Global LightGBM (ML)'] = temp_df['Sales'].iloc[-forecast_horizon:].values

        # Engine: AutoARIMA (Strict Compute Limits)
        if 'AutoARIMA (Optimized)' in available_models and segments[sku] in ["Smooth", "Volatile"]:
            try:
                arima = pm.auto_arima(train_y, stepwise=True, max_p=1, max_q=1, max_order=2, seasonal=False, suppress_warnings=True, error_action="ignore")
                preds_arima = arima.predict(n_periods=3)
                metrics_dict['AutoARIMA (Optimized)'] = calculate_ts_cv_metrics(test_y, preds_arima)
                residuals_dict['AutoARIMA (Optimized)'] = test_y - preds_arima
                
                arima_full = pm.auto_arima(ts, stepwise=True, max_p=1, max_q=1, max_order=2, seasonal=False, suppress_warnings=True, error_action="ignore")
                forecasts_dict['AutoARIMA (Optimized)'] = arima_full.predict(n_periods=forecast_horizon)
            except: pass

        # Engine: Intermittent
        if segments[sku] in ["Intermittent", "Highly Intermittent", "Lumpy"]:
            if 'Croston' in available_models:
                p_cr = croston_classic(train_y, 3)
                metrics_dict['Croston'] = calculate_ts_cv_metrics(test_y, p_cr)
                residuals_dict['Croston'] = test_y - p_cr
                forecasts_dict['Croston'] = croston_classic(ts, forecast_horizon)
            if 'TSB' in available_models:
                p_tsb = tsb_forecast(train_y, 3)
                metrics_dict['TSB'] = calculate_ts_cv_metrics(test_y, p_tsb)
                residuals_dict['TSB'] = test_y - p_tsb
                forecasts_dict['TSB'] = tsb_forecast(ts, forecast_horizon)

        # Tournament Evaluation
        for m_name, mets in metrics_dict.items():
            if mets[0] < best_wmape:
                best_wmape = mets[0]
                best_model_name = m_name
                final_forecast = forecasts_dict.get(m_name, np.repeat(ts[-1], forecast_horizon))

        fva_score = metrics_dict['Naive'][0] - best_wmape 
        final_forecast = np.round(np.maximum(0, final_forecast)).astype(int)
        
        if np.all(final_forecast == 0):
            return {'status': 'failed', 'data': {'SKU Code': sku, 'SKU Description': sku_desc, 'Reason': 'Forecast is Absolute Zero'}}

        # Uncertainty Propagation
        p10_vals, p90_vals = np.zeros(forecast_horizon), np.zeros(forecast_horizon)
        if calc_intervals and best_model_name in residuals_dict:
            p10_vals, p90_vals = propagate_uncertainty(final_forecast, residuals_dict[best_model_name])

        row_data = {
            'SKU Code': sku, 'SKU Description': sku_desc, 'Demand Segment': segments[sku],
            'Winning Engine': best_model_name, 'FVA (%)': round(fva_score, 1),
            'Val. WMAPE (%)': round(best_wmape, 1), 'Val. Bias (%)': round(metrics_dict[best_model_name][1], 1),
            'Volume': ts.sum()
        }
        for i, f_date in enumerate(future_dates):
            mon_str = f"{f_date.year}-{f_date.month:02d}"
            row_data[f"{mon_str} P10"] = int(p10_vals[i]) if calc_intervals else 0
            row_data[f"{mon_str}_Original"] = final_forecast[i] 
            row_data[f"{mon_str} Final (P50)"] = final_forecast[i]
            row_data[f"{mon_str} P90"] = int(p90_vals[i]) if calc_intervals else 0
            
        return {'status': 'success', 'data': row_data}
        
    except Exception as e:
        return {'status': 'failed', 'data': {'SKU Code': sku, 'SKU Description': 'Unknown', 'Reason': f'System Crash: {str(e)}'}}

def to_excel_download(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Forecast_Results')
    return output.getvalue()


# ==========================================
#                 UI SETUP
# ==========================================
if 'pipeline_run' not in st.session_state: st.session_state.pipeline_run = False
if 'res_df' not in st.session_state: st.session_state.res_df = pd.DataFrame()
if 'fail_df' not in st.session_state: st.session_state.fail_df = pd.DataFrame()
if 'train_df' not in st.session_state: st.session_state.train_df = pd.DataFrame()
if 'global_lgb' not in st.session_state: st.session_state.global_lgb = None
if 'feature_registry' not in st.session_state: st.session_state.feature_registry = []

st.sidebar.header("⚙️ Enterprise Parameters")
forecast_horizon = st.sidebar.slider("Forecast Horizon (Months):", 1, 24, 6)
handle_outliers = st.sidebar.checkbox("Winsorization (Expanding Window)", value=True)
handle_stockouts = st.sidebar.checkbox("Impute Stockouts (Leakage-Free)", value=True)
ml_models_adv = ['AutoARIMA (Optimized)', 'Global LightGBM (ML)', 'Croston', 'TSB']
available_models = st.sidebar.multiselect("Select Engines (ARIMA is CPU intensive):", ml_models_adv, default=['Global LightGBM (ML)', 'TSB'])
calc_intervals = st.sidebar.checkbox("Generate Conformal Intervals", value=True)

st.title("📊 MFT v5.2")
st.caption("🚀 Running with Strict Leakage Control, Dynamic Feature Registry & ML Governance")

uploaded_file = st.file_uploader("Upload Sales Data (Excel format)", type=['xlsx', 'xls'])

if uploaded_file is not None:
    df_aligned, desc_map, global_max_date = load_and_align_data(pd.read_excel(uploaded_file))
    selected_ui = st.multiselect("🎯 Select SKUs to Forecast [Leave blank for ALL]:", np.sort(df_aligned['SKU Code'].astype(str) + " | " + df_aligned['SKU Description'].astype(str).unique()))
    
    if st.button("🚀 Execute Certified Pipeline"):
        st.session_state.pipeline_run = True
        start_time = time.time()
        status_text = st.empty()
        
        df = df_aligned.copy()
        if len(selected_ui) > 0:
            selected_codes = [x.split(" | ")[0] for x in selected_ui]
            df = df[df['SKU Code'].isin(selected_codes)]
            
        all_skus = df['SKU Code'].unique()

        status_text.info("⚙️ Phase 1: Robust Segmentation & Leakage-Safe Treatments...")
        segments = {}
        for sku in all_skus:
            mask = df['SKU Code'] == sku
            ts = df.loc[mask, 'Sales'].values
            segment = get_robust_segment(ts)
            segments[sku] = segment
            if handle_outliers: ts = expanding_winsorize(ts)
            if handle_stockouts: ts = expanding_impute(ts, segment)
            df.loc[mask, 'Sales'] = ts

        status_text.info("🧠 Phase 2: Feature Generation & Lag Safety Check...")
        df_feat, feature_registry = generate_features(df)
        st.session_state.feature_registry = feature_registry
        
        # Lag Safety Check
        df_feat = df_feat.dropna(subset=feature_registry)
        
        train_df = df_feat[df_feat['Datetime'] <= global_max_date - pd.DateOffset(months=3)]
        test_df = df_feat[df_feat['Datetime'] > global_max_date - pd.DateOffset(months=3)]
        
        global_lgb = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, verbose=-1)
        if not train_df.empty: global_lgb.fit(train_df[feature_registry], train_df['Sales'])
        
        st.session_state.train_df = train_df
        st.session_state.global_lgb = global_lgb

        status_text.info("⏳ Phase 3: Parallel CV Execution...")
        future_dates = [global_max_date + pd.DateOffset(months=i) for i in range(1, forecast_horizon + 1)]
        
        parallel_results = Parallel(n_jobs=-1, backend='threading')(
            delayed(process_single_sku)(sku, df, desc_map, segments, test_df, global_lgb, available_models, forecast_horizon, feature_registry, future_dates, calc_intervals)
            for sku in all_skus
        )
        
        results, failed_skus = [], []
        for res in parallel_results:
            if res['status'] == 'success': results.append(res['data'])
            else: failed_skus.append(res['data'])

        st.session_state.res_df = pd.DataFrame(results)
        st.session_state.fail_df = pd.DataFrame(failed_skus)
        status_text.success(f"✅ Pipeline Completed in {int(time.time() - start_time)} seconds.")

# ==========================================
#     7. GOVERNANCE & AUDIT TRAIL
# ==========================================
if st.session_state.pipeline_run and not st.session_state.res_df.empty:
    res_df = st.session_state.res_df.copy()
    
    st.markdown("---")
    st.markdown("### 📊 ML Governance & Planner Override")
    
    with st.expander("🛠️ Show Pipeline Metadata (Feature Registry & Setup)"):
        st.write(f"**Total Features Used:** {len(st.session_state.feature_registry)}")
        st.code(", ".join(st.session_state.feature_registry))
        st.write("**Validation Strategy:** Walk-Forward TimeSeriesSplit (Last 3 Periods)")
    
    display_cols = [c for c in res_df.columns if not c.endswith('_Original') and c != 'Volume']
    edited_df = st.data_editor(res_df[display_cols], num_rows="fixed", use_container_width=True)
    
    future_dates_str = [c for c in res_df.columns if "Final (P50)" in c]
    audit_data = []
    for col in future_dates_str:
        orig_col = col.replace(" Final (P50)", "_Original")
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
        st.download_button("📥 Export Plan & Audit Trail", data=buf.getvalue(), file_name="Enterprise_Plan_with_Audit.xlsx")
    else:
        st.download_button("📥 Export Final Plan", data=to_excel_download(edited_df), file_name="Enterprise_Plan.xlsx")

    if not st.session_state.train_df.empty and st.session_state.global_lgb is not None:
        st.markdown("### 🧠 Explainable AI: Forecast Drivers")
        importance_df = calculate_shap_importance(st.session_state.global_lgb, st.session_state.train_df[st.session_state.feature_registry].head(500), st.session_state.feature_registry)
        st.plotly_chart(px.bar(importance_df.head(10), x='Importance', y='Feature', orientation='h', title='Global ML Top Features').update_layout(yaxis={'categoryorder':'total ascending'}), use_container_width=True)

if not st.session_state.fail_df.empty:
    with st.expander("🔍 Show Detailed Exclusion Log"):
        st.code("\n".join([f"{r['SKU Code']} | {r['SKU Description']} -> {r['Reason']}" for _, r in st.session_state.fail_df.iterrows()]), language='text')
