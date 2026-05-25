import streamlit as st
import pandas as pd
import numpy as np
import io
import time
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

# --- UI: MODE SELECTION ---
st.sidebar.header("🕹️ Operation Mode")
app_mode = st.sidebar.radio("Select Forecasting Engine:", ["Basic Mode", "Advanced Mode"])
st.sidebar.markdown("---")

# --- UI: DYNAMIC PARAMETERS BASED ON MODE ---
if app_mode == "Basic Mode":
    st.sidebar.header("⚙️ Basic Parameters")
    train_months = st.sidebar.number_input("Max Training Period (Months):", min_value=12, max_value=60, value=36)
    test_months = st.sidebar.slider("Validation Period (Months):", 1, 12, 3)
    forecast_horizon = st.sidebar.slider("Forecast Horizon (Months):", 1, 24, 6)

    use_min_sales = st.sidebar.checkbox("Enable Min Continuous Sales Rule", value=True)
    if use_min_sales:
        min_sales_req = st.sidebar.slider("Min Continuous Sales (Months):", 1, 12, 6)
    else:
        min_sales_req = 1

    st.sidebar.markdown("---")
    st.sidebar.subheader("🧠 Basic Models")
    ml_models_basic = ['Naive', 'Moving Average', 'Holt-Winters', 'SARIMA', 'XGBoost', 'LightGBM', 'CatBoost', 'Croston', 'SBA', 'TSB']
    available_models = st.sidebar.multiselect("Select Models to Test:", ml_models_basic, default=['Moving Average', 'Holt-Winters', 'SARIMA', 'XGBoost', 'Croston'])

elif app_mode == "Advanced Mode":
    st.sidebar.header("⚙️ Advanced Parameters")
    train_months = st.sidebar.number_input("Max Training History (Months):", min_value=12, max_value=60, value=36)
    forecast_horizon = st.sidebar.slider("Forecast Horizon (Months):", 1, 24, 6)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🛡️ Data Preprocessing")
    handle_outliers = st.sidebar.checkbox("Apply Outlier Capping (Winsorization)", value=True)
    handle_stockouts = st.sidebar.checkbox("Detect & Impute Stockouts", value=True)

    use_min_sales = st.sidebar.checkbox("Enable Min Continuous Sales Rule", value=True)
    if use_min_sales:
        min_sales_req = st.sidebar.slider("Min Continuous Sales (Months):", 1, 12, 6)
    else:
        min_sales_req = 1

    st.sidebar.markdown("---")
    st.sidebar.subheader("🧠 Enterprise Models")
    ml_models_adv = ['AutoARIMA (Optimized)', 'Global LightGBM (ML)', 'Croston (Intermittent)', 'Ensemble (LGBM + ARIMA)']
    available_models = st.sidebar.multiselect("Select Forecasting Engines:", ml_models_adv, default=['AutoARIMA (Optimized)', 'Global LightGBM (ML)', 'Ensemble (LGBM + ARIMA)'])

    calc_intervals = st.sidebar.checkbox("Generate Confidence Intervals (P10/P90)", value=True)

# --- SKU CONSOLIDATION (SHARED) ---
st.sidebar.markdown("---")
st.sidebar.subheader("🔄 SKU Consolidation")

def get_mapping_template():
    output = io.BytesIO()
    template_df = pd.DataFrame(columns=['Old SKU Code', 'New SKU Code'])
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        template_df.to_excel(writer, index=False, sheet_name='SKU_Mapping')
    return output.getvalue()

st.sidebar.download_button(
    label="Download SKU Mapping Template",
    data=get_mapping_template(),
    file_name="SKU_Mapping_Template.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
mapping_file = st.sidebar.file_uploader("Upload SKU Mapping Excel", type=['xlsx', 'xls'])

# --- LICENSE & INFO (SHARED) ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 📜 License & Info")
st.sidebar.info(
    "🐍 **Built with Python**\n\n"
    "This software is **open-source** and freely available for experimental use.\n\n"
    "💼 **For commercial use**, please contact:\n"
    "📧 [atillakdeniz@icloud.com](mailto:atillakdeniz@icloud.com)"
)

# --- AUTHOR (SHARED) ---
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style='text-align: center; color: #888; font-size: 12px;' 
         title='a humble demand planner'>
        Created by <b>Atilla AKDENİZ</b>
    </div>
    """, 
    unsafe_allow_html=True
)

st.title("📊 Mini Forecast Tool (MFT)")
if app_mode == "Advanced Mode":
    st.caption("🚀 Running in Enterprise Demand Planning Mode")
else:
    st.caption("🛠️ Running in Basic Statistical Mode")

# --- SALES TEMPLATE ---
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

# --- HELPER FUNCTIONS ---
def get_demand_segment(ts):
    non_zero = ts[ts > 0]
    if len(non_zero) == 0: return "Dead"
    adi = len(ts) / len(non_zero)
    cv2 = (np.std(non_zero) / np.mean(non_zero)) ** 2 if len(non_zero) > 1 else 0
    volatility = np.std(ts) / np.mean(ts) if np.mean(ts) > 0 else 0
    if adi <= 1.32 and cv2 <= 0.49: return "Smooth"
    elif adi <= 1.32 and cv2 > 0.49: return "Highly Volatile" if volatility > 1.0 else "Erratic"
    elif adi > 1.32 and cv2 <= 0.49: return "Intermittent"
    else: return "Lumpy"

def winsorize_series(ts, limit=3):
    mean, std = np.mean(ts), np.std(ts)
    if std == 0: return ts
    z_scores = (ts - mean) / std
    ts_capped = np.where(z_scores > limit, mean + limit * std, ts)
    ts_capped = np.where(z_scores < -limit, max(0, mean - limit * std), ts_capped)
    return ts_capped

def impute_stockouts(ts, segment):
    if segment in ["Smooth", "Erratic"] and len(ts) >= 3:
        ts_series = pd.Series(ts)
        rolling_mean = ts_series.replace(0, np.nan).rolling(window=3, min_periods=1, center=True).mean()
        ts_imputed = np.where(ts == 0, rolling_mean.fillna(0), ts)
        return ts_imputed
    return ts

def create_adv_features(df):
    df['Month'] = df['Datetime'].dt.month
    df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df = df.sort_values(['SKU Code', 'Datetime'])
    for lag in [1, 2, 3, 12]:
        df[f'Lag_{lag}'] = df.groupby('SKU Code')['Sales'].shift(lag)
    df['Rolling_Mean_3'] = df.groupby('SKU Code')['Lag_1'].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df['Rolling_Std_3'] = df.groupby('SKU Code')['Lag_1'].transform(lambda x: x.rolling(3, min_periods=1).std().fillna(0))
    df['Momentum'] = (df['Lag_1'] - df['Lag_2']) / (df['Lag_2'] + 1e-5)
    return df

def calculate_adv_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_pred = np.maximum(y_pred, 0)
    sum_true = np.sum(y_true)
    if sum_true == 0:
        wmape = 100.0 if np.sum(y_pred) > 0 else 0.0
        bias = 100.0 if np.sum(y_pred) > 0 else 0.0
    else:
        wmape = np.sum(np.abs(y_true - y_pred)) / sum_true * 100
        bias = (np.sum(y_pred) - sum_true) / sum_true * 100
    wmape = min(wmape, 999.0)
    return wmape, bias

def mape_loss_basic(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-9))

def croston_forecast(ts, horizon):
    a, n = np.array(ts), len(ts)
    p, q = np.zeros(n), np.zeros(n)
    alpha = 0.1
    last_p, last_q = 1.0, a[0] if a[0] > 0 else 1.0
    q_idx = 1
    for i in range(n):
        if a[i] > 0:
            last_q = alpha * a[i] + (1 - alpha) * last_q
            last_p = alpha * q_idx + (1 - alpha) * last_p
            q_idx = 1
        else: q_idx += 1
        p[i], q[i] = last_p, last_q
    return np.repeat(last_q / last_p, horizon)

def to_excel_download(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Forecast_Results')
    return output.getvalue()


# --- DATA UPLOAD & PREP ---
uploaded_file = st.file_uploader("Upload Sales Data (Excel format)", type=['xlsx', 'xls'])

if uploaded_file is not None:
    df_raw = pd.read_excel(uploaded_file)
    raw_row_count = len(df_raw)
    
    rename_dict = {'SKU Kodu': 'SKU Code', 'SKU Tanım': 'SKU Description', 'Tarih': 'Date', 'Satış KG': 'Sales'}
    df_raw = df_raw.rename(columns=rename_dict)
    if 'SKU Description' not in df_raw.columns: df_raw['SKU Description'] = "Unknown"
        
    df_raw['Datetime'] = pd.to_datetime(df_raw['Date'])
    global_max_date = df_raw['Datetime'].max() 
    
    desc_map = df_raw[['SKU Code', 'SKU Description']].drop_duplicates(subset=['SKU Code']).set_index('SKU Code')['SKU Description'].to_dict()
    df = df_raw.copy()
    if mapping_file is not None:
        map_df = pd.read_excel(mapping_file)
        map_rename = {'Eski SKU Kodu': 'Old SKU Code', 'Güncel SKU Kodu': 'New SKU Code'}
        map_df = map_df.rename(columns=map_rename)
        if 'Old SKU Code' in map_df.columns and 'New SKU Code' in map_df.columns:
            mapping_dict = dict(zip(map_df['Old SKU Code'], map_df['New SKU Code']))
            df['SKU Code'] = df['SKU Code'].replace(mapping_dict)
            st.success("SKUs successfully consolidated!")
            
    df = df.groupby(['SKU Code', 'Datetime'])['Sales'].sum().reset_index()
    df['SKU Description'] = df['SKU Code'].map(desc_map).fillna("Consolidated / Unknown")
    df = df.sort_values(['SKU Code', 'Datetime'])
    
    ui_choices = df['SKU Code'].astype(str) + " | " + df['SKU Description'].astype(str)
    ui_choices = np.sort(ui_choices.unique())
    selected_ui = st.multiselect("🎯 Select SKUs to Forecast [Leave blank for ALL]:", ui_choices)
    
    if st.button("🚀 Run Forecast"):
        timer_placeholder = st.empty()
        start_time = time.time()
        
        if len(selected_ui) > 0:
            selected_codes = [x.split(" | ")[0] for x in selected_ui]
            df = df[df['SKU Code'].isin(selected_codes)]
            
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Standardize timelines for BOTH modes
        all_skus = df['SKU Code'].unique()
        standardized_frames = []
        for sku in all_skus:
            group = df[df['SKU Code'] == sku].set_index('Datetime').resample('MS')['Sales'].sum().fillna(0)
            full_idx = pd.date_range(start=group.index.min(), end=global_max_date, freq='MS')
            group = group.reindex(full_idx, fill_value=0).reset_index().rename(columns={'index': 'Datetime'})
            group['SKU Code'] = sku
            standardized_frames.append(group)
            
        df = pd.concat(standardized_frames, ignore_index=True)
        results, failed_skus = [], []
        total_skus = len(all_skus)
        volume_by_sku = df.groupby('SKU Code')['Sales'].sum()

        # ==========================================
        #           BASIC MODE EXECUTION
        # ==========================================
        if app_mode == "Basic Mode":
            future_dates = [global_max_date + pd.DateOffset(months=i) for i in range(1, forecast_horizon + 1)]
            
            for idx, sku in enumerate(all_skus):
                elapsed = int(time.time() - start_time)
                mins, secs = divmod(elapsed, 60)
                timer_placeholder.markdown(f"⏱️ **Elapsed Time:** {mins} min {secs} sec")
                
                group = df[df['SKU Code'] == sku]
                sku_desc = desc_map.get(sku, 'Unknown')
                ts = group['Sales'].values
                
                if use_min_sales:
                    ts_series = pd.Series(ts)
                    continuous_sales = (ts_series > 0).rolling(window=min_sales_req).sum().max()
                    if continuous_sales < min_sales_req:
                        failed_skus.append({'SKU Code': sku, 'SKU Description': sku_desc, 'Reason': f'Failed Continuous Sales Rule'})
                        progress_bar.progress((idx + 1) / total_skus)
                        continue
                        
                ts_data = ts[-(train_months + test_months):]
                if len(ts_data) <= test_months + 2:
                    failed_skus.append({'SKU Code': sku, 'SKU Description': sku_desc, 'Reason': 'Insufficient Historical Data'})
                    progress_bar.progress((idx + 1) / total_skus)
                    continue
                    
                train_y = ts_data[:-test_months]
                test_y = ts_data[-test_months:]
                
                best_mape = float('inf')
                best_model_name = "Not Found"
                best_forecast = np.zeros(forecast_horizon)
                
                for m_name in available_models:
                    try:
                        preds = np.zeros(test_months)
                        f_preds = np.zeros(forecast_horizon)
                        
                        if m_name == 'Naive':
                            preds, f_preds = np.repeat(train_y[-1], test_months), np.repeat(train_y[-1], forecast_horizon)
                        elif m_name == 'Moving Average':
                            ma = pd.Series(train_y).rolling(3).mean().iloc[-1]
                            preds, f_preds = np.repeat(ma, test_months), np.repeat(ma, forecast_horizon)
                        elif m_name == 'Holt-Winters':
                            hw = ExponentialSmoothing(train_y, trend='add').fit()
                            preds, f_preds = hw.forecast(test_months), hw.forecast(forecast_horizon)
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
                            
                            if m_name == 'XGBoost': model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=50)
                            elif m_name == 'LightGBM': model = lgb.LGBMRegressor(n_estimators=50, verbose=-1)
                            else: model = CatBoostRegressor(iterations=50, verbose=0)
                            
                            model.fit(idx_train, train_y)
                            preds = model.predict(idx_test)
                            model.fit(idx_full, ts_data)
                            f_preds = model.predict(idx_future)
                        elif m_name in ['Croston', 'SBA', 'TSB']:
                            preds = croston_forecast(train_y, test_months)
                            f_preds = croston_forecast(ts_data, forecast_horizon)
                            
                        mape = mape_loss_basic(test_y, preds) * 100
                        if mape < best_mape:
                            best_mape = mape
                            best_model_name = m_name
                            best_forecast = f_preds
                    except: continue
                        
                best_forecast = np.where(best_forecast < 0, 0, best_forecast)
                best_forecast = np.round(best_forecast).astype(int)
                
                if np.all(best_forecast == 0) or best_model_name == "Not Found":
                    failed_skus.append({'SKU Code': sku, 'SKU Description': sku_desc, 'Reason': 'All models yielded Zero or Failed'})
                    progress_bar.progress((idx + 1) / total_skus)
                    continue
                    
                accuracy = max(0, 100 - best_mape)
                row_data = {
                    'SKU Code': sku,
                    'SKU Description': sku_desc,
                    'Selected Model': best_model_name,
                    'Validation MAPE (%)': round(best_mape, 2),
                    'V. Accuracy (%)': round(accuracy, 2),
                    'Volume': ts.sum()
                }
                
                for i, f_date in enumerate(future_dates):
                    col_name = f"{f_date.year}-{f_date.month:02d} Forecast"
                    row_data[col_name] = best_forecast[i] if i < len(best_forecast) else 0
                    
                results.append(row_data)
                progress_bar.progress((idx + 1) / total_skus)


        # ==========================================
        #          ADVANCED MODE EXECUTION
        # ==========================================
        elif app_mode == "Advanced Mode":
            status_text.text("⚙️ Preprocessing Data (Outliers, Stockouts, Segmentation)...")
            segments = {}
            for sku in all_skus:
                mask = df['SKU Code'] == sku
                ts = df.loc[mask, 'Sales'].values
                segment = get_demand_segment(ts)
                segments[sku] = segment
                if handle_outliers: ts = winsorize_series(ts)
                if handle_stockouts: ts = impute_stockouts(ts, segment)
                df.loc[mask, 'Sales'] = ts

            status_text.text("🧠 Training Global LightGBM Model across all SKUs...")
            df_feat = create_adv_features(df).dropna()
            train_df = df_feat[df_feat['Datetime'] <= global_max_date - pd.DateOffset(months=3)]
            test_df = df_feat[df_feat['Datetime'] > global_max_date - pd.DateOffset(months=3)]
            
            features = ['Lag_1', 'Lag_2', 'Lag_3', 'Lag_12', 'Rolling_Mean_3', 'Rolling_Std_3', 'Momentum', 'Month_Sin', 'Month_Cos']
            
            global_lgb = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, verbose=-1)
            if not train_df.empty: global_lgb.fit(train_df[features], train_df['Sales'])
            
            if calc_intervals and not train_df.empty:
                lgb_p10 = lgb.LGBMRegressor(objective='quantile', alpha=0.1, verbose=-1).fit(train_df[features], train_df['Sales'])
                lgb_p90 = lgb.LGBMRegressor(objective='quantile', alpha=0.9, verbose=-1).fit(train_df[features], train_df['Sales'])

            status_text.text("⏳ Running Local Models & Compiling Forecasts...")
            future_dates = [global_max_date + pd.DateOffset(months=i) for i in range(1, forecast_horizon + 1)]
            
            for idx, sku in enumerate(all_skus):
                elapsed = int(time.time() - start_time)
                mins, secs = divmod(elapsed, 60)
                timer_placeholder.markdown(f"⏱️ **Elapsed Time:** {mins} min {secs} sec")
                
                sku_df = df[df['SKU Code'] == sku].copy()
                ts = sku_df['Sales'].values
                sku_desc = desc_map.get(sku, 'Unknown')
                
                if use_min_sales:
                    ts_series = pd.Series(ts)
                    continuous_sales = (ts_series > 0).rolling(window=min_sales_req).sum().max()
                    if continuous_sales < min_sales_req:
                        failed_skus.append({'SKU Code': sku, 'SKU Description': sku_desc, 'Reason': f'Failed Continuous Sales Rule'})
                        progress_bar.progress((idx + 1) / total_skus)
                        continue

                if len(ts) < 15:
                    failed_skus.append({'SKU Code': sku, 'SKU Description': sku_desc, 'Reason': 'Insufficient Historical Data'})
                    progress_bar.progress((idx + 1) / total_skus)
                    continue
                    
                train_y = ts[:-3]
                test_y = ts[-3:]
                
                best_wmape = float('inf')
                best_model_name = "Naive (Fallback)"
                final_forecast = np.repeat(train_y[-1] if len(train_y)>0 else 0, forecast_horizon)
                
                metrics_dict, forecasts_dict = {}, {}
                
                preds_naive = np.repeat(train_y[-1], 3)
                f_preds_naive = np.repeat(ts[-1], forecast_horizon)
                metrics_dict['Naive'] = calculate_adv_metrics(test_y, preds_naive)
                
                if 'Global LightGBM (ML)' in available_models and not test_df[test_df['SKU Code'] == sku].empty:
                    sku_test_features = test_df[test_df['SKU Code'] == sku][features]
                    preds_lgb = global_lgb.predict(sku_test_features)
                    metrics_dict['Global LightGBM (ML)'] = calculate_adv_metrics(test_y, preds_lgb)
                    
                    temp_df = sku_df.copy()
                    for i in range(forecast_horizon):
                        temp_feat = create_adv_features(temp_df).iloc[-1:]
                        next_pred = global_lgb.predict(temp_feat[features])[0]
                        next_row = pd.DataFrame({'SKU Code': [sku], 'Datetime': [future_dates[i]], 'Sales': [max(0, next_pred)]})
                        temp_df = pd.concat([temp_df, next_row], ignore_index=True)
                    forecasts_dict['Global LightGBM (ML)'] = temp_df['Sales'].iloc[-forecast_horizon:].values

                if 'AutoARIMA (Optimized)' in available_models and segments[sku] in ["Smooth", "Highly Volatile", "Erratic"]:
                    try:
                        arima = pm.auto_arima(train_y, seasonal=True, m=12, stepwise=True, suppress_warnings=True, error_action="ignore", max_p=2, max_q=2)
                        preds_arima = arima.predict(n_periods=3)
                        metrics_dict['AutoARIMA (Optimized)'] = calculate_adv_metrics(test_y, preds_arima)
                        
                        arima_full = pm.auto_arima(ts, seasonal=True, m=12, stepwise=True, suppress_warnings=True, error_action="ignore")
                        forecasts_dict['AutoARIMA (Optimized)'] = arima_full.predict(n_periods=forecast_horizon)
                    except: pass

                if 'Croston (Intermittent)' in available_models and segments[sku] in ["Intermittent", "Lumpy"]:
                    preds_croston = croston_forecast(train_y, 3)
                    metrics_dict['Croston (Intermittent)'] = calculate_adv_metrics(test_y, preds_croston)
                    forecasts_dict['Croston (Intermittent)'] = croston_forecast(ts, forecast_horizon)

                if 'Ensemble (LGBM + ARIMA)' in available_models and 'Global LightGBM (ML)' in forecasts_dict and 'AutoARIMA (Optimized)' in forecasts_dict:
                    preds_ens = (preds_lgb + preds_arima) / 2
                    metrics_dict['Ensemble (LGBM + ARIMA)'] = calculate_adv_metrics(test_y, preds_ens)
                    forecasts_dict['Ensemble (LGBM + ARIMA)'] = (forecasts_dict['Global LightGBM (ML)'] + forecasts_dict['AutoARIMA (Optimized)']) / 2

                for m_name, mets in metrics_dict.items():
                    if mets[0] < best_wmape and m_name in available_models:
                        best_wmape = mets[0]
                        best_model_name = m_name
                        final_forecast = forecasts_dict.get(m_name, f_preds_naive)

                fva_score = metrics_dict['Naive'][0] - best_wmape 
                
                p10_vals = np.zeros(forecast_horizon)
                p90_vals = np.zeros(forecast_horizon)
                if calc_intervals and 'Global LightGBM (ML)' in forecasts_dict:
                    temp_feat = create_adv_features(temp_df).iloc[-forecast_horizon:]
                    p10_vals = np.maximum(0, lgb_p10.predict(temp_feat[features]))
                    p90_vals = np.maximum(final_forecast, lgb_p90.predict(temp_feat[features]))

                final_forecast = np.round(np.maximum(0, final_forecast)).astype(int)
                
                if np.all(final_forecast == 0):
                    failed_skus.append({'SKU Code': sku, 'SKU Description': sku_desc, 'Reason': 'All Forecasts yielded Zero'})
                    progress_bar.progress((idx + 1) / total_skus)
                    continue

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
                    row_data[f"{mon_str} P10"] = int(p10_vals[i]) if calc_intervals else 0
                    row_data[f"{mon_str} Final (P50)"] = final_forecast[i]
                    row_data[f"{mon_str} P90"] = int(p90_vals[i]) if calc_intervals else 0
                    
                results.append(row_data)
                progress_bar.progress((idx + 1) / total_skus)


        # ==========================================
        #             COMMON RESULTS RENDER
        # ==========================================
        final_elapsed = int(time.time() - start_time)
        f_mins, f_secs = divmod(final_elapsed, 60)
        timer_placeholder.success(f"✅ **Process Completed in:** {f_mins} min {f_secs} sec")
        status_text.empty()
        
        res_df = pd.DataFrame(results)
        fail_df = pd.DataFrame(failed_skus)
        
        st.markdown("---")
        st.markdown("### 📝 Execution Summary")
        st.write(f"- **Total Rows Analyzed:** {raw_row_count:,}")
        st.write(f"- **Total Unique SKUs Checked:** {total_skus:,}")
        st.write(f"- **Successfully Forecasted SKUs:** {len(results):,}")
        
        if not fail_df.empty:
            st.write(f"- **SKUs Excluded from Forecast:** {len(failed_skus):,}")
            st.markdown("#### Exclusion Reasons Summary:")
            for reason, count in fail_df['Reason'].value_counts().items():
                st.markdown(f"- **{count} SKUs**: *{reason}*")
                
        if not res_df.empty:
            st.markdown("---")
            
            if app_mode == "Basic Mode":
                res_df['Weighted Accuracy'] = res_df['V. Accuracy (%)'] * res_df['Volume']
                portfolio_accuracy = res_df['Weighted Accuracy'].sum() / (res_df['Volume'].sum() + 1e-5)
                st.success(f"🎯 **Portfolio Overall Validation Accuracy:** {round(portfolio_accuracy, 2)}%")
                
                model_stats = res_df.groupby('Selected Model').apply(
                    lambda x: pd.Series({
                        'SKU_Count': len(x),
                        'Model_WMAPE': (x['Validation MAPE (%)'] * x['Volume']).sum() / (x['Volume'].sum() + 1e-5)
                    })
                ).reset_index()
                model_stats['V. Accuracy (%)'] = (100 - model_stats['Model_WMAPE']).clip(lower=0).round(2)
                
                fig_model = px.pie(model_stats, values='SKU_Count', names='Selected Model', custom_data=['V. Accuracy (%)'], title="Model Contribution to Portfolio", hole=0.3)
                fig_model.update_traces(hovertemplate="<b>%{label}</b><br>%{value} SKUs (%{percent})<br>Model Accuracy: %{customdata[0]}%")
                st.plotly_chart(fig_model, use_container_width=True)
                
                display_df = res_df.drop(columns=['Volume', 'Weighted Accuracy']).copy()
                st.download_button("📥 Download Results as Excel", data=to_excel_download(display_df), file_name="Basic_Forecast_Results.xlsx")
                
                for col in ['Validation MAPE (%)', 'V. Accuracy (%)']:
                    display_df[col] = display_df[col].fillna(0).astype(str).str.replace('.', ',')
                for col in [c for c in display_df.columns if 'Forecast' in c]:
                    display_df[col] = display_df[col].fillna(0).apply(lambda x: f"{int(float(x)):,}".replace(',', '.'))
                st.dataframe(display_df)

            elif app_mode == "Advanced Mode":
                if not calc_intervals:
                    res_df = res_df.loc[:, ~res_df.columns.str.contains('P10|P90')]
                    
                st.markdown("### 📊 Interactive Planner View (Editable)")
                st.info("💡 **Planner Override:** You can manually edit the Final (P50) forecast columns below before exporting to ERP.")
                
                display_cols = [c for c in res_df.columns if c != 'Volume']
                edited_df = st.data_editor(res_df[display_cols], num_rows="dynamic", use_container_width=True)
                
                st.download_button("📥 Export Final Plan to Excel", data=to_excel_download(edited_df), file_name="Enterprise_Demand_Plan.xlsx")
                
                st.markdown("---")
                st.markdown("### 📈 Pipeline Diagnostics")
                col1, col2, col3 = st.columns(3)
                port_wmape = (res_df['Val. WMAPE (%)'] * res_df['Volume']).sum() / (res_df['Volume'].sum() + 1e-5)
                positive_fva = len(res_df[res_df['FVA (%)'] > 0])
                
                col1.metric("Total SKUs Processed", f"{len(res_df):,}")
                col2.metric("Portfolio WMAPE", f"{round(port_wmape, 1)}%")
                col3.metric("SKUs with Positive FVA", f"{positive_fva} ({round(positive_fva/len(res_df)*100)}%)")
                    
                col_a, col_b = st.columns(2)
                fig_eng = px.pie(res_df, names='Winning Engine', title='Engine Allocation Strategy', hole=0.3, color_discrete_sequence=px.colors.qualitative.Set2)
                col_a.plotly_chart(fig_eng, use_container_width=True)
                fig_seg = px.pie(res_df, names='Demand Segment', title='Enterprise Demand Segmentation', hole=0.3, color_discrete_sequence=px.colors.qualitative.Pastel)
                col_b.plotly_chart(fig_seg, use_container_width=True)

        else:
            st.warning("No results generated for the selected criteria.")
            
        if not fail_df.empty:
            st.markdown("---")
            with st.expander("🔍 Show Detailed Exclusion Log (Collapsible)"):
                log_text = "--- FORECAST EXCLUSION LOG ---\n"
                for _, r in fail_df.iterrows():
                    log_text += f"{r['SKU Code']} | {r['SKU Description']} -> {r['Reason']}\n"
                st.code(log_text, language='text')
