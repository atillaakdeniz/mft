# mft
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

[Raw Data Upload] ➡️ [SKU Mapping & Consolidation] ➡️ [Outlier & Stockout Treatment]
         ⬇️
[Global Feature Engineering Matrix] ➡️ [Model Tournament & Cross-Validation] ➡️

----------------------------------------------------------------------------------------------------------------------------------------------------

TÜRKÇE

# 📊 Mini Forecast Tool (MFT) - Enterprise Demand Planning Platform

MFT (Mini Forecast Tool), FMCG (Hızlı Tüketim Malları) sektöründeki talep planlama süreçlerini optimize etmek, basit istatistiksel tahmin modelleri ile modern küresel makine öğrenmesi (Global Machine Learning) mimarileri arasındaki açığı kapatmak için Python ve Streamlit ile geliştirilmiş **Kurumsal Düzeyde Bir Talep Tahminleme ve S&OP Platformudur.**

Yazılım, binlerce benzersiz stok kartını (SKU), promosyonel dalgalanmaları, yok-satma (stockout) geçmişini ve karmaşık zaman serisi dinamiklerini saniyeler içinde analiz ederek tedarik zinciri ekiplerine kararlarında rehberlik eder.

---

## 🕹️ Operasyonel Modlar (Dual-Engine Architecture)

MFT, kullanıcı gereksinimlerine göre şekillenebilen iki farklı tahmin motoruna sahiptir:

### 1. 🛠️ Basic Mode (Temel İstatistiksel Mod)
Hızlı, doğrusal ve klasik tedarik zinciri yaklaşımları için tasarlanmıştır. Her SKU için bağımsız çalışır.
* **Modeller:** Naive, Moving Average, Holt-Winters, Klasik SARIMA $(1,1,1)$, XGBoost, LightGBM, CatBoost, Croston, SBA, TSB.
* **Metrik:** Standart MAPE (Mean Absolute Percentage Error) kısıtlarına göre en iyi modeli seçer.
* **Girdi/Çıktı:** Doğrudan nokta tahmini (Point Forecast) üretir.

### 2. 🚀 Advanced Mode (Gelişmiş Kurumsal Mod)
Global FMCG devlerinin kullandığı gelişmiş zaman serisi mühendisliği ve olasılıksal tahminleme altyapısını sunar.
* **Gelişmiş Veri Önişleme:** Promosyonel sıçramalar için Z-Score tabanlı **Winsorization (Capping)**; yok-satma dönemleri için otomatik **Stockout Imputation** (Rolling Mean dolgusu).
* **Global ML Altyapısı:** Tüm ürün havuzunu tek bir panel veri setinde birleştirerek çalışan **Global LightGBM** motoru (Shared Learning).
* **Akıllı İstatistiksel Motor:** AIC/BIC optimizasyonlu adım tabanlı **AutoARIMA**.
* **Olasılıksal Tahminleme (Confidence Intervals):** Emniyet stoku ve risk yönetimi için **P10 (Kötümser), P50 (Hedef) ve P90 (İyimser)** senaryo çıktıları.
* **Katma Değer Analizi (FVA):** Gelişmiş modellerin basit kopyalamaya (Naive) kıyasla sağladığı **Forecast Value Added** başarısını ölçer.

---

## 🛠️ Mimari ve Veri Akışı (Data Pipeline)

```text
[Raw Data Upload] ➡️ [SKU Mapping & Consolidation] ➡️ [Outlier & Stockout Treatment]
         ⬇️
[Global Feature Engineering Matrix] ➡️ [Model Tournament & Cross-Validation] ➡️ [Interactive Planner Override]
