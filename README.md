# 🚦 TrafficSense
### AI-Powered Traffic Volume & Traffic Condition Prediction



## 🚀 Live Demo

**🌐 Web App:** https://trafficsense.streamlit.app/

**📂 GitHub Repository:** https://github.com/yaswanth1723/TrafficSense


![Python](https://img.shields.io/badge/Python-3.12-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-success)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)


An end-to-end machine learning project that predicts **hourly interstate traffic volume** and **traffic volume category** using historical weather and time data, deployed as an interactive Streamlit web application.

---

## 📌 Project Overview

This project builds a complete, production-style ML pipeline — from raw data to a deployed prediction app — around the Metro Interstate Traffic Volume dataset. It solves two related but distinct problems:

- **Regression** — predict the exact hourly traffic volume (vehicle count).
- **Classification** — predict a categorical traffic volume category (`Low`, `Moderate`, `High`).

The project follows a clean, reproducible structure: preprocessing and EDA are separated from model training, only the best-performing model per task is persisted, and the deployed app loads exclusively those final models.

---

## 📊 Dataset

**Source:** [Metro Interstate Traffic Volume Dataset](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume) (UCI Machine Learning Repository)

Hourly weather and traffic volume data for westbound I-94, collected between **2012 and 2018**, including:

| Column | Description |
|---|---|
| `holiday` | US national/regional holiday name (or none) |
| `temp` | Temperature (Kelvin) |
| `rain_1h` | Rainfall in the last hour (mm) |
| `snow_1h` | Snowfall in the last hour (mm) |
| `clouds_all` | Cloud cover (%) |
| `weather_main` | Short weather category (e.g. Clear, Rain, Snow) |
| `weather_description` | Detailed weather description |
| `date_time` | Timestamp of the reading (hourly) |
| `traffic_volume` | Hourly traffic volume (target) |

The raw dataset (48,204 rows) contained known data quality issues — a literal `"None"` string for non-holidays, duplicate timestamps, invalid `0 K` temperature readings, and a single extreme `rain_1h` outlier — all of which are addressed in the preprocessing notebook.

---

## 🎯 Problem Statement

Traffic congestion is influenced by a combination of time-based patterns (rush hour, weekday vs. weekend, seasonality) and weather conditions. This project frames traffic prediction as two supervised learning tasks:

1. **Regression:** `traffic_volume` — a continuous target, useful for precise flow estimation.
2. **Classification:** `traffic_level` — a derived 3-class target (`Low` / `Moderate` / `High`), useful for an independent estimate of the hourly volume category.

---

## 🧩 Features

After cleaning and feature engineering, the model-ready dataset includes:

- **Weather features:** `temp`, `rain_1h`, `snow_1h`, `clouds_all`, one-hot encoded `weather_main`
- **Calendar features:** `month`, `day`, `hour`, `day_of_week`, `is_weekend`, `is_holiday`
- **Cyclical time features:** `hour_sin`/`hour_cos`, `month_sin`/`month_cos` — encode hour and month on a circular scale so, e.g., 11 PM and midnight are recognized as adjacent
- **Derived target:** `traffic_level` (Low / Moderate / High), built from quantile (tercile) thresholds on `traffic_volume`

Full details are in [`01_EDA_and_Preprocessing.ipynb`](01_EDA_and_Preprocessing.ipynb).

---

## 🔍 Exploratory Data Analysis

Key findings from EDA (see the notebook for full visualizations):

- Traffic volume shows a strong **bimodal daily pattern** — clear morning and evening rush-hour peaks.
- Traffic volume is **consistently lower on weekends** and on **holidays**.
- Weather categories (e.g. `Snow`, `Fog`) visibly shift the traffic volume distribution, even though raw weather numerics have weak *linear* correlation with volume alone.
- The engineered `traffic_level` target is well balanced across all three classes after quantile-based thresholding.

---

## 🤖 Models Used

**Part A — Regression**
- Linear Regression *(baseline)*
- XGBoost Regressor

**Part B — Classification**
- Decision Tree
- Random Forest
- XGBoost Classifier

All models are trained on an 80/20 train-test split (stratified for classification) using the same leak-free feature set, with `traffic_volume`, `traffic_level`, and `traffic_level_encoded` mutually excluded from each other's feature set to prevent target leakage.

---

## 📈 Model Comparison

### Regression

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 809.58 | 1033.94 | 0.7271 |
| **XGBoost Regressor** | **205.86** | **328.88** | **0.9724** |

### Classification

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Decision Tree | 0.9013 | 0.9011 | 0.9013 | 0.9011 |
| Random Forest | 0.9119 | 0.9119 | 0.9119 | 0.9116 |
| **XGBoost Classifier** | **0.9193** | **0.9193** | **0.9193** | **0.9191** |

---

## 🏆 Final Models

| Task | Selected Model | Why |
|---|---|---|
| Regression | **XGBoost Regressor** | Highest R² (0.9724) and lowest MAE/RMSE among candidates — captures non-linear interactions between time and weather features that Linear Regression cannot. |
| Classification | **XGBoost Classifier** | Highest weighted F1 score (0.9191), balancing precision and recall across all three traffic-level classes better than Decision Tree or Random Forest. |

Only these two models are saved (`models/xgb_regressor.pkl`, `models/xgb_classifier.pkl`) and loaded by the deployed app — no other candidate models are persisted or shipped.

---



## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/yaswanth1723/TrafficSense.git
cd TrafficSense

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

To regenerate the models from scratch, run the notebooks in order:

```bash
jupyter notebook 01_EDA_and_Preprocessing.ipynb
jupyter notebook 02_Model_Training.ipynb
```

---

## 🚀 Running the Streamlit App

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (typically `http://localhost:8501`) in your browser. Enter a date, time, holiday, weather condition, and weather metrics to get an instant traffic volume and congestion level prediction.

---

## 🔮 Future Improvements

- Incorporate external features such as public holidays' regional variants, school calendars, or major local events.
- Add time-series cross-validation (e.g. `TimeSeriesSplit`) instead of a random split to better reflect real-world deployment where the model predicts on unseen future dates.
- Experiment with sequence models (e.g. LSTM, Temporal Fusion Transformer) to capture longer-range temporal dependencies.
- Add model monitoring/drift detection for the deployed app to track prediction quality over time.
- Containerize the app with Docker and add CI/CD for automated retraining and deployment.

---

## 📄 License

This project is open source and available for personal and educational use.


## Functional fixes and verification (20 October 2026)

Use the pinned XGBoost 3.3.0 runtime with the regenerated artifacts. Run the notebooks
using Python 3.12. In Streamlit Community Cloud, select Python 3.12 in App settings;
XGBoost 3.3.0 cannot install under Python 3.11. Run the notebooks
from the project root. Holiday markers now apply to all recorded hours on that date;
the app's holiday input remains a manual override. Regression evaluation and display
both clamp negative counts to zero.

Low means volume <= 2,274, Moderate means volume <= 4,598, and High means above 4,598.
These are volume categories, not measured congestion. The classifier is independent
of the regressor, so its estimate can disagree with the displayed volume's category.

The tables above describe the rebuilt artifacts on the original random 80/20 splits.
Preprocessing statistics and category boundaries still use the full dataset, and
the test split is still used to select candidates. Treat these as project comparison
scores, not an untouched final or future forecasting evaluation.

Run functional checks after installing requirements:

```bash
python -m unittest -v test_functional
```

See [AUDIT_FIXES.md](AUDIT_FIXES.md) for exact changes, results, and limitations.

### Prediction reliability warnings

The app warns when temperature, rain, snow or cloud cover is outside the regressor's
fitted training ranges. These bounds are saved from `X_train` during model training
in `models/training_ranges.json`. A separate warning explains any negative estimate
adjusted to zero: it is an unreliable estimate, not evidence of an empty road.
Inputs remain unchanged; the warnings do not silently cap weather values.

A `count:poisson` regressor was compared with the current squared-error settings on
identical 64% training / 16% validation / 20% test partitions. Validation MAE was
219.51 versus 208.73 and RMSE was 358.61 versus 339.61. Although Poisson produced
no negatives across the dataset or 4,752 stress inputs, it did not meet the
predeclared validation accuracy criteria, so the shipped regressor was retained.
Run `python compare_nonnegative.py` to reproduce the comparison; it never
overwrites model artifacts. Full results are in `nonnegative_comparison.json`.
