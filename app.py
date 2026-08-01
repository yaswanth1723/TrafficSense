"""
Metro Interstate Traffic Volume Predictor
==========================================
Streamlit application that serves the final, selected regression and
classification models trained in `02_Model_Training.ipynb`.

Only the production models are loaded:
    - models/xgb_regressor.pkl   -> predicts exact traffic_volume
    - models/xgb_classifier.pkl  -> predicts traffic_level (Low / Moderate / High)
"""

import datetime as dt

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------
st.set_page_config(
    page_title="Traffic Volume Predictor",
    page_icon="🚦",
    layout="centered",
)

# -----------------------------------------------------------------
# Constants — must mirror the preprocessing performed in
# 01_EDA_and_Preprocessing.ipynb / 02_Model_Training.ipynb
# -----------------------------------------------------------------
MODEL_FEATURE_ORDER = [
    "temp", "rain_1h", "snow_1h", "clouds_all", "is_holiday",
    "year", "month", "day", "hour", "day_of_week", "is_weekend",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "weather_Clouds", "weather_Drizzle", "weather_Fog", "weather_Haze",
    "weather_Mist", "weather_Rain", "weather_Smoke", "weather_Snow",
    "weather_Squall", "weather_Thunderstorm",
]

# "Clear" was the reference category dropped during one-hot encoding
# (drop_first=True), so it is represented by all weather_* columns == 0.
WEATHER_OPTIONS = [
    "Clear", "Clouds", "Drizzle", "Fog", "Haze",
    "Mist", "Rain", "Smoke", "Snow", "Squall", "Thunderstorm",
]

US_HOLIDAYS = [
    "None", "New Years Day", "Martin Luther King Jr Day", "Washingtons Birthday",
    "Memorial Day", "Independence Day", "State Fair", "Labor Day",
    "Columbus Day", "Veterans Day", "Thanksgiving Day", "Christmas Day",
]

TRAFFIC_LEVEL_MAP = {0: "Low", 1: "Moderate", 2: "High"}
TRAFFIC_LEVEL_COLOR = {"Low": "#2E7D32", "Moderate": "#F9A825", "High": "#C62828"}


# -----------------------------------------------------------------
# Model loading (cached so models are read from disk only once)
# -----------------------------------------------------------------
@st.cache_resource
def load_models():
    regressor = joblib.load("models/xgb_regressor.pkl")
    classifier = joblib.load("models/xgb_classifier.pkl")
    return regressor, classifier


def build_feature_row(
    input_date: dt.date,
    input_time: dt.time,
    temp_celsius: float,
    rain_1h: float,
    snow_1h: float,
    clouds_all: int,
    holiday: str,
    weather: str,
) -> pd.DataFrame:
    """Transform raw user input into the exact feature schema the models expect."""

    timestamp = dt.datetime.combine(input_date, input_time)

    year = timestamp.year
    month = timestamp.month
    day = timestamp.day
    hour = timestamp.hour
    day_of_week = timestamp.weekday()  # 0 = Monday
    is_weekend = int(day_of_week >= 5)
    is_holiday = int(holiday != "None")

    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    temp_kelvin = temp_celsius + 273.15

    row = {
        "temp": temp_kelvin,
        "rain_1h": rain_1h,
        "snow_1h": snow_1h,
        "clouds_all": clouds_all,
        "is_holiday": is_holiday,
        "year": year,
        "month": month,
        "day": day,
        "hour": hour,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "month_sin": month_sin,
        "month_cos": month_cos,
    }

    # One-hot encode weather (Clear -> all zeros, the dropped reference category)
    for option in WEATHER_OPTIONS:
        if option == "Clear":
            continue
        row[f"weather_{option}"] = int(weather == option)

    features_df = pd.DataFrame([row])
    return features_df[MODEL_FEATURE_ORDER]


# -----------------------------------------------------------------
# UI
# -----------------------------------------------------------------
st.title("🚦 TrafficSense")
st.subheader("AI-Powered Traffic Volume & Traffic Condition Prediction")

st.caption(
    "Predict hourly traffic volume and traffic congestion level using XGBoost-based machine learning models trained on the Metro Interstate Traffic Volume dataset."
)

regressor, classifier = load_models()

st.subheader("Input Parameters")

col1, col2 = st.columns(2)
with col1:
    input_date = st.date_input("Date", value=dt.date.today())
    holiday = st.selectbox("Holiday", options=US_HOLIDAYS, index=0)
    weather = st.selectbox("Weather", options=WEATHER_OPTIONS, index=0)
with col2:
    input_time = st.time_input("Time", value=dt.time(hour=8, minute=0))
    temp_celsius = st.slider("Temperature (°C)", min_value=-40.0, max_value=45.0, value=15.0, step=0.5)
    clouds_all = st.slider("Cloud Cover (%)", min_value=0, max_value=100, value=40)

col3, col4 = st.columns(2)
with col3:
    rain_1h = st.number_input("Rain in last 1 hour (mm)", min_value=0.0, max_value=200.0, value=0.0, step=0.1)
with col4:
    snow_1h = st.number_input("Snow in last 1 hour (mm)", min_value=0.0, max_value=50.0, value=0.0, step=0.1)

st.divider()

if st.button("Predict Traffic", type="primary", use_container_width=True):
    features_df = build_feature_row(
        input_date=input_date,
        input_time=input_time,
        temp_celsius=temp_celsius,
        rain_1h=rain_1h,
        snow_1h=snow_1h,
        clouds_all=clouds_all,
        holiday=holiday,
        weather=weather,
    )

    predicted_volume = regressor.predict(features_df)[0]
    predicted_level_encoded = classifier.predict(features_df)[0]
    predicted_level = TRAFFIC_LEVEL_MAP[int(predicted_level_encoded)]

    level_proba = classifier.predict_proba(features_df)[0]

    st.subheader("Prediction Results")

    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.metric("Predicted Traffic Volume", f"{predicted_volume:,.0f} vehicles/hour")
    with res_col2:
        st.markdown(
            f"""
            <div style="padding: 0.7rem 1rem; border-radius: 0.5rem;
                        background-color: {TRAFFIC_LEVEL_COLOR[predicted_level]}20;
                        border: 1px solid {TRAFFIC_LEVEL_COLOR[predicted_level]};">
                <span style="font-size: 0.85rem; color: gray;">Predicted Traffic Level</span><br>
                <span style="font-size: 1.6rem; font-weight: 700;
                             color: {TRAFFIC_LEVEL_COLOR[predicted_level]};">
                    {predicted_level}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("**Class probabilities**")
    proba_df = pd.DataFrame(
        {"Traffic Level": ["Low", "Moderate", "High"], "Probability": level_proba}
    ).set_index("Traffic Level")
    st.bar_chart(proba_df)

    with st.expander("View model input features"):
        st.dataframe(features_df.T.rename(columns={0: "value"}))

st.divider()
st.caption(
    "Models: XGBoost Regressor (traffic_volume) & XGBoost Classifier (traffic_level) — "
    "selected in `02_Model_Training.ipynb` based on test-set R² and weighted F1 score respectively."
)
