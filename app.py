"""
AQI Predictor — Streamlit Web App
------------------------------------------------
A dashboard UI for the XGBoost air quality index model.

BEFORE YOU RUN THIS:
  1. Install dependencies:
       pip install streamlit pandas scikit-learn xgboost
  2. Make sure "aqi_data.csv" is in this same folder.
  3. Run with:
       streamlit run app.py
     (NOT "python app.py" — Streamlit apps must be launched with the
     "streamlit run" command.)
"""

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

# ---------------------------------------------------------------------------
# Page config + styling
# ---------------------------------------------------------------------------

st.set_page_config(page_title="AQI Predictor", page_icon="🌫️", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #F4F7F8; }
    section[data-testid="stSidebar"] { background-color: #E4EBEE; }
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #D3DEE3;
        border-radius: 10px;
        padding: 16px;
    }
    div.stButton > button {
        background-color: #2E6E8E;
        color: #F4F7F8;
        font-weight: 600;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #3E86AA;
        color: #F4F7F8;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Train the model (cached so it only trains once per session)
# ---------------------------------------------------------------------------

FEATURES = ["PM2_5", "PM10", "NO2", "SO2", "CO", "O3", "Temperature", "Humidity"]


@st.cache_resource
def load_model():
    df = pd.read_csv("aqi_data.csv")

    X = df[FEATURES]
    y = df["AQI"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(n_estimators=250, max_depth=4, learning_rate=0.07, random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    return model, r2, mae, len(df)


model, r2, mae, n_rows = load_model()


# ---------------------------------------------------------------------------
# AQI category scale (US EPA breakpoints)
# ---------------------------------------------------------------------------

CATEGORIES = [
    ("Good", 0, 50, "#4CAF50"),
    ("Moderate", 51, 100, "#D9A400"),
    ("Unhealthy for Sensitive Groups", 101, 150, "#FF9800"),
    ("Unhealthy", 151, 200, "#E53935"),
    ("Very Unhealthy", 201, 300, "#8E24AA"),
    ("Hazardous", 301, 500, "#6D1B23"),
]


def categorize(value):
    for name, lo, hi, color in CATEGORIES:
        if lo <= value <= hi:
            return name, color
    return "Hazardous", "#6D1B23"


# ---------------------------------------------------------------------------
# Sidebar — Model Control
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🌫️ Model Control")
    st.markdown("### Model Info")
    st.markdown("🔹 **Algorithm:** XGBoost Regressor")
    st.markdown(f"🔹 **Dataset:** Air Quality Readings ({n_rows} rows)")
    st.markdown(f"🔹 **Test R² Score:** {r2:.3f}")
    st.markdown(f"🔹 **Test MAE:** {mae:.1f} AQI points")

    st.markdown("---")
    st.markdown("### Settings")
    show_scale = st.checkbox("Show AQI category scale", value=True)


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

st.title("Air Quality Index Predictor")
st.caption("Enter pollutant and weather readings to estimate the Air Quality Index.")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Pollutants")
    pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, max_value=500.0, value=88.0, step=1.0)
    pm10 = st.number_input("PM10 (µg/m³)", min_value=0.0, max_value=600.0, value=119.0, step=1.0)
    no2 = st.number_input("NO2 (µg/m³)", min_value=0.0, max_value=200.0, value=54.0, step=1.0)
    so2 = st.number_input("SO2 (µg/m³)", min_value=0.0, max_value=100.0, value=9.0, step=0.5)
    co = st.number_input("CO (mg/m³)", min_value=0.0, max_value=20.0, value=0.7, step=0.1)
    o3 = st.number_input("O3 (µg/m³)", min_value=0.0, max_value=250.0, value=55.0, step=1.0)

with col2:
    st.markdown("### Weather")
    temperature = st.slider("Temperature (°C)", -10.0, 50.0, 26.0)
    humidity = st.slider("Humidity (%)", 0.0, 100.0, 69.0)

st.markdown("---")

if st.button("🌫️ Predict AQI", type="primary"):
    new_data = pd.DataFrame(
        [[pm25, pm10, no2, so2, co, o3, temperature, humidity]],
        columns=FEATURES,
    )
    pred = float(model.predict(new_data)[0])
    label, color = categorize(pred)

    st.markdown(
        f"""
        <div style="background-color:{color}22; border:2px solid {color};
                    border-radius:12px; padding:26px; text-align:center; margin-top:10px;">
            <div style="font-size:13px; letter-spacing:2px; color:#4B5A63; text-transform:uppercase;">
                Predicted AQI
            </div>
            <div style="font-size:58px; font-weight:bold; color:{color}; line-height:1.2;">
                {pred:.1f}
            </div>
            <div style="font-size:20px; font-weight:600; color:{color};">
                {label}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if show_scale:
        st.markdown("#### AQI Category Scale")
        cols = st.columns(len(CATEGORIES))
        for c, (name, lo, hi, color) in zip(cols, CATEGORIES):
            is_current = name == label
            border = f"2px solid {color}" if is_current else "1px solid #D3DEE3"
            bg = f"{color}22" if is_current else "#FFFFFF"
            hi_label = f"{hi}" if hi < 500 else "500+"
            c.markdown(
                f"""
                <div style="background:{bg}; border:{border}; border-radius:8px;
                            padding:8px 4px; text-align:center;">
                    <div style="font-size:11px; font-weight:700; color:{color};">{name}</div>
                    <div style="font-size:10px; color:#4B5A63;">{lo}\u2013{hi_label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
