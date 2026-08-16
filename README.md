# 🌫️ AQI Predictor

A web app that predicts the Air Quality Index (AQI) from pollutant concentrations and weather readings, using a trained XGBoost regression model.

Built as part of a hands-on machine learning mini-project series.

---

## 🔍 Overview

Enter pollutant levels (PM2.5, PM10, NO2, SO2, CO, O3) and weather conditions (temperature, humidity), and the app predicts the AQI value in real time — then classifies it into its official EPA category (Good → Hazardous) with a color-coded result.

**[ Screenshot pending — insert a screenshot of the running app here ]**

---

## ✨ Features

- Real-time AQI prediction from 8 pollutant + weather inputs
- Color-coded result matching the official US EPA AQI scale
- Category breakdown showing exactly where a prediction falls (Good, Moderate, Unhealthy, etc.)
- Sidebar with live model metadata — algorithm, dataset size, R² score, and Mean Absolute Error
- Clean two-panel dashboard UI, no ML background required to use it

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| Model | XGBoost (`XGBRegressor`) |
| Data handling | pandas |
| Evaluation | scikit-learn |
| Web app | Streamlit |

---

## 📊 Dataset

- 400 air quality readings
- Features: `PM2_5`, `PM10`, `NO2`, `SO2`, `CO`, `O3`, `Temperature`, `Humidity`
- Target: `AQI`

---

## 🤖 Model

```python
XGBRegressor(
    n_estimators=250,
    max_depth=4,
    learning_rate=0.07,
    random_state=42
)
```

Trained with an 80/20 train-test split. Performance metrics (R², MAE) are computed live on startup and shown in the app's sidebar.

> **Note:** AQI values in the training data range from roughly 15 to 186. Predictions for extreme pollutant readings well outside that range are extrapolations and should be treated with caution.

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/your-username/aqi-predictor.git
cd aqi-predictor
```

### 2. Install dependencies

```bash
pip install streamlit pandas scikit-learn xgboost
```

### 3. Run the app

```bash
streamlit run app.py
```

The app opens automatically in your browser at `http://localhost:8501`.

---

## 📁 Project Structure

```
aqi-predictor/
├── app.py                        # Streamlit web app
├── aqi_prediction_xgboost.py     # Original standalone training script
├── aqi_data.csv                  # Dataset
└── .streamlit/
    └── config.toml               # App theme
```

---

## ⚠️ Limitations

- Trained on a relatively small dataset (400 rows) — not validated against real-world air quality monitoring data
- Predictions are illustrative and **not intended for public health or official air quality decisions**
- Accuracy degrades for pollutant readings far outside the training data's range

---

## 🎓 Credits

Built as part of a machine learning project series, with guidance from **[Teacher's name here]**.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
