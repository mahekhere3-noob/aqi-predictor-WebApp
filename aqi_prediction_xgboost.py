import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

# Load Dataset
# Expected columns: PM2_5, PM10, NO2, SO2, CO, O3, Temperature, Humidity, AQI
data = pd.read_csv("aqi_data.csv")

# Features and Target
X = data[["PM2_5", "PM10", "NO2", "SO2", "CO", "O3", "Temperature", "Humidity"]]
y = data["AQI"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = XGBRegressor(
    n_estimators=250,
    max_depth=4,
    learning_rate=0.07,
    random_state=42
)
model.fit(X_train, y_train)

# Test Accuracy
prediction = model.predict(X_test)

print("Model R² Score:", r2_score(y_test, prediction))
print("Mean Absolute Error:", mean_absolute_error(y_test, prediction), "AQI points")

# User Prediction
pm25 = float(input("Enter PM2.5 (µg/m³): "))
pm10 = float(input("Enter PM10 (µg/m³): "))
no2 = float(input("Enter NO2 (µg/m³): "))
so2 = float(input("Enter SO2 (µg/m³): "))
co = float(input("Enter CO (mg/m³): "))
o3 = float(input("Enter O3 (µg/m³): "))
temperature = float(input("Enter Temperature (°C): "))
humidity = float(input("Enter Humidity (%): "))

new_data = [[pm25, pm10, no2, so2, co, o3, temperature, humidity]]

result = model.predict(new_data)

print(f"\nPredicted AQI: {result[0]:.1f}")
