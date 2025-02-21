import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("air_quality_health_impact_data.csv")

# Ensure column names are clean
df.columns = df.columns.str.strip()

# Debugging: Print available columns
print("Available Columns:", df.columns)

# Check if required columns exist
required_features = ['Temperature', 'Humidity', 'WindSpeed']
required_targets = ['PM2_5', 'O3', 'NO2', 'PM10', 'SO2','AQI']

missing_features = [col for col in required_features if col not in df.columns]
missing_targets = [col for col in required_targets if col not in df.columns]

if missing_features:
    raise KeyError(f"Missing feature columns: {missing_features}")
if missing_targets:
    raise KeyError(f"Missing target columns: {missing_targets}")

# Feature and target selection
X = df[required_features]
y = df[required_targets]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "aqi_model.pkl")

print("Model training completed successfully.")
