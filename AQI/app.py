from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import logging

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the trained model
try:
    model = joblib.load("aqi_model.pkl")
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    model = None

# Define city-wise dummy data (Replace with real-time data in future)
city_data = {
    "Delhi": {"temperature": 30, "humidity": 40, "wind_speed": 5},
    "Mumbai": {"temperature": 28, "humidity": 70, "wind_speed": 8},
    "Kolkata": {"temperature": 32, "humidity": 60, "wind_speed": 4},
    "Chennai": {"temperature": 29, "humidity": 75, "wind_speed": 6}
}

# Mapping input keys to match model's training feature names
FEATURE_MAPPING = {
    "temperature": "Temperature",
    "humidity": "Humidity",
    "wind_speed": "WindSpeed"
}

@app.route("/predict", methods=["GET"])
def predict():
    if model is None:
        return jsonify({"error": "Model not available"}), 500

    city = request.args.get("city")
    date = request.args.get("date")
    time = request.args.get("time")

    if not city or city not in city_data:
        return jsonify({"error": "Invalid or missing city parameter"}), 400

    try:
        # Prepare input data
        input_data = city_data[city]
        input_df = pd.DataFrame([input_data])

        # Rename columns to match the model's trained feature names
        input_df.rename(columns=FEATURE_MAPPING, inplace=True)

        # Make Prediction
        prediction = model.predict(input_df)[0]

        response = {
            "city": city,
            "date": date,
            "time": time,
            "pm25": round(prediction[0], 2),
            "co": round(prediction[1], 2),
            "no2": round(prediction[2], 2),
            "pm10": round(prediction[3], 2),
            "so2": round(prediction[4], 2)
        }

        return jsonify(response)

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
