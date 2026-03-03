from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os
from datetime import datetime

app = Flask(__name__)

# Folder where model files are stored
MODEL_PATH = os.path.join(os.getcwd(), 'models')

# Load model and encoders
model = joblib.load(os.path.join(MODEL_PATH, 'random_forest_model.pkl'))
station_enc = joblib.load(os.path.join(MODEL_PATH, 'station_encoder.pkl'))
type_enc = joblib.load(os.path.join(MODEL_PATH, 'type_encoder.pkl'))
target_enc = joblib.load(os.path.join(MODEL_PATH, 'target_encoder.pkl'))

print("✅ Model and encoders loaded successfully")

# ----------------------------
# Home Route
# ----------------------------
@app.route('/')
def index():
    stations_list = sorted(station_enc.classes_)
    types_list = sorted(type_enc.classes_)
    return render_template(
        'index.html',
        stations=stations_list,
        types=types_list
    )

# ----------------------------
# Prediction Route
# ----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Parse date
        journey_date = datetime.strptime(data['date'], '%Y-%m-%d')
        month = journey_date.month
        day_of_week = journey_date.weekday()

        # Encode categorical inputs
        origin_val = station_enc.transform([data['origin']])[0]
        dest_val = station_enc.transform([data['destination']])[0]
        type_val = type_enc.transform([data['type']])[0]

        # Feature order must match training
        features = [[
            month,
            day_of_week,
            origin_val,
            dest_val,
            type_val
        ]]

        # Get probability for DELAY class
        probs = model.predict_proba(features)[0]

        # Find index of DELAY class
        delay_index = list(target_enc.classes_).index("DELAY")

        delay_probability = probs[delay_index] * 100

        return jsonify({
            "probability": round(delay_probability, 2)
        })

    except Exception as e:
        print("Prediction Error:", e)
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':

    app.run()
