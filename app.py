from flask import Flask, render_template, request, jsonify
import joblib
import os
import numpy as np
from datetime import datetime

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models")

model = joblib.load(os.path.join(MODEL_PATH, "random_forest_model.pkl"))
station_enc = joblib.load(os.path.join(MODEL_PATH, "station_encoder.pkl"))
type_enc = joblib.load(os.path.join(MODEL_PATH, "type_encoder.pkl"))
target_enc = joblib.load(os.path.join(MODEL_PATH, "target_encoder.pkl"))

DELAY_INDEX = list(target_enc.classes_).index("DELAY")

@app.route("/")
def index():
    stations_list = sorted(station_enc.classes_)
    types_list = sorted(type_enc.classes_)
    return render_template("index.html", stations=stations_list, types=types_list)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    journey_date = datetime.strptime(data["date"], "%Y-%m-%d")
    month = journey_date.month
    day_of_week = journey_date.weekday()

    origin_val = station_enc.transform([data["origin"]])[0]
    dest_val = station_enc.transform([data["destination"]])[0]
    type_val = type_enc.transform([data["type"]])[0]

    features = np.array([[month, day_of_week, origin_val, dest_val, type_val]])

    probs = model.predict_proba(features)[0]
    delay_probability = probs[DELAY_INDEX] * 100

    return jsonify({"probability": round(delay_probability, 2)})

@app.route("/health")
def health():
    return "OK", 200
