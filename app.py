from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

stress_model = joblib.load("stress_model.pkl")
help_model = joblib.load("help_model.pkl")
kmeans = joblib.load("kmeans.pkl")
scaler = joblib.load("scaler.pkl")

FEATURES = [
    "Age_Group",
    "Gender",
    "Sleep_Hours",
    "Screen_Time",
    "Free_Time_Activity",
    "Exercise_Frequency",
    "Anxious_Exhausted_Frequency",
    "Current_Emotional_State"
]

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    # ✅ SAFETY: ensure all keys exist
    for col in FEATURES:
        if col not in data:
            return jsonify({"error": f"Missing field: {col}"}), 400

    # ✅ FIX 1: DataFrame instead of list
    X_df = pd.DataFrame([data], columns=FEATURES)

    stress = int(stress_model.predict(X_df)[0])
    help_seek = int(help_model.predict(X_df)[0])

    sleep_map = {
        "Less than 6 hours": 1,
        "6-8 hours": 2,
        "More than 8 hours": 3
    }

    anxiety_map = {
        "Never": 1,
        "Rarely": 2,
        "Sometimes": 3,
        "Often": 4,
        "Always": 5
    }

    cluster_input = np.array([[
        3 if stress else 1,
        sleep_map.get(data["Sleep_Hours"], 2),
        anxiety_map.get(data["Anxious_Exhausted_Frequency"], 3)
    ]])

    cluster = int(kmeans.predict(scaler.transform(cluster_input))[0])

    return jsonify({
        "stress_risk": "High" if stress else "Low",
        "help_recommended": "Yes" if help_seek else "No",
        "risk_cluster": cluster
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
