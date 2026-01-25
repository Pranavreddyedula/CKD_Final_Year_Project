import os
import pickle
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

# -----------------------------
# Flask App
# -----------------------------
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model", "ckd_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")

# -----------------------------
# Load model & scaler ONCE
# -----------------------------
model = load_model(MODEL_PATH)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def login():
    return render_template("login.html")


@app.route("/home")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            # ----------------------------------------
            # IMPORTANT: 25 FEATURES (MATCH TRAINING)
            # ----------------------------------------
            features = [
                0,  # ✅ Dummy ID column (required)

                float(request.form["age"]),
                float(request.form["bp"]),
                float(request.form["sg"]),
                float(request.form["al"]),
                float(request.form["su"]),
                float(request.form["bgr"]),
                float(request.form["bu"]),
                float(request.form["sc"]),
                float(request.form["sod"]),
                float(request.form["pot"]),
                float(request.form["hemo"]),
                float(request.form["pcv"]),
                float(request.form["wc"]),
                float(request.form["rc"]),
                float(request.form["rbc"]),
                float(request.form["pc"]),
                float(request.form["pcc"]),
                float(request.form["ba"]),
                float(request.form["htn"]),
                float(request.form["dm"]),
                float(request.form["cad"]),
                float(request.form["appet"]),
                float(request.form["pe"]),
                float(request.form["ane"]),
            ]

            # Convert to NumPy
            input_data = np.array(features).reshape(1, -1)

            # Scale input
            input_scaled = scaler.transform(input_data)

            # Predict
            prediction = model.predict(input_scaled)
            confidence = float(prediction[0][0]) * 100

            if prediction[0][0] >= 0.5:
                result = "CKD Detected"
                color = "danger"
                image = "ckd.jpg"
            else:
                result = "No CKD Detected"
                color = "success"
                image = "no_ckd.jpg"

            return render_template(
                "result.html",
                result=result,
                confidence=f"{confidence:.2f}",
                color=color,
                image=image,
            )

        except Exception as e:
            return f"Prediction error: {e}"

    return render_template("predict.html")


# -----------------------------
# Run App (Local only)
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
