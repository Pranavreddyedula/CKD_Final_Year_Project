import os
import pickle
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ---------- PATHS ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "ckd_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")

# ---------- LOAD MODEL & SCALER ONCE ----------
model = load_model(MODEL_PATH)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# ---------- ROUTES ----------
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
            # ✅ EXACT 25 FEATURES (MODEL EXPECTS THIS)
            features = [
                0,  # dummy id column

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

            input_data = np.array(features).reshape(1, -1)

            # scale input
            input_scaled = scaler.transform(input_data)

            prediction = model.predict(input_scaled)[0][0]
            confidence = round(prediction * 100, 2)

            result = "CKD Detected" if prediction > 0.5 else "No CKD Detected"

            return render_template(
                "result.html",
                result=result,
                confidence=confidence
            )

        except Exception as e:
            return f"Prediction error: {e}"

    return render_template("predict.html")


# ---------- RUN ----------
if __name__ == "__main__":
    app.run()
