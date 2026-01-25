import os
import pickle
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import tensorflow as tf

# ---- TensorFlow safety for Render ----
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
tf.get_logger().setLevel("ERROR")

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model", "ckd_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")

# ---- Load model ONCE ----
model = load_model(MODEL_PATH, compile=False)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            features = [
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

            X = np.array(features).reshape(1, -1)
            X_scaled = scaler.transform(X)

            pred = model.predict(X_scaled, verbose=0)[0][0]
            confidence = round(pred * 100, 2)

            result = "CKD Detected" if pred > 0.5 else "No CKD Detected"

            return render_template(
                "result.html",
                result=result,
                confidence=confidence
            )

        except Exception as e:
            return f"Prediction error: {e}"

    return render_template("predict.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
