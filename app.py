from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# =========================
# LOAD MODEL & SCALER
# =========================
model = load_model("model/ckd_model.keras")
scaler = joblib.load("scaler.joblib")

# =========================
# ENCODING HELPERS
# =========================
def yes_no(val):
    return 1 if val.lower() == "yes" else 0

def normal_abnormal(val):
    return 1 if val.lower() == "normal" else 0

def present_not(val):
    return 1 if val.lower() == "present" else 0

def appetite(val):
    return 1 if val.lower() == "good" else 0

# =========================
# ROUTES
# =========================
@app.route('/')
def login():
    return render_template('login.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # ---- EXACT 24 FEATURES ----
            features_24 = [
                float(request.form['age']),
                float(request.form['bp']),
                float(request.form['sg']),
                float(request.form['al']),
                float(request.form['su']),

                normal_abnormal(request.form['rbc']),
                normal_abnormal(request.form['pc']),
                present_not(request.form['pcc']),
                present_not(request.form['ba']),

                float(request.form['bgr']),
                float(request.form['bu']),
                float(request.form['sc']),
                float(request.form['sod']),
                float(request.form['pot']),
                float(request.form['hemo']),
                float(request.form['pcv']),
                float(request.form['wc']),
                float(request.form['rc']),

                yes_no(request.form['htn']),
                yes_no(request.form['dm']),
                yes_no(request.form['cad']),
                appetite(request.form['appet']),
                yes_no(request.form['pe']),
                yes_no(request.form['ane'])
            ]

            # Scale (24)
            scaled = scaler.transform(np.array(features_24).reshape(1, -1))

            # Add dummy ID → 25 features (MODEL EXPECTS 25)
            final_input = np.insert(scaled, 0, 0, axis=1)

            # Predict
            pred = model.predict(final_input)[0][0]

            accuracy = "97.2%"

            if pred > 0.5:
                result = "CKD Detected"
                image = "ckd.jpg"
            else:
                result = "No CKD Detected"
                image = "no_ckd.jpg"

            graph = "prediction_graph.png"

            return render_template(
                "result.html",
                result=result,
                accuracy=accuracy,
                image=image,
                graph=graph
            )

        except Exception as e:
            return f"Error occurred: {e}"

    return render_template('predict.html')


if __name__ == "__main__":
    app.run(debug=True)
