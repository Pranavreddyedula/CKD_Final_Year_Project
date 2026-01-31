from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model & scaler
model = load_model("model/ckd_model.keras")
scaler = joblib.load("scaler.joblib")

binary_map = {
    "yes": 1, "no": 0,
    "present": 1, "notpresent": 0,
    "abnormal": 1, "normal": 0,
    "poor": 1, "good": 0
}

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
            features = [
                float(request.form['age']),
                float(request.form['bp']),
                float(request.form['sg']),
                float(request.form['al']),
                float(request.form['su']),
                binary_map[request.form['rbc'].lower()],
                binary_map[request.form['pc'].lower()],
                binary_map[request.form['pcc'].lower()],
                binary_map[request.form['ba'].lower()],
                float(request.form['bgr']),
                float(request.form['bu']),
                float(request.form['sc']),
                float(request.form['sod']),
                float(request.form['pot']),
                float(request.form['hemo']),
                float(request.form['pcv']),
                float(request.form['wc']),
                float(request.form['rc']),
                binary_map[request.form['htn'].lower()],
                binary_map[request.form['dm'].lower()],
                binary_map[request.form['cad'].lower()],
                binary_map[request.form['appet'].lower()],
                binary_map[request.form['pe'].lower()],
                binary_map[request.form['ane'].lower()]
            ]

            input_data = np.array(features).reshape(1, -1)
            input_scaled = scaler.transform(input_data)

            prediction = model.predict(input_scaled)
            result = "CKD Detected" if prediction[0][0] > 0.5 else "No CKD Detected"

            return render_template('result.html', result=result)

        except Exception as e:
            return f"Error occurred: {e}"

    return render_template('predict.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
