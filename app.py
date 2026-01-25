from flask import Flask, render_template, request
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ===============================
# Load trained model and scaler
# ===============================
model = load_model("model/ckd_model.keras")
scaler = joblib.load("model/scaler.pkl")


# ===============================
# Routes
# ===============================
@app.route('/')
def login():
    return render_template("login.html")


@app.route('/home')
def home():
    return render_template("home.html")


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':

        # Collect input features (ORDER MUST MATCH TRAINING)
        features = [
            0,  # dummy id (kept for consistency)

            float(request.form['age']),
            float(request.form['bp']),
            float(request.form['sg']),
            float(request.form['al']),
            float(request.form['su']),
            float(request.form['rbc']),
            float(request.form['pc']),
            float(request.form['pcc']),
            float(request.form['ba']),
            float(request.form['bgr']),
            float(request.form['bu']),
            float(request.form['sc']),
            float(request.form['sod']),
            float(request.form['pot']),
            float(request.form['hemo']),
            float(request.form['pcv']),
            float(request.form['wc']),
            float(request.form['rc']),
            float(request.form['htn']),
            float(request.form['dm']),
            float(request.form['cad']),
            float(request.form['appet']),
            float(request.form['pe']),
            float(request.form['ane'])
        ]

        # Convert to numpy array
        input_data = np.array(features).reshape(1, -1)

        # 🔹 Apply SAME scaler used during training
        input_data = scaler.transform(input_data)

        # 🔹 Prediction
        prediction = model.predict(input_data)[0][0]

        # Probabilities
        ckd_prob = prediction * 100
        normal_prob = 100 - ckd_prob

        # ===============================
        # Generate prediction graph
        # ===============================
        labels = ['CKD', 'Normal']
        values = [ckd_prob, normal_prob]

        plt.figure(figsize=(4,4))
        plt.bar(labels, values, color=['red', 'green'])
        plt.ylabel("Probability (%)")
        plt.title("CKD Prediction Probability")
        plt.ylim(0, 100)

        os.makedirs("static/images", exist_ok=True)
        plt.savefig("static/images/prediction_graph.png")
        plt.close()

        # Result text & image
        if prediction > 0.5:
            result = "CKD Detected"
            image = "ckd.jpg"
        else:
            result = "No CKD Detected"
            image = "no_ckd.jpg"

        return render_template(
            "result.html",
            result=result,
            accuracy="96.2%",
            image=image,
            graph="prediction_graph.png"
        )

    return render_template("predict.html")


# ===============================
# Required for local + Render
# ===============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
