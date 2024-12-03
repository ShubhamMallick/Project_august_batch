from flask import Flask, request, render_template
import pickle
import numpy as np

# Initialize Flask application
application = Flask(__name__)
app = application

# Load the scaler and model
scaler = pickle.load(open(r"C:\Users\aditya\Desktop\Pregrad\Pregrad_august\Model\standardScaler.pkl", "rb"))
model = pickle.load(open(r"C:\Users\aditya\Desktop\Pregrad\Pregrad_august\Model\modelForPrediction.pkl", "rb"))

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route for prediction
@app.route('/predictdata', methods=['POST'])
def predict_datapoint():
    result = ""

    if request.method == 'POST':
        try:
            # Retrieve form data
            Pregnancies = int(request.form.get('Pregnancies'))
            Glucose = float(request.form.get('Glucose'))
            BloodPressure = float(request.form.get('BloodPressure'))
            SkinThickness = float(request.form.get('SkinThickness'))
            Insulin = float(request.form.get('Insulin'))
            BMI = float(request.form.get('BMI'))
            DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
            Age = float(request.form.get('Age'))

            # Transform data and make prediction
            new_data = scaler.transform([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
            predict = model.predict(new_data)

            # Determine result
            result = 'Diabetic' if predict[0] == 1 else 'Non-Diabetic'
        except Exception as e:
            result = f"Error occurred: {e}"

    return render_template('single_prediction.html', result=result)

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
