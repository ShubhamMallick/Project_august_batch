from flask import Flask, request, render_template, send_from_directory
import pickle
import os
import numpy as np

application = Flask(__name__)
app = application

# Load scaler and model
scaler = pickle.load(open(r"C:\Users\aditya\Desktop\Pregrad\Pregrad_august\Model\standardScaler.pkl", "rb"))
model = pickle.load(open(r"C:\Users\aditya\Desktop\Pregrad\Pregrad_august\Model\modelForPrediction.pkl", "rb"))

# Favicon route
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, 'static'),
        'favicon.ico',
        mimetype='image/vnd.microsoft.icon'
    )

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    result = ""

    if request.method == 'POST':
        # Gather inputs from the form
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        BUI = float(request.form.get('BUI'))

        # Preprocess and predict
        new_data = scaler.transform([[Temperature, RH, Ws, FFMC, DMC, ISI, BUI]])
        predict = model.predict(new_data)

        if predict[0] == 1:
            result = 'Fire'
        else:
            result = 'No Fire'

        return render_template('home.html', result=result)
    else:
        return render_template('home.html', result=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
