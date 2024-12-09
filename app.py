from flask import Flask, request, render_template
import pickle
import numpy as np

application = Flask(__name__)
app = application

# Load scaler and model
scaler = pickle.load(open(r"C:\Users\aditya\Desktop\Pregrad\Pregrad_august\Model\standardScaler.pkl", "rb"))
model = pickle.load(open(r"C:\Users\aditya\Desktop\Pregrad\Pregrad_august\Model\modelForPrediction.pkl", "rb"))

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
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        BUI = float(request.form.get('BUI'))
        FWI = float(request.form.get('FWI'))

        # Preprocess and predict
        new_data = scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, BUI, FWI]])
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
