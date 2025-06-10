from flask import Flask, render_template, request
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model  
import numpy as np
import joblib
from keras import backend as k
from keras.saving import register_keras_serializable
import tensorflow.keras.backend as k


def r2_score(y_true, y_pred):
    ss_res = k.sum(k.square(y_true - y_pred))
    ss_total = k.sum(k.square(y_true - k.mean(y_true)))
    return 1 - ss_res / (ss_total + k.epsilon())


#model = load_model('models/final_model.keras')

model = load_model('models/final_model.keras', custom_objects={'r2_score': r2_score})


scaler_data = joblib.load('models/scaler_data.sav')
scaler_target = joblib.load('models/scaler_target.sav')

app = Flask(__name__)

@app.route('/')
def index():

    return render_template('Details.html')

@app.route('/getresult', methods=['POST'])

def getresult():
    gender = float(request.form['gender_text'])
    age = int(request.form['age'])
    tc = int(request.form['tc'])
    hdl = int(request.form['hdl'])
    smoke = int(request.form['smoke_text'])
    bpm = int(request.form['bpm_text'])
    diab = int(request.form['diab_text'])

    test_data = np.array([gender, age, tc, hdl, smoke, bpm, diab]).reshape(1,-1)

    test_data_scaled = scaler_data.transform(test_data)

    prediction = model.predict(test_data_scaled)

    prediction = scaler_target.inverse_transform(prediction)
    print(prediction)

    prediction = prediction[0][0]
    return render_template('Details.html', result=prediction)

app.run(debug=True)
