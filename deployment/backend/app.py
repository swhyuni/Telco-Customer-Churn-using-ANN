import pandas as pd 
import numpy as np
import joblib
import json
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model

# App Initialization
app = Flask(__name__)

# Load The Models
with open('model_scaler.pkl', 'rb') as file_1:
  model_scaler = joblib.load(file_1)

with open('model_encoder.pkl', 'rb') as file_2:
  model_encoder = joblib.load(file_2)

with open('num_columns.txt', 'r') as file_3:
  num_columns = json.load(file_3)

with open('cat_columns.txt', 'r') as file_4:
  cat_columns = json.load(file_4)

model_ann = load_model('WA_Fn-UseC_-Telco-Customer-Churn.h5')

# Route : Homepage
@app.route('/')
def home():
    return '<h1> It Works <h1>'

@app.route('/predict', methods=['POST'])
def Churn_predict():
    args = request.json

    data_inf = {
        'customerID' : args.get('customerID'),
        'gender' : args.get('gender'),
        'SeniorCitizen' : args.get('SeniorCitizen'),
        'Partner' : args.get('Partner'),
        'Dependents' : args.get('Dependents'),
        'tenure' : args.get('tenure'),
        'PhoneService' : args.get('PhoneService'),
        'MultipleLines' : args.get('MultipleLines'),
        'InternetService' : args.get('InternetService'),
        'OnlineSecurity' : args.get('OnlineSecurity'),
        'OnlineBackup' : args.get('OnlineBackup'),
        'DeviceProtection' : args.get('DeviceProtection'),
        'TechSupport' : args.get('TechSupport'),
        'StreamingTV' : args.get('StreamingTV'),
        'StreamingMovies' : args.get('StreamingMovies'),
        'Contract' : args.get('Contract'),
        'PaperlessBilling' : args.get('PaperlessBilling'),
        'PaymentMethod' : args.get('PaymentMethod'),
        'MonthlyCharges' : args.get('MonthlyCharges'),
        'TotalCharges' : args.get('TotalCharges'),
        'Churn' : args.get('Churn')
    }

    print('[DEBUG] Data Inference: ', data_inf)

    # Transform inference set
    data_inf = pd.DataFrame([data_inf])
    
    # split between numerical and categorical
    # split between numerical and categorical
    data_inf_num = data_inf[num_columns]
    data_inf_cat = data_inf[cat_columns]

    # Feature Scaling & encoding
    data_inf_num_scaled = model_scaler.transform(data_inf_num)
    data_inf_cat_encoded = model_encoder.transform(data_inf_cat)

    # concate num & cat 
    data_inf_final = np.concatenate([data_inf_num_scaled, data_inf_cat_encoded], axis=1)

    # membuat dataframe
    data_inf_final_df = pd.DataFrame(data_inf_final, columns=[num_columns+cat_columns])
    data_inf_final_df2 = data_inf_final_df[['Contract', 'OnlineSecurity', 'TechSupport', 'OnlineBackup','tenure','DeviceProtection', 'SeniorCitizen']]

    # Predict using Neural Network
    y_pred_inf = model_ann.predict(data_inf_final_df2)
    y_pred_inf = np.where(y_pred_inf >= 0.5, 1, 0)

    if y_pred_inf == 0:
        label = 'NOT Churn'
    else:
        label = 'Churn'

    print('[DEBUG] Result : ', y_pred_inf, label)
    print('')

    response = jsonify(
        result = str(y_pred_inf),
        label_names = label
    )

    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0')


    

    


