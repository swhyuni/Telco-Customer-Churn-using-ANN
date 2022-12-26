import streamlit as st
import pandas as pd
import requests


def run():
    with st.form(key='form_parameters'):
        customerID = st.text_input('customerID')
        gender = st.radio('gender', ('Male', 'Female'))
        SeniorCitizen = st.radio('SeniorCitizen', ('Yes', 'No'))
        Partner = st.radio('Partner', ('Yes', 'No'))
        Dependents = st.radio('Dependents', ('Yes', 'No'))
        tenure = st.number_input('tenure', min_value=0, value=1)
        PhoneService = st.radio('PhoneService', ('Yes', 'No'))
        MultipleLines = st.radio('MultipleLines', ('Yes', 'No', 'No phone service'))
        InternetService = st.radio('InternetService', ('No', 'DSL', 'Fiber optic'))
        OnlineSecurity = st.radio('OnlineSecurity', ('Yes', 'No', 'No internet service'))
        OnlineBackup = st.radio('OnlineBackup', ('Yes', 'No', 'No internet service'))
        DeviceProtection = st.radio('DeviceProtection', ('Yes', 'No', 'No internet service'))
        TechSupport = st.radio('TechSupport', ('Yes', 'No', 'No internet service'))
        StreamingTV = st.radio('StreamingTV', ('Yes', 'No', 'No internet service'))
        StreamingMovies = st.radio('StreamingMovies', ('Yes', 'No', 'No internet service'))
        Contract = st.radio('Contract', ('Month-to-month', 'One year', 'Two year'))
        PaperlessBilling = st.radio('PaperlessBilling', ('Yes', 'No'))
        PaymentMethod = st.radio('PaymentMethod', ('Electronic check', 'Mailed check','Bank transfer (automatic)','Credit card (automatic)'))
        MonthlyCharges = st.number_input('MonthlyCharges', min_value=1, value=1)
        TotalCharges= st.text_input('TotalCharges')

        submitted = st.form_submit_button('Predict')

    # Create A New Data
    data_inf = {
        'customerID' : customerID,
        'gender' : gender,
        'SeniorCitizen' : SeniorCitizen,
        'Partner' : Partner,
        'Dependents' : Dependents,
        'tenure' : tenure,
        'PhoneService' : PhoneService,
        'MultipleLines' : MultipleLines,
        'InternetService' : InternetService,
        'OnlineSecurity' : OnlineSecurity,
        'OnlineBackup' : OnlineBackup,
        'DeviceProtection' : DeviceProtection,
        'TechSupport' : TechSupport,
        'StreamingTV' : StreamingTV,
        'StreamingMovies' : StreamingMovies,
        'Contract' : Contract,
        'PaperlessBilling' : PaperlessBilling,
        'PaymentMethod' : PaymentMethod,
        'MonthlyCharges' : MonthlyCharges,
        'TotalCharges' : TotalCharges
    }

    if submitted:
        #Show Inference DataFrame
        st.dataframe(pd.DataFrame([data_inf]))
        print('[DEBUG] Data Inference : \n', data_inf)

        # Predict
        URL = 'https://backend-customers-churn-swhyuni.koyeb.app/predict'
        r = requests.post(URL, json=data_inf)

        if r.status_code == 200:
            res = r.json()
            st.write('## Prediction : ', res['label_names'])
            print('[DEBUG] Result : ', res)
            print('')
        else:
            st.write('Error with status code ', str(r.status_code))
        

if __name__ == '__main__':
    run()




