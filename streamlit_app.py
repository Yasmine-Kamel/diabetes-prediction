import streamlit as st
import joblib
import numpy as np
model = joblib.load('model/svm_diabetes_model.joblib')
scaler = joblib.load('model/scaler.joblib')

st.title('Diabetes Prediction')
pregnancies = st.number_input('Pregnancies', min_value=0)
glucose = st.number_input('Glucose', min_value=0)
bp = st.number_input('BloodPressure', min_value=0)
skin = st.number_input('SkinThickness', min_value=0)
insulin = st.number_input('Insulin', min_value=0)
bmi = st.number_input('BMI', min_value=0.0, format='%f')
dpf = st.number_input('DiabetesPedigreeFunction', min_value=0.0, format='%f')
age = st.number_input('Age', min_value=0)

if st.button('Predict'):
    input_data = np.array([pregnancies,glucose,bp,skin,insulin,bmi,dpf,age]).reshape(1,-1)
    input_scaled = scaler.transform(input_data)
    pred = model.predict(input_scaled)
    st.write('Diabetic' if pred[0]==1 else 'Not diabetic')