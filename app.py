import streamlit as st
import joblib
import numpy as np

scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

st.title("Churn Prediction App")

age = st.number_input("Enter Age", min_value=18, max_value=100, value=30)
tenure = st.number_input("Enter Tenure", min_value=0, max_value=100, value=12)
monthlycharge = st.number_input("Enter Monthly Charges", min_value=30, max_value=150, value=70)
gender = st.selectbox("Enter Gender", ["Male", "Female"])

gender_encoded = 1 if gender == "Male" else 0

if st.button("Predict"):
    data = np.array([[age, tenure, monthlycharge, gender_encoded]])
    data = scaler.transform(data)
    prediction = model.predict(data)[0]

    if prediction == 1:
        st.error("Customer is likely to CHURN")
    else:
        st.success("Customer is likely to STAY")
