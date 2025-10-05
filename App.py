# App.py

import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load model
model = pickle.load(open("Customer_churn_model.pkl", "rb"))

st.set_page_config(page_title="Customer Churn Prediction",  page_icon="📉", layout="centered")

st.title("Customer Churn Prediction App")
st.write("Predict whether a customer is likely to churn (leave) based on their service details.")

# Input form
st.header("Enter Customer Details")

gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Has Partner", ["Yes", "No"])
dependents = st.selectbox("Has Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
monthly = st.number_input("Monthly Charges ($)", 20.0, 120.0, 50.0)
total = st.number_input("Total Charges ($)", 0.0, 8000.0, 500.0)

# Create input dataframe
input_data = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [senior],
    "Partner": [partner],
    "Dependents": [dependents],
    "tenure": [tenure],
    "PhoneService": [phone_service],
    "InternetService": [internet],
    "Contract": [contract],
    "PaymentMethod": [payment],
    "MonthlyCharges": [monthly],
    "TotalCharges": [total]
})

# Load the original dataset to ensure same dummy columns
df = pd.read_csv("customer_churn.csv")
df_encoded = pd.get_dummies(df, drop_first=True)

# Get all feature columns used in training
X_columns = df_encoded.drop("Churn_Yes", axis=1).columns

# Apply same one-hot encoding to input
input_encoded = pd.get_dummies(input_data, drop_first=True)

# Reindex input data to match training columns (fill missing columns with 0)
input_encoded = input_encoded.reindex(columns=X_columns, fill_value=0)

# Predict button
if st.button("Predict Churn"):
    prediction = model.predict(input_encoded)[0]
    if prediction == 1:
        st.error("The customer is likely to CHURN.")
    else:
        st.success("The customer is likely to STAY.")
