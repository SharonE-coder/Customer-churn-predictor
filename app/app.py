import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Load Model and Scaler

@st.cache_resource
def load_model():
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_path, "../models/best_model.pkl")
        scaler_path = os.path.join(base_path, "../models/scaler.pkl")

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"Model or scaler file not found: {e}")
        st.stop()

model, scaler = load_model()

# Streamlit UI

st.title("💡 Customer Churn Predictor")
st.write("Predict whether a customer is likely to churn based on their details.")

# Collect User Input

gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", ["No", "Yes"])
tenure = st.number_input("Tenure (months)", min_value=0, max_value=100)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0)
total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0)
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
)


# Prepare Input Data

input_data = pd.DataFrame(
    {
        "tenure": [tenure],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges],
        "gender_Male": [1 if gender == "Male" else 0],
        "SeniorCitizen": [1 if senior == "Yes" else 0],
        "Contract_One year": [1 if contract == "One year" else 0],
        "Contract_Two year": [1 if contract == "Two year" else 0],
        "InternetService_Fiber optic": [1 if internet_service == "Fiber optic" else 0],
        "InternetService_No": [1 if internet_service == "No" else 0],
        "PaymentMethod_Credit card (automatic)": [
            1 if payment_method == "Credit card (automatic)" else 0
        ],
        "PaymentMethod_Electronic check": [
            1 if payment_method == "Electronic check" else 0
        ],
        "PaymentMethod_Mailed check": [
            1 if payment_method == "Mailed check" else 0
        ],
    }
)

# Align input with model columns
model_features = model.feature_names_in_
for col in model_features:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[model_features]


# Scale Features

x_scaled = scaler.transform(input_data)
x_scaled = pd.DataFrame(x_scaled, columns=input_data.columns)


# Prediction

if st.button("Predict Churn"):
    pred = model.predict(x_scaled)[0]
    proba = model.predict_proba(x_scaled)[0][1]

    if pred == 1:
        st.error(f"This customer is **likely to churn**! (Probability: {proba:.2f})")
    else:
        st.success(f"This customer is **likely to stay**. (Probability: {proba:.2f})")
