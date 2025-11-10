import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Customer Churn Predictor",
    #page_icon="💡",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ==================== LOAD MODEL & SCALER ====================
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
        st.error(f"❌ Model or scaler file not found.\n\nDetails: {e}")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        st.stop()

model, scaler = load_model()

# ==================== HEADER ====================
st.markdown("""
# Customer Churn Predictor
Easily predict whether a telecom customer is **likely to churn or stay** based on their details.
---
""")

# ==================== SIDEBAR INFO ====================
with st.sidebar:
    st.header("About the App")
    st.write("""
    This app uses a trained **Random Forest model**
    to predict customer churn.
    
    Enter details on the left and click **Predict Churn**.
    """)
    st.markdown("---")
    st.caption("Made by [Nomdorah Marcus](https://github.com/SharonE-coder)")
    st.caption("Model: Random Forest | Framework: Streamlit")

# ==================== INPUT FORM ====================
st.subheader("Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

with col2:
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=75.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=500.0)
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )

# ==================== PREPARE INPUT DATA ====================
try:
    input_data = pd.DataFrame(
        {
            "tenure": [float(tenure)],
            "MonthlyCharges": [float(monthly_charges)],
            "TotalCharges": [float(total_charges)],
            "gender_Male": [1 if gender == "Male" else 0],
            "SeniorCitizen": [1 if senior == "Yes" else 0],
            "Contract_One year": [1 if contract == "One year" else 0],
            "Contract_Two year": [1 if contract == "Two year" else 0],
            "InternetService_Fiber optic": [1 if internet_service == "Fiber optic" else 0],
            "InternetService_No": [1 if internet_service == "No" else 0],
            "PaymentMethod_Credit card (automatic)": [1 if payment_method == "Credit card (automatic)" else 0],
            "PaymentMethod_Electronic check": [1 if payment_method == "Electronic check" else 0],
            "PaymentMethod_Mailed check": [1 if payment_method == "Mailed check" else 0],
        }
    )

    # Align input with model features
    model_features = model.feature_names_in_
    for col in model_features:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[model_features]

    # Scale features
    x_scaled = scaler.transform(input_data)
except Exception as e:
    st.error(f"Error preparing input data: {e}")
    st.stop()

# ==================== PREDICTION ====================
if st.button("Predict Churn"):
    try:
        pred = model.predict(x_scaled)[0]
        proba = model.predict_proba(x_scaled)[0][1]

        st.markdown("---")
        st.subheader("Prediction Result")

        if pred == 1:
            st.error(f"This customer is **likely to churn**.\n\n**Probability:** {proba:.2f}")
        else:
            st.success(f"This customer is **likely to stay**.\n\n**Probability:** {proba:.2f}")

        st.markdown("---")
        st.caption("Model: Random Forest | Scaled with StandardScaler")

    except Exception as e:
        st.error(f"Prediction error: {e}")
