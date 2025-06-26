import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Page config
st.set_page_config(page_title="Financial Fraud Detector", page_icon="💰", layout="centered")

# Load the model
@st.cache_resource
def load_model():
    """Load the trained model from pickle file"""
    try:
        with open('fraudLR.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Model file not found. Please make sure 'fraudLR.pkl' exists.")
        return None

model = load_model()

# App title and description
st.title("💰 Financial Fraud Detection System")
st.markdown("Enter transaction details below to check if it might be fraudulent.")

# Transaction type mapping (from the notebook)
type_map = {
    "CASH_IN": 0,
    "CASH_OUT": 1,
    "DEBIT": 2,
    "PAYMENT": 3,
    "TRANSFER": 4,
}

# Create input form
col1, col2 = st.columns(2)

with col1:
    step = st.number_input("Step (time index)", min_value=0, value=1)
    trans_type = st.selectbox("Transaction Type", list(type_map.keys()))
    amount = st.number_input("Amount", min_value=0.0, value=100.0, format="%f")
    oldbalanceOrg = st.number_input("Old Balance - Origin", min_value=0.0, value=0.0, format="%f")

with col2:
    newbalanceOrg = st.number_input("New Balance - Origin", min_value=0.0, value=0.0, format="%f")
    oldbalanceDest = st.number_input("Old Balance - Destination", min_value=0.0, value=0.0, format="%f")
    newbalanceDest = st.number_input("New Balance - Destination", min_value=0.0, value=0.0, format="%f")
    isFlaggedFraud = st.selectbox("Is Flagged as Fraud", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# Predict button
if st.button("Predict"):
    if model:
        # Prepare input features
        features = np.array([[
            step,
            type_map[trans_type],
            amount,
            oldbalanceOrg,
            newbalanceOrg,
            oldbalanceDest,
            newbalanceDest,
            isFlaggedFraud
        ]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
        
        # Display result
        if prediction == 1:
            st.error(f"⚠️ Potential Fraud Detected! (Confidence: {probability:.2%})")
        else:
            st.success(f"✅ Transaction appears legitimate. (Confidence: {1-probability:.2%})")

# Information section
st.markdown("---")
with st.expander("ℹ️ About this app"):
    st.write("""
    - **Model**: Logistic Regression trained on synthetic financial transaction data
    - **Features**: The model analyzes transaction step, type, amount, and account balances
    - **Data Source**: Synthetic Financial Datasets for Fraud Detection
    - **Confidence**: Probability score from the model's prediction
    """)