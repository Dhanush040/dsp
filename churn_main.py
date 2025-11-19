
import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import os

MODEL_FILE = 'logistic_regression_model.joblib'
ENC_FILE = 'encoders_gender_payment.joblib'
FEAT_FILE = 'selected_features.joblib'

for f in (MODEL_FILE, ENC_FILE, FEAT_FILE):
    if not os.path.exists(f):
        st.error(f"Required file not found: {f}. Run training script first.")
        st.stop()

model = load(MODEL_FILE)
encoders = load(ENC_FILE)            # dict: {col_name: {category: code, ...}, ...}
selected_features = load(FEAT_FILE)  # list of feature names in exact order

# App UI
st.title("Customer Churn Prediction App (Logistic Regression)")
st.header("Enter Customer Information")

# Provide the same set of inputs (we'll build mapping from encoders)
# Numeric inputs
tenure = st.number_input("Tenure (in months)", min_value=0, max_value=240, value=12)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=10000.0, value=79.85, format="%.2f")
total_charges = st.number_input("Total Charges", min_value=0.0, max_value=1_000_000.0, value=round(tenure * monthly_charges, 2), format="%.2f")

# For categorical inputs, derive choices from saved encoders so app matches training
def options_for(col):
    if col in encoders:
        return list(encoders[col].keys())
    return None

internet_service = st.selectbox("Internet Service", options_for('InternetService') or ['DSL', 'Fiber optic', 'No'])
contract = st.selectbox("Contract", options_for('Contract') or ['Month-to-month', 'One year', 'Two year'])
gender = st.selectbox("Gender", options_for('gender') or ['Male', 'Female'])
payment_method = st.selectbox("Payment Method", options_for('PaymentMethod') or [
    'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
])

# Build numeric input vector in the same order as selected_features
# Convert categorical inputs using saved encoders
input_map = {
    'tenure': float(tenure),
    'MonthlyCharges': float(monthly_charges),
    'TotalCharges': float(total_charges),
    'InternetService': encoders['InternetService'][internet_service] if 'InternetService' in encoders else 0,
    'Contract': encoders['Contract'][contract] if 'Contract' in encoders else 0,
    'gender': encoders['gender'][gender] if 'gender' in encoders else 0,
    'PaymentMethod': encoders['PaymentMethod'][payment_method] if 'PaymentMethod' in encoders else 0
}

# Create feature vector preserving order of selected_features
try:
    feature_vector = [ float(input_map[f]) for f in selected_features ]
except KeyError as e:
    st.error(f"Feature {e} missing from input_map. Check selected_features saved during training.")
    st.stop()

arr = np.array([feature_vector], dtype=float)

# Optional safety: check model expects same number of features
n_expected = getattr(model, "n_features_in_", None)
if n_expected is not None and n_expected != arr.shape[1]:
    st.error(f"Model expects {n_expected} features but input has {arr.shape[1]}.")
    st.stop()

# Make prediction
prediction = model.predict(arr)[0]
proba = None
if hasattr(model, "predict_proba"):
    try:
        proba = model.predict_proba(arr)[0,1]
    except Exception:
        proba = None

# Display result
st.header("Prediction Result")
if int(prediction) == 0:
    st.success(
        f"This customer is likely to stay.\n\n"
        f"Gender: {gender}\n"
        f"Payment Method: {payment_method}"
    )
else:
    st.error(
        f"This customer is likely to churn.\n\n"
        f"Gender: {gender}\n"
        f"Payment Method: {payment_method}"
    )

if proba is not None:
    st.info(f"Churn probability: {proba*100:.2f}%")

# Show the numeric vector sent to model (for debugging)
st.subheader("Numeric feature vector sent to model (in order)")
st.write(dict(zip(selected_features, feature_vector)))
