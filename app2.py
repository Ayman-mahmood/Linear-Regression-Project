import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="centered")

model = joblib.load("fraud_model.pkl")
encoder = joblib.load("label_encoders.pkl")["merchant_category"]

st.title("💳 Credit Card Fraud Detection")

amount = st.number_input("Transaction Amount", min_value=0.0, value=100.0)

transaction_hour = st.number_input("Transaction Hour", 0, 23, 12)

merchant_category = st.selectbox(
    "Merchant Category",
    ["Electronics", "Travel", "Grocery", "Food", "Clothing"]
)

foreign_transaction = st.selectbox(
    "Foreign Transaction",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

location_mismatch = st.selectbox(
    "Location Mismatch",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

device_trust_score = st.number_input(
    "Device Trust Score",
    min_value=0,
    value=80
)

velocity_last_24h = st.number_input(
    "Transactions in Last 24 Hours",
    min_value=0,
    value=5
)

cardholder_age = st.number_input(
    "Cardholder Age",
    min_value=18,
    max_value=100,
    value=30
)

if st.button("Predict Transaction", use_container_width=True):

    merchant_category = encoder.transform([merchant_category])[0]

    X = pd.DataFrame({
        "amount": [amount],
        "transaction_hour": [transaction_hour],
        "merchant_category": [merchant_category],
        "foreign_transaction": [foreign_transaction],
        "location_mismatch": [location_mismatch],
        "device_trust_score": [device_trust_score],
        "velocity_last_24h": [velocity_last_24h],
        "cardholder_age": [cardholder_age]
    })

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]
    confidence = probability[prediction] * 100

    if prediction == 1:
        st.error("❌ Fraudulent Transaction")
    else:
        st.success("✅ Legitimate Transaction")

    st.metric("Confidence", f"{confidence:.2f}%")

st.divider()

st.subheader("About")

st.write(
    "This application uses a Random Forest model trained on historical transaction data to predict whether a credit card transaction is fraudulent."
)

st.info("This project is for educational purposes only.")