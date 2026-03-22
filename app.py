import streamlit as st
import joblib
import numpy as np

# Modeli yüklə
rf = joblib.load("models/random_forest_model.pkl")
xgb = joblib.load("models/xgboost_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Başlıq
st.title("💳 Card Activity Churn Predictor")
st.caption("Predicting card disengagement using transaction behavioral patterns")

st.divider()

# Input-lar
st.subheader("Customer Profile")

card_age = st.slider("Card Age (months)", 1, 72, 24)
avg_txn = st.slider("Avg Monthly Transactions", 0.0, 40.0, 12.0)
avg_amt = st.slider("Avg Transaction Amount (AZN)", 1.0, 300.0, 60.0)
months_since = st.slider("Months Since Last Transaction", 0, 6, 1)
trend = st.slider("Transaction Trend", -1.0, 1.0, 0.0)
merchants = st.slider("Unique Merchant Count", 1, 25, 8)
intl = st.selectbox("International Usage", [0, 1])
complaints = st.slider("Complaint Count", 0, 10, 0)

st.divider()

# Model seçimi
model_choice = st.radio("Model", ["Random Forest", "XGBoost"])

# Proqnoz
if st.button("Predict"):
    input_data = np.array([[card_age, avg_txn, avg_amt, months_since,
                            trend, merchants, intl, complaints]])
    input_scaled = scaler.transform(input_data)

    model = rf if model_choice == "Random Forest" else xgb
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    st.divider()
    st.subheader("Result")

    if pred == 1:
        st.error(f"🔴 CHURN RISK — Probability: {prob:.1%}")
    else:
        st.success(f"🟢 ACTIVE — Churn Probability: {prob:.1%}")

    st.metric("Churn Probability", f"{prob:.1%}")