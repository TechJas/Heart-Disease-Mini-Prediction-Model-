import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load model and scaler
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Heart Disease Prediction", page_icon="🫀", layout="wide")
st.title("🫀 Heart Disease Prediction")

st.markdown("---")
st.markdown("**Random Forest Model** | Test F1: 98.47%")

# Sidebar inputs
st.sidebar.header("📋 Patient Information")
age = st.sidebar.number_input("Age (years)", 1, 120, 55)
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
heart_rate = st.sidebar.number_input("Heart Rate (bpm)", 30, 200, 75)
sbp = st.sidebar.number_input("Systolic Blood Pressure (mmHg)", 60, 250, 120)
dbp = st.sidebar.number_input("Diastolic Blood Pressure (mmHg)", 30, 150, 80)
blood_sugar = st.sidebar.number_input("Blood Sugar (mg/dL)", 50, 500, 120)
ck_mb = st.sidebar.number_input("CK-MB (ng/mL)", 0.0, 500.0, 5.0, 0.1)
troponin = st.sidebar.number_input("Troponin (ng/mL)", 0.0, 15.0, 0.05, 0.01)

if st.sidebar.button("🔍 Predict"):
    gender_encoded = 1 if gender == "Male" else 0
    input_data = np.array([[age, gender_encoded, heart_rate, sbp, dbp, blood_sugar, ck_mb, troponin]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Result")
        if prediction == 1:
            st.error("⚠️ **Heart Disease Detected**")
        else:
            st.success("✅ **No Heart Disease**")

    with col2:
        st.subheader("Confidence")
        st.metric("Negative", f"{probability[0]*100:.2f}%")
        st.metric("Positive", f"{probability[1]*100:.2f}%")

    st.markdown("---")
    st.subheader("Input Summary")
    col_a, col_b = st.columns(2)
    with col_a:
        st.write(f"**Age:** {age}")
        st.write(f"**Gender:** {gender}")
        st.write(f"**Heart Rate:** {heart_rate} bpm")
        st.write(f"**Systolic BP:** {sbp} mmHg")
    with col_b:
        st.write(f"**Diastolic BP:** {dbp} mmHg")
        st.write(f"**Blood Sugar:** {blood_sugar} mg/dL")
        st.write(f"**CK-MB:** {ck_mb} ng/mL")
        st.write(f"**Troponin:** {troponin} ng/mL")
