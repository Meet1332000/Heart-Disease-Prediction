import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "heart_disease_model.pkl")

model = joblib.load(MODEL_PATH)

st.title("❤️ Heart Disease Prediction System")
st.write("Enter the patient details below.")

age = st.number_input("Age", 18, 100, 50)
sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.selectbox("Chest pain type", [0, 1, 2, 3])
trestbps = st.number_input("Resting blood pressure", 50, 250, 120)
chol = st.number_input("Cholesterol", 50, 700, 200)
fbs = st.selectbox(
    "Fasting blood sugar",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)
restecg = st.selectbox("Resting ECG result", [0, 1, 2])
thalach = st.number_input("Maximum heart rate achieved", 50, 250, 150)
exang = st.selectbox(
    "Exercise-induced angina",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)
oldpeak = st.number_input("Oldpeak value", 0.0, 10.0, 1.0, step=0.1)
slope = st.selectbox("Slope", [0, 1, 2])
ca = st.selectbox("Number of major vessels", [0, 1, 2, 3, 4])
thal = st.selectbox("Thal value", [0, 1, 2, 3])

if st.button("Predict"):
    patient_data = pd.DataFrame(
        [[
            age, sex, cp, trestbps, chol, fbs, restecg,
            thalach, exang, oldpeak, slope, ca, thal
        ]],
        columns=[
            "age", "sex", "cp", "trestbps", "chol", "fbs",
            "restecg", "thalach", "exang", "oldpeak",
            "slope", "ca", "thal"
        ]
    )

    prediction = model.predict(patient_data)[0]
    probabilities = model.predict_proba(patient_data)[0]

    if prediction == 1:
        st.error("Heart disease detected")
        st.write(f"Confidence: {probabilities[1] * 100:.2f}%")
    else:
        st.success("No heart disease detected")
        st.write(f"Confidence: {probabilities[0] * 100:.2f}%")

st.warning("Academic prediction tool only — not a medical diagnosis.")
