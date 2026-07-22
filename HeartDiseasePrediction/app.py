import os

import joblib
import pandas as pd
import streamlit as st


# Page settings
st.set_page_config(
    page_title="Heart Disease Prediction",
    layout="centered"
)


# Model path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "heart_disease_model.pkl")


# Load trained model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as error:
    st.error(f"Model could not be loaded: {error}")
    st.stop()


# Website title
st.title("Heart Disease Prediction System")

st.write(
    "Please enter the patient's clinical information to generate a prediction."
)


# Patient input form
with st.form("prediction_form"):

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=50,
        step=1
    )

    sex = st.selectbox(
        "Sex",
        options=[0, 1],
        format_func=lambda value: "Female" if value == 0 else "Male"
    )

    cp = st.selectbox(
        "Chest Pain Type",
        options=[0, 1, 2, 3]
    )

    trestbps = st.number_input(
        "Resting Blood Pressure",
        min_value=50,
        max_value=250,
        value=120,
        step=1
    )

    chol = st.number_input(
        "Cholesterol",
        min_value=50,
        max_value=700,
        value=200,
        step=1
    )

    fbs = st.selectbox(
        "Fasting Blood Sugar Above 120 mg/dL",
        options=[0, 1],
        format_func=lambda value: "No" if value == 0 else "Yes"
    )

    restecg = st.selectbox(
        "Resting ECG Result",
        options=[0, 1, 2]
    )

    thalach = st.number_input(
        "Maximum Heart Rate Achieved",
        min_value=50,
        max_value=250,
        value=150,
        step=1
    )

    exang = st.selectbox(
        "Exercise-Induced Angina",
        options=[0, 1],
        format_func=lambda value: "No" if value == 0 else "Yes"
    )

    oldpeak = st.number_input(
        "Oldpeak",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1
    )

    slope = st.selectbox(
        "Slope",
        options=[0, 1, 2]
    )

    ca = st.selectbox(
        "Number of Major Vessels",
        options=[0, 1, 2, 3, 4]
    )

    thal = st.selectbox(
        "Thalassemia Result",
        options=[0, 1, 2, 3]
    )

    submitted = st.form_submit_button(
        "Predict",
        type="primary",
        use_container_width=True
    )


# Prediction
if submitted:

    patient_data = pd.DataFrame(
        {
            "age": [age],
            "sex": [sex],
            "cp": [cp],
            "trestbps": [trestbps],
            "chol": [chol],
            "fbs": [fbs],
            "restecg": [restecg],
            "thalach": [thalach],
            "exang": [exang],
            "oldpeak": [oldpeak],
            "slope": [slope],
            "ca": [ca],
            "thal": [thal]
        }
    )

    try:
        prediction = int(model.predict(patient_data)[0])

        confidence = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(patient_data)[0]
            confidence = float(probabilities[prediction] * 100)

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("High Predicted Risk of Heart Disease")
        else:
            st.success("Low Predicted Risk of Heart Disease")

        if confidence is not None:
            st.write(f"**Model Confidence: {confidence:.2f}%**")

    except Exception as error:
        st.error(f"Prediction could not be completed: {error}")


st.markdown("---")

st.caption(
    "This application is intended for educational purposes only."
)
