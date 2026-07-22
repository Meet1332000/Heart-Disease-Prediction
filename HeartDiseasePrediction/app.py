import os

import joblib
import pandas as pd
import streamlit as st


# Page settings
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️"
)


# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "heart_disease_model.pkl")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as error:
    st.error(f"Model could not be loaded: {error}")
    st.stop()


# Website title
st.title("❤️ Heart Disease Prediction System")

st.write(
    "Enter the patient’s clinical information below to generate "
    "a machine-learning prediction."
)

st.info(
    "This application is developed for academic purposes only."
)


# Patient input form
with st.form("prediction_form"):

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=50
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
        value=120
    )

    chol = st.number_input(
        "Cholesterol",
        min_value=50,
        max_value=700,
        value=200
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
        value=150
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

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(patient_data)[0]
            confidence = probabilities[prediction] * 100
        else:
            confidence = None

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("Higher predicted risk of heart disease")
        else:
            st.success("Lower predicted risk of heart disease")

        if confidence is not None:
            st.write(f"Model confidence: **{confidence:.2f}%**")
            st.progress(confidence / 100)

    except Exception as error:
        st.error(f"Prediction could not be completed: {error}")


st.markdown("---")

st.caption(
    "Developed by Manmeet Singh | Supervisor: Dr. Taha Hussein Rassem"
)

st.caption(
    "Disclaimer: This system is not a medical diagnosis. "
    "Consult a qualified healthcare professional for medical advice."
)
