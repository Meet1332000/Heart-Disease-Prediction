import os
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st


# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)


# =========================================================
# LOAD TRAINED MODEL
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "heart_disease_model.pkl")


@st.cache_resource
def load_model():
    """Load the trained machine-learning model."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "The file 'heart_disease_model.pkl' was not found. "
            "Make sure it is uploaded to the same GitHub folder as app.py."
        )

    return joblib.load(MODEL_PATH)


try:
    model = load_model()

except Exception as error:
    st.error(f"Unable to load the trained model: {error}")
    st.stop()


# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown(
    """
    <style>
        .main-title {
            text-align: center;
            font-size: 45px;
            font-weight: 800;
            margin-bottom: 5px;
        }

        .subtitle {
            text-align: center;
            font-size: 18px;
            color: #777777;
            margin-bottom: 25px;
        }

        .result-card {
            padding: 25px;
            border-radius: 12px;
            margin-top: 20px;
            margin-bottom: 20px;
        }

        .higher-risk {
            background-color: #fff1f1;
            border-left: 8px solid #d62728;
        }

        .lower-risk {
            background-color: #eefaf1;
            border-left: 8px solid #2ca02c;
        }

        .footer {
            text-align: center;
            color: #777777;
            font-size: 14px;
            margin-top: 30px;
            margin-bottom: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.title("📌 Project Information")

    st.markdown("### Developer")
    st.write("**Manmeet Singh**")

    st.markdown("### University")
    st.write("Ulster University")

    st.markdown("### Course")
    st.write("MSc Computer Science and Technology")

    st.markdown("### Project")
    st.write("Heart Disease Prediction Using Machine Learning")

    st.markdown("### Supervisor")
    st.write("Dr. Taha Hussein Rassem")

    st.markdown("### Deployed Model")
    st.write("Random Forest Classifier")

    st.markdown("### Recorded Accuracy")
    st.success("98.54%")

    st.markdown("---")

    st.info(
        "This application was developed for academic and research "
        "purposes only."
    )


# =========================================================
# PAGE HEADER
# =========================================================
st.markdown(
    '<div class="main-title">❤️ Heart Disease Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        MSc Dissertation Project | Ulster University |
        Developed by Manmeet Singh
    </div>
    """,
    unsafe_allow_html=True
)

st.warning(
    "This system is an academic machine-learning demonstration. "
    "It must not be treated as a medical diagnosis or a replacement "
    "for professional healthcare advice."
)


# =========================================================
# PATIENT INPUT FORM
# =========================================================
with st.form("heart_disease_form"):
    st.subheader("Enter Patient Clinical Information")

    left_column, right_column = st.columns(2)

    with left_column:
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
            options=[0, 1, 2, 3],
            format_func=lambda value: {
                0: "0 — Typical angina",
                1: "1 — Atypical angina",
                2: "2 — Non-anginal pain",
                3: "3 — Asymptomatic"
            }[value]
        )

        trestbps = st.number_input(
            "Resting Blood Pressure (mm Hg)",
            min_value=50,
            max_value=250,
            value=120,
            step=1
        )

        chol = st.number_input(
            "Serum Cholesterol (mg/dL)",
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
            options=[0, 1, 2],
            format_func=lambda value: {
                0: "0 — Normal",
                1: "1 — ST-T wave abnormality",
                2: "2 — Left ventricular hypertrophy"
            }[value]
        )

    with right_column:
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
            "Oldpeak: ST Depression Induced by Exercise",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1
        )

        slope = st.selectbox(
            "Slope of Peak Exercise ST Segment",
            options=[0, 1, 2],
            format_func=lambda value: {
                0: "0 — Upsloping",
                1: "1 — Flat",
                2: "2 — Downsloping"
            }[value]
        )

        ca = st.selectbox(
            "Number of Major Vessels",
            options=[0, 1, 2, 3, 4]
        )

        thal = st.selectbox(
            "Thalassemia Result",
            options=[0, 1, 2, 3],
            format_func=lambda value: {
                0: "0 — Unknown",
                1: "1 — Normal",
                2: "2 — Fixed defect",
                3: "3 — Reversible defect"
            }[value]
        )

    submit_button = st.form_submit_button(
        "🔍 Run Heart Disease Prediction",
        type="primary",
        use_container_width=True
    )


# =========================================================
# PREDICTION
# =========================================================
if submit_button:

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
            probability_values = model.predict_proba(patient_data)[0]

            if prediction == 1:
                confidence = float(probability_values[1] * 100)
            else:
                confidence = float(probability_values[0] * 100)

        st.markdown("---")
        st.subheader("Prediction Result")

        if prediction == 1:
            result_text = "Higher Predicted Risk"
            result_description = (
                "The machine-learning model predicted the presence "
                "of heart disease."
            )
            card_class = "higher-risk"

        else:
            result_text = "Lower Predicted Risk"
            result_description = (
                "The machine-learning model did not predict the "
                "presence of heart disease."
            )
            card_class = "lower-risk"

        confidence_html = ""

        if confidence is not None:
            confidence_html = (
                f"<h3>Model confidence: {confidence:.2f}%</h3>"
            )

        st.markdown(
            f"""
            <div class="result-card {card_class}">
                <h2>{result_text}</h2>
                <p>{result_description}</p>
                {confidence_html}
            </div>
            """,
            unsafe_allow_html=True
        )

        if confidence is not None:
            st.progress(
                confidence / 100,
                text=f"Prediction confidence: {confidence:.2f}%"
            )

        st.caption(
            "A high confidence score does not guarantee that the "
            "prediction is medically correct."
        )

        st.subheader("Submitted Patient Information")

        display_data = pd.DataFrame(
            {
                "Clinical Attribute": [
                    "Age",
                    "Sex",
                    "Chest pain type",
                    "Resting blood pressure",
                    "Cholesterol",
                    "Fasting blood sugar above 120 mg/dL",
                    "Resting ECG",
                    "Maximum heart rate",
                    "Exercise-induced angina",
                    "Oldpeak",
                    "Slope",
                    "Major vessels",
                    "Thalassemia"
                ],
                "Entered Value": [
                    age,
                    "Male" if sex == 1 else "Female",
                    cp,
                    f"{trestbps} mm Hg",
                    f"{chol} mg/dL",
                    "Yes" if fbs == 1 else "No",
                    restecg,
                    thalach,
                    "Yes" if exang == 1 else "No",
                    oldpeak,
                    slope,
                    ca,
                    thal
                ]
            }
        )

        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True
        )

        generated_time = datetime.now().strftime(
            "%d %B %Y at %H:%M"
        )

        confidence_report = (
            f"{confidence:.2f}%"
            if confidence is not None
            else "Not available"
        )

        prediction_report = f"""
HEART DISEASE PREDICTION REPORT

Generated: {generated_time}

PROJECT INFORMATION
Developer: Manmeet Singh
University: Ulster University
Course: MSc Computer Science and Technology
Supervisor: Dr. Taha Hussein Rassem
Project: Heart Disease Prediction Using Machine Learning
Model: Random Forest Classifier

PREDICTION RESULT
Result: {result_text}
Model Confidence: {confidence_report}

PATIENT INFORMATION
Age: {age}
Sex: {"Male" if sex == 1 else "Female"}
Chest Pain Type: {cp}
Resting Blood Pressure: {trestbps} mm Hg
Cholesterol: {chol} mg/dL
Fasting Blood Sugar Above 120 mg/dL: {"Yes" if fbs == 1 else "No"}
Resting ECG: {restecg}
Maximum Heart Rate: {thalach}
Exercise-Induced Angina: {"Yes" if exang == 1 else "No"}
Oldpeak: {oldpeak}
Slope: {slope}
Number of Major Vessels: {ca}
Thalassemia Result: {thal}

DISCLAIMER
This prediction was produced by an academic machine-learning model.
It is not a medical diagnosis and must not replace advice from a
qualified healthcare professional.
"""

        st.download_button(
            label="📄 Download Prediction Report",
            data=prediction_report,
            file_name="heart_disease_prediction_report.txt",
            mime="text/plain",
            use_container_width=True
        )

    except Exception as error:
        st.error(f"Prediction could not be completed: {error}")

        st.info(
            "Check that the model was trained using these exact columns: "
            "age, sex, cp, trestbps, chol, fbs, restecg, thalach, "
            "exang, oldpeak, slope, ca and thal."
        )


# =========================================================
# ABOUT THE PROJECT
# =========================================================
st.markdown("---")
st.header("👨‍💻 About This Project")

st.write(
    """
    This web application was developed as part of an MSc dissertation
    project. It uses a trained machine-learning model to estimate the
    presence or absence of heart disease from clinical patient
    attributes.

    The project compared Logistic Regression, Random Forest and
    XGBoost. The deployed application uses the saved Random Forest
    model.
    """
)

project_column, technical_column = st.columns(2)

with project_column:
    st.markdown("### Academic Information")
    st.write("**Developer:** Manmeet Singh")
    st.write("**University:** Ulster University")
    st.write("**Course:** MSc Computer Science and Technology")
    st.write("**Supervisor:** Dr. Taha Hussein Rassem")
    st.write("**Academic Year:** 2026")

with technical_column:
    st.markdown("### Technical Information")
    st.write("**Programming Language:** Python")
    st.write("**Web Framework:** Streamlit")
    st.write("**Data Processing:** Pandas")
    st.write("**Model Storage:** Joblib")
    st.write("**Deployed Model:** Random Forest Classifier")
    st.write("**Recorded Test Accuracy:** 98.54%")

st.error(
    "Medical Disclaimer: This application is intended only for "
    "education and research. Its output must not be used to make "
    "medical decisions. Anyone concerned about their health should "
    "consult a qualified healthcare professional."
)

st.markdown(
    """
    <div class="footer">
        © 2026 Manmeet Singh | Ulster University<br>
        Heart Disease Prediction Using Machine Learning
    </div>
    """,
    unsafe_allow_html=True
)
