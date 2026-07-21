import joblib
import pandas as pd

# Load the trained model
model = joblib.load("heart_disease_model.pkl")

print("\nHeart Disease Prediction System")
print("--------------------------------")

try:
    age = float(input("Age: "))
    sex = int(input("Sex (1 = Male, 0 = Female): "))
    cp = int(input("Chest pain type (0 to 3): "))
    trestbps = float(input("Resting blood pressure: "))
    chol = float(input("Cholesterol level: "))
    fbs = int(input("Fasting blood sugar (1 = Yes, 0 = No): "))
    restecg = int(input("Resting ECG result (0 to 2): "))
    thalach = float(input("Maximum heart rate achieved: "))
    exang = int(input("Exercise-induced angina (1 = Yes, 0 = No): "))
    oldpeak = float(input("Oldpeak value: "))
    slope = int(input("Slope value (0 to 2): "))
    ca = int(input("Number of major vessels (0 to 4): "))
    thal = int(input("Thal value (0 to 3): "))

    patient_data = pd.DataFrame(
        [[
            age,
            sex,
            cp,
            trestbps,
            chol,
            fbs,
            restecg,
            thalach,
            exang,
            oldpeak,
            slope,
            ca,
            thal
        ]],
        columns=[
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal"
        ]
    )

    prediction = model.predict(patient_data)[0]
    probability = model.predict_proba(patient_data)[0]

    if prediction == 1:
        confidence = probability[1] * 100
        print("\nPrediction: Heart disease detected")
        print(f"Confidence: {confidence:.2f}%")
    else:
        confidence = probability[0] * 100
        print("\nPrediction: No heart disease detected")
        print(f"Confidence: {confidence:.2f}%")

except ValueError:
    print("\nError: Please enter numbers only.")

except FileNotFoundError:
    print("\nError: heart_disease_model.pkl was not found.")

except Exception as error:
    print("\nAn error occurred:", error)