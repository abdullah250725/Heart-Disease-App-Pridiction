import streamlit as st
import pickle
import numpy as np
import pandas as pd

# --- 1. Page Config  ---
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")

# --- 2. Load Model and Scaler ---
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open('Models/rf_classifier.pkl', 'rb'))
        scaler = pickle.load(open('Models/scaler.pkl', 'rb'))
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        return None, None

model, scaler = load_model()

# --- 3. Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Predictor", "About"])

# --- 4. Page: Home ---
if page == "Home":
    st.title("❤️ Heart Disease Risk Prediction")
    st.write("Use the sidebar to navigate to the predictor or learn more.")

# --- 5. Page: Predictor ---
elif page == "Predictor":
    st.title("🧠 Heart Disease Predictor")

    if not model or not scaler:
        st.warning("Model not loaded. Please check files.")
    else:
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            with col1:
                gender = st.radio("Gender", ["Male", "Female"])
                age = st.slider("Age", 20, 100, 50)
                smoker = st.radio("Current Smoker", ["No", "Yes"])
                cigs = st.slider("Cigarettes/Day", 0, 50, 0)
                meds = st.radio("BP Medications", ["No", "Yes"])
            with col2:
                stroke = st.radio("Prior Stroke", ["No", "Yes"])
                hyp = st.radio("Hypertension", ["No", "Yes"])
                diabetes = st.radio("Diabetes", ["No", "Yes"])
                chol = st.slider("Total Cholesterol", 100, 600, 200)
                sysBP = st.slider("Systolic BP", 80, 300, 120)
                diaBP = st.slider("Diastolic BP", 40, 200, 80)
                bmi = st.slider("BMI", 15.0, 50.0, 25.0)
                hr = st.slider("Heart Rate", 40, 150, 72)
                glucose = st.slider("Glucose", 40, 400, 100)

            submitted = st.form_submit_button("Predict Risk")

        if submitted:
            try:
                input_data = np.array([
                    1 if gender == "Male" else 0,
                    age,
                    1 if smoker == "Yes" else 0,
                    cigs,
                    1 if meds == "Yes" else 0,
                    1 if stroke == "Yes" else 0,
                    1 if hyp == "Yes" else 0,
                    1 if diabetes == "Yes" else 0,
                    chol, sysBP, diaBP, bmi, hr, glucose
                ]).reshape(1, -1)

                scaled = scaler.transform(input_data)
                pred = model.predict(scaled)[0]
                prob = model.predict_proba(scaled)[0][1]

                st.success(f"Risk Probability: {prob:.2%}")
                if pred == 1:
                    st.error("⚠️ High Risk! Please consult a cardiologist.")
                else:
                    st.info("✅ Low Risk. Keep maintaining your health.")

                # Feature Importance
                if hasattr(model, "feature_importances_"):
                    st.subheader("Feature Importances")
                    features = ["Gender", "Age", "Smoking", "Cigs/Day", "BP Meds",
                                "Stroke", "Hypertension", "Diabetes", "Cholesterol",
                                "Systolic BP", "Diastolic BP", "BMI", "Heart Rate", "Glucose"]
                    st.bar_chart(pd.DataFrame({
                        "Features": features,
                        "Importance": model.feature_importances_
                    }).set_index("Features"))

            except Exception as e:
                st.error(f"Prediction failed: {e}")

# --- 6. Page: About ---
elif page == "About":
    st.title("📘 About This App")
    st.write("This app predicts heart disease risk based on health metrics.")
    st.write("Built using Streamlit and a Random Forest model.")
