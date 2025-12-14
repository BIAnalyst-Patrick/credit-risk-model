import streamlit as st
import pandas as pd
import joblib
import gdown
import os

st.title("📊 Loan Default Prediction System")

PREPROCESSOR_FILE = "preprocessor.pkl"
MODEL_FILE = "model.pkl"

PREPROCESSOR_ID = "1-Ft8KAsvU3jp3Ke97v3L7HL7B65Dk41k"
MODEL_ID = "19Non_M6rh0s-vLJVkNzmtJZ9U236o9WG"

# -----------------------------
# Function to download files
# -----------------------------
def download_from_drive(file_id, output):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output, quiet=False)

# -----------------------------
# Load Preprocessor + Model
# -----------------------------
@st.cache_resource
def load_artifacts():
    try:
        # Download if missing
        if not os.path.exists(PREPROCESSOR_FILE):
            st.warning("Downloading preprocessor...")
            download_from_drive(PREPROCESSOR_ID, PREPROCESSOR_FILE)

        if not os.path.exists(MODEL_FILE):
            st.warning("Downloading model...")
            download_from_drive(MODEL_ID, MODEL_FILE)

        st.info("Loading... please wait")

        preprocessor = joblib.load(PREPROCESSOR_FILE)
        model = joblib.load(MODEL_FILE)

        return preprocessor, model

    except Exception as e:
        st.error(f"❌ Error loading model components: {e}")
        return None, None


preprocessor, model = load_artifacts()

if preprocessor is None or model is None:
    st.stop()


# -----------------------------
# User Input Form
# -----------------------------
st.header("Enter Loan Applicant Details")

col1, col2 = st.columns(2)

with col1:
    loan_amount = st.number_input("Loan Amount", min_value=500, max_value=5000000, value=50000)
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    income = st.number_input("Monthly Income", min_value=0, max_value=500000, value=30000)

with col2:
    employment_type = st.selectbox("Employment Type", ["Salaried", "Self-employed"])
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
    dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=1)


# Dictionary to DataFrame
input_data = pd.DataFrame({
    "loan_amount": [loan_amount],
    "age": [age],
    "income": [income],
    "employment_type": [employment_type],
    "credit_score": [credit_score],
    "dependents": [dependents]
})


# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Predict Loan Default Risk"):
    try:
        processed = preprocessor.transform(input_data)
        prediction = model.predict(processed)[0]
        probability = model.predict_proba(processed)[0][1]

        st.subheader("🔍 Prediction Result")

        if prediction == 1:
            st.error(f"❌ High Risk of Default (Probability: {probability:.2f})")
        else:
            st.success(f"✅ Low Risk of Default (Probability: {probability:.2f})")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
