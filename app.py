import streamlit as st
import pandas as pd
import requests
import joblib
from io import BytesIO

st.set_page_config(page_title="Loan Default Predictor", layout="centered")

# -------------------------
# GOOGLE DRIVE DIRECT DOWNLOAD (bypass confirmation)
# -------------------------
def download_from_gdrive(file_id):
    """
    Downloads large files from Google Drive by bypassing the confirmation token.
    Returns raw bytes.
    """
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                return value
        return None

    URL = "https://drive.usercontent.google.com/download"

    session = requests.Session()
    response = session.get(URL, params={"id": file_id}, stream=True)

    token = get_confirm_token(response)
    if token:
        params = {"id": file_id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    file_data = b""
    for chunk in response.iter_content(32768):
        if chunk:
            file_data += chunk

    return file_data


# -------------------------
# LOAD MODELS (cached)
# -------------------------
@st.cache_resource(show_spinner=True)
def load_models():
    try:
        st.write("⏳ Downloading model files...")

        # Your Google Drive file IDs **(from your earlier message)**:
        PREPROCESSOR_ID = "1-Ft8KAsvU3jp3Ke97v3L7HL7B65Dk41k"
        MODEL_ID = "19Non_M6rh0s-vLJVkNzmtJZ9U236o9WG"

        # Download files
        preprocessor_bytes = download_from_gdrive(PREPROCESSOR_ID)
        model_bytes = download_from_gdrive(MODEL_ID)

        # Load into memory
        preprocessor = joblib.load(BytesIO(preprocessor_bytes))
        model = joblib.load(BytesIO(model_bytes))

        return preprocessor, model

    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None


# Load models
preprocessor, model = load_models()

if preprocessor is None or model is None:
    st.stop()  # stop app safely


# -------------------------
# APP INTERFACE
# -------------------------
st.title("📊 Loan Default Prediction App")
st.write("Enter the client details below to estimate default risk.")


# -------------------------
# USER INPUT FIELDS
# -------------------------
credit_score = st.number_input("Credit Score", 300, 900, 650)
age = st.number_input("Age", 18, 100, 30)
monthly_income = st.number_input("Monthly Income (KES)", 0, 2000000, 50000)
loan_amount = st.number_input("Loan Amount (KES)", 0, 5000000, 100000)
loan_term = st.number_input("Loan Term (Months)", 1, 84, 12)
interest_rate = st.number_input("Interest Rate (%)", 1.0, 25.0, 12.0)

gender = st.selectbox("Gender", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", ["Single", "Married"])
education_level = st.selectbox("Education Level", ["High School", "Diploma", "Bachelors"])
loan_purpose = st.selectbox("Loan Purpose", ["Personal", "Business"])

# Convert to DataFrame
input_df = pd.DataFrame({
    "credit_score": [credit_score],
    "age": [age],
    "monthly_income": [monthly_income],
    "loan_amount": [loan_amount],
    "loan_term_months": [loan_term],
    "interest_rate": [interest_rate],
    "gender": [gender],
    "marital_status": [marital_status],
    "education_level": [education_level],
    "loan_purpose": [loan_purpose],
})


# -------------------------
# PREDICT BUTTON
# -------------------------
if st.button("Predict Default Risk"):
    try:
        # Preprocess → Predict
        transformed = preprocessor.transform(input_df)
        probability = model.predict_proba(transformed)[0][1]

        st.subheader("🔍 Prediction Result")
        st.write(f"**Estimated Probability of Default:** `{probability:.2%}`")

        if probability >= 0.40:
            st.error("⚠️ High Risk — Recommend further review or tighter lending terms.")
        elif probability >= 0.20:
            st.warning("🟠 Medium Risk — Consider moderate risk controls.")
        else:
            st.success("🟢 Low Risk — Eligible for standard approval.")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
