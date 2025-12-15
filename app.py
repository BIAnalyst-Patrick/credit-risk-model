import streamlit as st
import pandas as pd
import joblib
import gdown
import os

# -----------------------------------------------------
# Google Drive direct download links
# -----------------------------------------------------
PREPROCESSOR_URL = "https://drive.google.com/uc?id=1-Ft8KAsvU3jp3Ke97v3L7HL7B65Dk41k&export=download"
MODEL_URL        = "https://drive.google.com/uc?id=19Non_M6rh0s-vLJVkNzmtJZ9U236o9WG&export=download"

PREPROCESSOR_PATH = "preprocessor.pkl"
MODEL_PATH = "model.pkl"

# -----------------------------------------------------
# Download files if missing
# -----------------------------------------------------
def download_if_missing(url, path):
    if not os.path.exists(path):
        gdown.download(url, path, quiet=False)

# -----------------------------------------------------
# Load model + preprocessor
# -----------------------------------------------------
@st.cache_resource
def load_artifacts():
    try:
        download_if_missing(PREPROCESSOR_URL, PREPROCESSOR_PATH)
        download_if_missing(MODEL_URL, MODEL_PATH)

        pre = joblib.load(PREPROCESSOR_PATH)
        model = joblib.load(MODEL_PATH)

        return pre, model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None

pre, model = load_artifacts()

# -----------------------------------------------------
# Auto-computed feature generator (FIXED)
# -----------------------------------------------------
def compute_missing_features(df):
    """
    Fill 15 missing features needed by the model.
    Uses scalar extraction to avoid Series ambiguity errors.
    """

    # Extract scalars
    loan_amount = df["loan_amount"].iloc[0]
    loan_term = df["loan_term_months"].iloc[0]
    monthly_income = df["monthly_income"].iloc[0]
    score = df["credit_score"].iloc[0]

    # Static values
    df["loan_purpose"] = "personal"
    df["gender"] = "male"
    df["marital_status"] = "single"
    df["education_level"] = "bachelor"
    df["employment_status"] = "employed"

    # Derived fields
    annual_income = monthly_income * 12
    df["annual_income"] = annual_income
    df["other_income"] = 0
    df["num_of_open_accounts"] = 4
    df["num_of_past_defaults"] = 0
    df["months_with_bank"] = 24
    df["num_direct_debits"] = 3
    df["card_txns_per_month"] = 20

    # Derived calculations
    df["loan_to_income"] = loan_amount / max(annual_income, 1)
    df["installment_ratio"] = (loan_amount / max(loan_term, 1)) / max(monthly_income, 1)

    # Credit band logic
    if score < 580:
        band = "poor"
    elif score < 670:
        band = "fair"
    elif score < 740:
        band = "good"
    elif score < 800:
        band = "very_good"
    else:
        band = "excellent"

    df["credit_band"] = band

    return df


# -----------------------------------------------------
# Streamlit UI
# -----------------------------------------------------
st.title("📊 Loan Default Predictor — 7‑Input Smart Version")

st.write("Enter customer details below:")

loan_amount = st.number_input("Loan Amount", min_value=1000, max_value=200000, value=5000)
loan_term = st.number_input("Loan Term (months)", min_value=6, max_value=72, value=24)
interest_rate = st.number_input("Interest Rate (%)", min_value=1.0, max_value=40.0, value=12.0)
age = st.number_input("Age", min_value=18, max_value=75, value=30)
monthly_income = st.number_input("Monthly Income", min_value=500, max_value=50000, value=3000)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
avg_balance = st.number_input("Average Monthly Bank Balance", min_value=0, max_value=200000, value=20000)

# -----------------------------------------------------
# Predict
# -----------------------------------------------------
if st.button("Predict Default Risk"):
    if pre is None or model is None:
        st.error("❌ Model not loaded.")
    else:
        try:
            # Build DataFrame with 7 visible fields
            df = pd.DataFrame([{
                "loan_amount": loan_amount,
                "loan_term_months": loan_term,
                "interest_rate": interest_rate,
                "age": age,
                "monthly_income": monthly_income,
                "credit_score": credit_score,
                "avg_monthly_balance": avg_balance
            }])

            # Auto-fill 15 missing fields
            df = compute_missing_features(df)

            # Predict
            X_processed = pre.transform(df)
            pred_proba = model.predict_proba(X_processed)[0][1]
            pred_class = model.predict(X_processed)[0]

            st.success(f"📈 **Default Probability: {pred_proba:.2%}**")

            if pred_class == 1:
                st.error("⚠️ HIGH RISK: Likely to default")
            else:
                st.success("✅ LOW RISK: Unlikely to default")

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
