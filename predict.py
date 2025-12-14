import joblib
import pandas as pd
import numpy as np

# -------------------------------------------
# 1. Load model + preprocessing components
# -------------------------------------------
MODEL_PATH = "loan_default_model_v1.pkl"

print("Loading model...")
model_package = joblib.load(MODEL_PATH)

preprocessor = model_package["preprocessor"]
model = model_package["model"]
feature_names = model_package["feature_names"]

print("Model loaded successfully.\n")

# -------------------------------------------
# 2. Define a function to collect inputs
# -------------------------------------------
def collect_user_input():
    print("Please enter loan applicant details:\n")

    data = {}

    # Numeric inputs
    data["loan_amount"] = float(input("Loan amount: "))
    data["loan_term_months"] = int(input("Loan term (months): "))
    data["interest_rate"] = float(input("Interest rate (%): "))
    data["age"] = int(input("Age: "))
    data["annual_income"] = float(input("Annual income: "))
    data["monthly_income"] = float(input("Monthly income: "))
    data["other_income"] = float(input("Other income: "))
    data["credit_score"] = int(input("Credit score (300–850): "))
    data["num_of_open_accounts"] = int(input("Number of open accounts: "))
    data["num_of_past_defaults"] = int(input("Past defaults: "))
    data["avg_monthly_balance"] = float(input("Average monthly balance: "))
    data["months_with_bank"] = int(input("Months with the bank: "))
    data["num_direct_debits"] = int(input("Number of direct debits: "))
    data["num_card_txns_6m"] = int(input("Card transactions in 6 months: "))
    data["payment_delay_days"] = int(input("Payment delay days: "))

    # Categorical inputs
    data["loan_purpose"] = input("Loan purpose (e.g., Personal, Business, Mortgage): ")
    data["gender"] = input("Gender (Male/Female): ")
    data["marital_status"] = input("Marital status (Single/Married/Divorced): ")
    data["education_level"] = input("Education (High School, Diploma, Bachelors, Masters): ")
    data["employment_status"] = input("Employment status (Employed/Self-Employed/Unemployed): ")

    return pd.DataFrame([data])


# -------------------------------------------
# 3. Predict PD (Probability of Default)
# -------------------------------------------
def make_prediction(input_df):
    # Ensure columns match training order
    input_df = input_df.reindex(columns=feature_names)

    # Preprocess
    transformed = preprocessor.transform(input_df)

    # Predict probability
    pd_default = model.predict_proba(transformed)[0][1]

    return pd_default


# -------------------------------------------
# 4. Convert PD into risk bands
# -------------------------------------------
def risk_category(pd):
    if pd >= 0.70:
        return "Very High Risk"
    elif pd >= 0.50:
        return "High Risk"
    elif pd >= 0.30:
        return "Medium Risk"
    elif pd >= 0.15:
        return "Low Risk"
    else:
        return "Very Low Risk"


# -------------------------------------------
# 5. Main program logic
# -------------------------------------------
def main():
    print("\n==============================")
    print("  LOAN DEFAULT PREDICTOR v1.0")
    print("==============================\n")

    user_df = collect_user_input()
    pd_value = make_prediction(user_df)
    category = risk_category(pd_value)

    print("\n-----------------------------------")
    print(f" Probability of Default (PD): {pd_value:.4f}")
    print(f" Risk Category: {category}")
    print("-----------------------------------\n")

    print("Interpretation:")
    print("- Low PD → customer likely to repay")
    print("- High PD → customer poses repayment risk\n")


if __name__ == "__main__":
    main()
