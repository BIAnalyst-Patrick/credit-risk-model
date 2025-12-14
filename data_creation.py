import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta

np.random.seed(42)

# -----------------------------
# 1) SIZE OF DATASET
# -----------------------------
N = 30000

# -----------------------------
# 2) BASIC CUSTOMER & LOAN IDs
# -----------------------------
customer_id = np.arange(100000, 100000 + N)
loan_id = np.arange(500000, 500000 + N)

# -----------------------------
# 3) DEMOGRAPHICS
# -----------------------------
age = np.random.normal(38, 10, N).astype(int)
age = np.clip(age, 21, 65)

gender = np.random.choice(['Male', 'Female'], N, p=[0.55, 0.45])
marital_status = np.random.choice(['Single', 'Married', 'Divorced'], N, p=[0.45, 0.45, 0.10])
education_level = np.random.choice(
    ['High School', 'Diploma', 'Bachelors', 'Masters'],
    N, 
    p=[0.35, 0.25, 0.30, 0.10]
)
employment_status = np.random.choice(
    ['Employed', 'Self-employed', 'Unemployed'],
    N,
    p=[0.70, 0.20, 0.10]
)

# -----------------------------
# 4) INCOME
# -----------------------------
monthly_income = np.random.normal(60000, 25000, N).astype(int)
monthly_income = np.clip(monthly_income, 10000, 300000)
annual_income = monthly_income * 12
other_income = np.random.normal(5000, 3000, N).clip(0)

# -----------------------------
# 5) CREDIT HISTORY
# -----------------------------
credit_score = np.random.normal(620, 80, N).astype(int)
credit_score = np.clip(credit_score, 300, 850)

num_of_open_accounts = np.random.poisson(3, N)
num_of_past_defaults = np.random.binomial(2, 0.15, N)

# -----------------------------
# 6) ACCOUNT BEHAVIOR
# -----------------------------
avg_monthly_balance = np.random.normal(80000, 40000, N).clip(5000)
months_with_bank = np.random.normal(60, 25, N).clip(6)
num_direct_debits = np.random.poisson(4, N)
num_card_txns_6m = np.random.poisson(25, N)

# -----------------------------
# 7) LOAN FEATURES
# -----------------------------
loan_amount = np.random.normal(300000, 200000, N).clip(50000, 2000000)
interest_rate = np.round(np.random.uniform(10, 22, N), 2)
loan_term_months = np.random.choice([12, 24, 36, 48, 60], N, p=[0.20, 0.25, 0.30, 0.15, 0.10])

loan_purpose = np.random.choice(
    ['Personal', 'Business', 'Education', 'Home Improvement', 'Medical'],
    N,
    p=[0.40, 0.25, 0.15, 0.10, 0.10]
)

# -----------------------------
# 8) DATES
# -----------------------------
start_dates = [datetime(2018, 1, 1) + timedelta(days=int(x)) for x in np.random.randint(0, 2000, N)]
application_dates = [d - timedelta(days=np.random.randint(5, 30)) for d in start_dates]
approval_dates = [d - timedelta(days=np.random.randint(1, 5)) for d in start_dates]
last_payment_date = [d + timedelta(days=np.random.randint(60, 900)) for d in start_dates]

# -----------------------------
# 9) MONTHLY INSTALLMENTS
# -----------------------------
monthly_installment = loan_amount / loan_term_months + \
    (loan_amount * (interest_rate/100)) / loan_term_months

outstanding_balance = loan_amount - (monthly_installment * np.random.uniform(0.1, 0.9, N))
outstanding_balance = outstanding_balance.clip(0)

# -----------------------------
# 10) TARGET VARIABLE: DEFAULT
# -----------------------------
# Higher risk if:
# - low credit score
# - low income relative to loan amount
# - high past defaults
# - low bank balance

risk_score = (
    (700 - credit_score)*0.015 +
    (loan_amount / (monthly_income+1))*0.02 +
    num_of_past_defaults*0.3 +
    (50000 / (avg_monthly_balance+1))*0.02
)

default_prob = 1 / (1 + np.exp(-risk_score))

loan_status = np.random.binomial(1, default_prob)  # 1 = default
loan_status = np.where(loan_status==1, "Defaulted", "Paid")

payment_delay_days = np.where(
    loan_status == "Defaulted",
    np.random.randint(90, 180, N),
    np.random.randint(0, 30, N)
)

# -----------------------------
# 11) COMBINE INTO DATAFRAME
# -----------------------------
df = pd.DataFrame({
    "customer_id": customer_id,
    "loan_id": loan_id,
    "loan_amount": loan_amount.astype(int),
    "loan_start_date": start_dates,
    "loan_term_months": loan_term_months,
    "loan_purpose": loan_purpose,
    "loan_status": loan_status,
    "outstanding_balance": outstanding_balance.astype(int),
    "monthly_installment": monthly_installment.astype(int),
    "interest_rate": interest_rate,
    "application_date": application_dates,
    "approval_date": approval_dates,
    "age": age,
    "gender": gender,
    "marital_status": marital_status,
    "education_level": education_level,
    "employment_status": employment_status,
    "annual_income": annual_income,
    "monthly_income": monthly_income,
    "other_income": other_income.astype(int),
    "credit_score": credit_score,
    "num_of_open_accounts": num_of_open_accounts,
    "num_of_past_defaults": num_of_past_defaults,
    "avg_monthly_balance": avg_monthly_balance.astype(int),
    "months_with_bank": months_with_bank.astype(int),
    "num_direct_debits": num_direct_debits,
    "num_card_txns_6m": num_card_txns_6m,
    "payment_delay_days": payment_delay_days,
    "last_payment_date": last_payment_date,
})

df.head()

# sace to csv
df.to_csv("synthetic_loan_data.csv", index=False)
