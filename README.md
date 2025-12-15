# Loan Default Prediction Model

## Executive Summary

This project presents a **Loan Default Prediction Model**, which utilizes historical performance data from 250,000 loans to predict the likelihood of loan default. The model aims to reduce credit losses in the digital lending portfolio by correctly identifying high-risk borrowers. With an **AUC of 0.82**, the model helps the bank avoid an estimated **18–25%** of annual default losses while keeping the impact on loan approval volumes minimal.

## Business Problem

The current digital lending portfolio faces a **12% six-month default rate**, resulting in significant annual financial loss. The lack of predictive screening relies heavily on manual decision-making, leading to:

* **High false approvals**: Risky loans that pass through (approximately 1 in 8 loans default).
* **Inefficient rejections**: Creditworthy customers incorrectly declined.

This leads to avoidable losses and missed revenue opportunities, underlining the need for a more efficient predictive solution.

## Model Overview

* **Model Type**: Random Forest with calibrated probability output
* **Prediction Horizon**: 6-month probability of default
* **Key Performance Metrics**:

  * **AUC**: 0.82 (The model ranks risky vs. safe borrowers correctly 82% of the time).
  * **Risk Capture Rate**: The top 20% highest scores capture 61% of all defaults.
  * **False Positive Rate**: 14% at recommended threshold.
  * **Calibration Accuracy**: 28-32% observed defaults when 30% default risk is predicted.

  Try The model here https://credit-risks.streamlit.app/

## Portfolio Impact

* **Default Reduction**: The model helps avoid **18-25%** of total portfolio losses (roughly 2.2-3.0% of total losses).
* **Approval Impact**: The risk cutoff affects approximately **20%** of applicants, with **61%** identified as high-risk and **39%** potentially eligible under a tighter review.
* **Financial Outcome**: Expected annual loss avoided is between **USD $1.4M–$2.1M**.

## Risks & Limitations

* **Data Drift**: Accuracy may decrease by 5-10% if customer behavior or macro conditions change, requiring quarterly monitoring.
* **Operational Misuse**: A high threshold may disproportionately drop approvals (e.g., 30% cutoff results in a 15–18% approval drop).
* **Model Bias**: Demographic features must be monitored to ensure predicted risk variance across groups remains under 3%.

## Recommendations

### Recommended Operating Threshold:

* **20% risk score cutoff**: This balances loss reduction (18-25%) with manageable approval impact (8-10%).

### Decision Options:

* **Conservative (High Protection)**: 15% cutoff

  * Loss Reduction: ~30%
  * Approval Impact: ~15% decline
* **Balanced Strategy (Recommended)**: 20% cutoff

  * Loss Reduction: 18-25%
  * Approval Impact: 8-10%
* **Growth-Focused**: 25% cutoff

  * Loss Reduction: 10-12%
  * Approval Impact: 4-6%

### Next Steps:

1. **Implement the model** in the digital loan workflow with monthly monitoring dashboards.
2. **Pilot the model** for 60 days to confirm real-world risk capture.
3. **Review the operational threshold** quarterly to ensure optimal performance.

---

### Additional Information

For a deeper understanding of the model's mechanics and to contribute to further development, feel free to explore the code and data in this repository. Please refer to the documentation for setup instructions, model deployment guidelines, and further discussions on performance and tuning options.

