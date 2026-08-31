# Credit Risk Prediction — XGBoost vs Random Forest

A Machine Learning project for predicting credit default risk using **XGBoost** and **Random Forest**. This project explores the complete machine learning workflow, from exploratory data analysis and preprocessing to model training, hyperparameter tuning, evaluation, and feature importance analysis.

The main objective is to compare the performance and stability of XGBoost and Random Forest in classifying whether a borrower is likely to default on a loan.

---

## 📌 Project Overview

Credit risk prediction is an important problem in the financial industry. Incorrectly identifying borrowers with a high probability of default can lead to significant financial losses.

This project applies two ensemble learning algorithms:

- **XGBoost (Extreme Gradient Boosting)**
- **Random Forest**

Both models are trained and evaluated using a Credit Risk Dataset containing borrower and loan characteristics.

The project focuses on:

- Exploratory Data Analysis (EDA)
- Missing value handling
- Outlier detection and removal
- Feature engineering
- Data binning
- Categorical encoding
- Feature scaling
- Dataset splitting
- Model training
- Hyperparameter tuning
- Performance comparison
- Feature importance analysis
- ROC Curve and AUC evaluation

---

## 🎯 Objectives

The project aims to:

1. Implement XGBoost and Random Forest for credit risk classification.
2. Compare the performance of both models using:
   - Accuracy
   - Precision
   - Recall
   - F1-Score
   - Confusion Matrix
   - ROC-AUC
3. Identify the features that contribute most to credit risk prediction.
4. Determine which model provides the most stable performance across training, validation, and testing datasets.

---

## 📊 Dataset

The project uses a **Credit Risk Dataset** containing:

- **32,581 rows**
- **12 columns**

The dataset contains numerical and categorical variables describing borrower characteristics and loan information.

### Features

| Feature | Description |
|---|---|
| `person_age` | Borrower's age |
| `person_income` | Annual borrower income |
| `person_home_ownership` | Home ownership status |
| `person_emp_length` | Employment length |
| `loan_intent` | Purpose of the loan |
| `loan_grade` | Loan grade from A to G |
| `loan_amnt` | Requested loan amount |
| `loan_int_rate` | Loan interest rate |
| `loan_status` | Target variable indicating loan status |
| `loan_percent_income` | Loan amount as a percentage of income |
| `cb_person_default_on_file` | Previous default indicator |
| `cb_person_cred_hist_length` | Length of credit history |

The target variable is:

```text
loan_status
0 → No default
1 → Default
