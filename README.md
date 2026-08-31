# Credit Risk Prediction — XGBoost vs Random Forest

Machine Learning project for **credit risk classification** using **XGBoost** and **Random Forest**.

This project compares both models to predict whether a borrower is likely to default on a loan and identifies the most influential factors affecting the prediction.

## 🎯 Objectives

- Implement XGBoost and Random Forest for credit risk classification
- Compare model performance using Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and ROC-AUC
- Identify the most influential features in credit risk prediction

## 📊 Dataset

Credit Risk Dataset consisting of:

- **32,581 records**
- **12 features**
- Numerical and categorical variables
- Target: `loan_status`

Main features include:

`person_age`, `person_income`, `person_home_ownership`, `person_emp_length`, `loan_intent`, `loan_grade`, `loan_amnt`, `loan_int_rate`, `loan_percent_income`, and credit history features.

## 🔄 Workflow

```text
EDA
 ↓
Data Preprocessing
 ↓
Feature Engineering
 ↓
Data Splitting
 ↓
XGBoost & Random Forest
 ↓
Hyperparameter Tuning
 ↓
Model Evaluation
 ↓
Feature Importance & ROC-AUC
```

## ⚙️ Preprocessing

- Missing value imputation using median
- Outlier detection using IQR
- Feature engineering
- Data binning
- One-Hot Encoding
- Label Encoding
- StandardScaler

Dataset split:

**70% Training — 15% Validation — 15% Testing**

## 🤖 Models

### XGBoost
Gradient boosting ensemble model for tabular classification.

### Random Forest
Ensemble learning model based on multiple decision trees.

## 🏆 Results

### Test Accuracy

| Model | Accuracy |
|---|---:|
| XGBoost Tuning 1 | 92.09% |
| **XGBoost Tuning 2** | **95.50%** |
| **XGBoost Tuning 3** | **95.63%** |
| Random Forest Tuning 1 | 81.53% |
| Random Forest Tuning 2 | 84.49% |
| Random Forest Tuning 3 | 86.18% |

Although **XGBoost Tuning 3** achieved the highest test accuracy, **XGBoost Tuning 2** was selected as the preferred model because it provided more stable performance across training, validation, and testing data. :contentReference[oaicite:0]{index=0}

### ROC-AUC

**XGBoost Tuning 3: 0.986**

**XGBoost Tuning 2: 0.984**

## 🔎 Feature Importance

The most influential features in XGBoost Tuning 2 were:

- `loan_to_income_ratio`
- `loan_int_rate`
- `int_rate_to_loan_amt_ratio`
- `person_age`
- `person_emp_length`

`loan_to_income_ratio` was the most influential feature in the model. :contentReference[oaicite:1]{index=1}

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Jupyter Notebook / Google Colab

## 👨‍💻 Author

**Rafki Sahasika Riyuda**

Computer Systems Graduate  
AI / Machine Learning • Computer Vision • Data Analytics

📧 rafkirafki551@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/rafkiSahasikaRiyuda)

---

⭐ This project was developed as part of a Machine Learning practicum at Universitas Sriwijaya.
