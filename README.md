# Credit Risk Prediction: XGBoost vs Random Forest

<p align="center">
  <b>Machine Learning Classification Project for Credit Default Risk Prediction</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-ML-red?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Random%20Forest-ML-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Scikit--learn-ML-orange?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-purple?style=for-the-badge&logo=pandas&logoColor=white" />
</p>

---

## 📌 Overview

This project focuses on **credit risk prediction using Machine Learning**, specifically by comparing the performance of **XGBoost** and **Random Forest** classification algorithms.

The main objective is to build a predictive model capable of classifying whether a borrower is likely to **default on a loan or not**, while also analyzing the factors that contribute most significantly to the prediction.

The project covers a complete Machine Learning workflow:

**Dataset Selection → Exploratory Data Analysis → Data Preprocessing → Feature Engineering → Dataset Splitting → Model Training → Hyperparameter Tuning → Model Evaluation → Feature Importance → ROC-AUC Analysis**

The experiment was conducted using a **Credit Risk Dataset containing 32,581 records and 12 columns**. :contentReference[oaicite:1]{index=1}

---

## 🎯 Objectives

The main objectives of this project are:

- Implement **XGBoost** and **Random Forest** for credit risk classification.
- Compare the performance of both algorithms.
- Evaluate models using:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - Confusion Matrix
  - ROC Curve
  - AUC
- Identify the most influential features in credit risk prediction.
- Determine which model provides the most stable performance on unseen data. :contentReference[oaicite:2]{index=2}

---

## 🧠 Problem Statement

Credit risk assessment is an important task in the financial industry because inaccurate predictions can potentially lead to financial losses.

This project addresses the following questions:

1. How can XGBoost and Random Forest be applied to predict credit risk?
2. How do the performances of XGBoost and Random Forest compare when classifying borrower default risk?
3. Which features have the greatest influence on the prediction results? :contentReference[oaicite:3]{index=3}

---

# 📊 Dataset

The dataset used in this project is a **Credit Risk Dataset** containing information about borrowers and their loan characteristics.

### Dataset Size

| Property | Value |
|---|---:|
| Number of Records | 32,581 |
| Number of Features | 12 |
| Numerical Features | 8 |
| Categorical Features | 4 |
| Target | `loan_status` |

The dataset contains borrower information such as age, income, home ownership, employment length, loan amount, interest rate, and previous default history. :contentReference[oaicite:4]{index=4}

---

## 📋 Dataset Features

| Feature | Description | Type |
|---|---|---|
| `person_age` | Borrower's age | Numerical |
| `person_income` | Annual borrower income | Numerical |
| `person_home_ownership` | Home ownership status | Categorical |
| `person_emp_length` | Employment length | Numerical |
| `loan_intent` | Purpose of the loan | Categorical |
| `loan_grade` | Loan grade from A to G | Categorical |
| `loan_amnt` | Requested loan amount | Numerical |
| `loan_int_rate` | Loan interest rate | Numerical |
| `loan_status` | Loan repayment/default status | Target |
| `loan_percent_income` | Loan amount as percentage of income | Numerical |
| `cb_person_default_on_file` | Previous default indicator | Categorical |
| `cb_person_cred_hist_length` | Length of credit history | Numerical |

The target variable is:

```text
0 → Non-default
1 → Default
```

The dataset also contains a class imbalance issue, which is an important consideration in credit risk classification. :contentReference[oaicite:5]{index=5}

---

# 🔎 Exploratory Data Analysis

Before training the models, Exploratory Data Analysis (EDA) was performed to understand the quality and characteristics of the dataset.

The EDA process included:

- Descriptive statistics
- Missing value analysis
- Outlier detection
- Correlation analysis
- Data distribution analysis
- Boxplot visualization
- Correlation heatmap

### Missing Values

The main missing values were found in:

| Feature | Missing Values |
|---|---:|
| `person_emp_length` | 887 |
| `loan_int_rate` | 3,095 |

Other variables did not contain missing values. :contentReference[oaicite:6]{index=6}

### Outlier Analysis

Several extreme values were identified during EDA.

One notable example was:

```text
person_age = 144
```

which was identified as an unrealistic/extreme value and treated as an outlier during preprocessing. Other extreme values were also observed in `loan_int_rate` and `loan_amnt`. :contentReference[oaicite:7]{index=7}

### Correlation Analysis

The correlation analysis showed a strong relationship between:

```text
person_age
        ↕
cb_person_cred_hist_length
```

with a correlation coefficient of approximately:

```text
0.86
```

Other notable relationships were found among loan amount, interest rate, and loan-to-income related variables. :contentReference[oaicite:8]{index=8}

---

# 🧹 Data Preprocessing

Several preprocessing techniques were applied before model training.

## 1. Missing Value Imputation

Missing values in:

- `person_emp_length`
- `loan_int_rate`

were handled using **median imputation**.

Median was selected because it is less sensitive to extreme values than the mean. :contentReference[oaicite:9]{index=9}

---

## 2. Outlier Handling

The **Interquartile Range (IQR)** method was used to detect extreme observations.

The general rule applied was:

```text
Lower Bound = Q1 - 1.5 × IQR

Upper Bound = Q3 + 1.5 × IQR
```

Observations outside these boundaries were treated as outliers. :contentReference[oaicite:10]{index=10}

---

## 3. Feature Engineering

Additional features were created to represent relationships between loan and borrower characteristics.

### Created Features

```text
loan_to_income_ratio
loan_to_employment_length_ratio
int_rate_to_loan_amt_ratio
```

These features were designed to provide additional information about the relationship between:

- Loan amount
- Borrower income
- Employment length
- Interest rate :contentReference[oaicite:11]{index=11}

---

## 4. Data Binning

Binning was applied to numerical variables to create meaningful groups.

The following variables were grouped:

```text
person_income
loan_amnt
```

This produced categorical representations such as income groups and loan amount groups. :contentReference[oaicite:12]{index=12}

---

## 5. Categorical Encoding

Categorical variables were converted into numerical representations using:

### One-Hot Encoding

Applied to categorical features without intrinsic ordering, such as:

```text
person_home_ownership
loan_intent
```

### Label Encoding

Applied to ordinal/categorical variables such as:

```text
loan_grade
cb_person_default_on_file
```

:contentReference[oaicite:13]{index=13}

---

## 6. Feature Scaling

Numerical features were standardized using:

```python
StandardScaler()
```

The purpose was to standardize numerical variables into a comparable scale. :contentReference[oaicite:14]{index=14}

---

# ✂️ Dataset Splitting

The processed dataset was divided into three subsets:

| Dataset | Ratio |
|---|---:|
| Training | 70% |
| Validation | 15% |
| Testing | 15% |

The splitting process used:

```python
train_test_split()
```

with:

```python
random_state=42
stratify=y
```

This approach was used to ensure that the training, validation, and testing sets preserved the class distribution. :contentReference[oaicite:15]{index=15}

Example:

```python
X = df_balanced.drop(columns=['loan_status'])
y = df_balanced['loan_status']

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)
```

---

# 🤖 Machine Learning Models

Two ensemble learning algorithms were evaluated.

---

## XGBoost

**XGBoost (Extreme Gradient Boosting)** is a boosting-based ensemble algorithm that builds decision trees sequentially, with each subsequent tree attempting to reduce errors made by previous trees.

The model was selected because of its:

- Computational efficiency
- Predictive performance
- Regularization capability
- Ability to handle complex tabular datasets :contentReference[oaicite:16]{index=16}

Basic implementation:

```python
from xgboost import XGBClassifier

XGB = XGBClassifier(
    random_state=42
)

XGB.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=True
)
```

---

## Random Forest

Random Forest is an ensemble learning method that combines multiple decision trees.

Each tree produces a prediction, and the final classification is obtained through aggregation or majority voting. :contentReference[oaicite:17]{index=17}

Implementation:

```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    random_state=42
)

rf_model.fit(
    X_train,
    y_train
)
```

---

# ⚙️ Hyperparameter Tuning

Three manual tuning configurations were tested for each model.

## XGBoost

| Parameter | Tuning 1 | Tuning 2 | Tuning 3 |
|---|---:|---:|---:|
| `n_estimators` | 300 | 500 | 750 |
| `learning_rate` | 0.05 | 0.07 | 0.10 |
| `max_depth` | 3 | 5 | 6 |
| `subsample` | 0.8 | 0.9 | 1.0 |
| `colsample_bytree` | 0.8 | 0.9 | 1.0 |
| `random_state` | 42 | 42 | 42 |

## Random Forest

| Parameter | Tuning 1 | Tuning 2 | Tuning 3 |
|---|---:|---:|---:|
| `n_estimators` | 250 | 500 | 750 |
| `max_depth` | 3 | 5 | 8 |
| `min_samples_split` | 10 | 15 | 20 |
| `min_samples_leaf` | 4 | 5 | 10 |
| `max_features` | sqrt | sqrt | sqrt |
| `random_state` | 42 | 42 | 42 |

The hyperparameter configurations above were tested manually to observe the effect of model complexity and regularization on classification performance. :contentReference[oaicite:18]{index=18}

---

# 📈 Model Evaluation

The models were evaluated using multiple metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC Curve
- AUC

These metrics were selected to provide a more comprehensive assessment than accuracy alone, especially for a classification problem involving class imbalance. :contentReference[oaicite:19]{index=19}

---

# 🏆 Results

## Test Accuracy Comparison

| Model | Test Accuracy |
|---|---:|
| XGBoost Tuning 1 | **92.09%** |
| XGBoost Tuning 2 | **95.50%** |
| XGBoost Tuning 3 | **95.63%** |
| Random Forest Tuning 1 | **81.53%** |
| Random Forest Tuning 2 | **84.49%** |
| Random Forest Tuning 3 | **86.18%** |

The results show that XGBoost achieved substantially better testing performance than Random Forest across the tuning configurations. :contentReference[oaicite:20]{index=20}

---

# ⭐ Selected Model: XGBoost Tuning 2

Although **XGBoost Tuning 3** achieved the highest test accuracy, **XGBoost Tuning 2** was selected as the preferred configuration because it demonstrated more consistent performance across the training, validation, and testing datasets. :contentReference[oaicite:21]{index=21}

### XGBoost Tuning 2

| Dataset | Accuracy |
|---|---:|
| Training | 96.35% |
| Validation | 95.89% |
| Testing | **95.50%** |

This relatively consistent performance suggests stronger generalization compared with configurations that showed greater variation across subsets.

---

# 📊 Detailed XGBoost Tuning 2 Performance

### Test Set

| Metric | Class 0 | Class 1 |
|---|---:|---:|
| Precision | 0.923761 | 0.991163 |
| Recall | 0.991814 | 0.918145 |
| F1-Score | 0.956579 | 0.953258 |

Overall test accuracy:

```text
95.50%
```

These results indicate that XGBoost Tuning 2 maintained strong performance in identifying both classes. :contentReference[oaicite:22]{index=22}

---

# 🔲 Confusion Matrix

For **XGBoost Tuning 2**, the confusion matrix on the test dataset was:

| | Prediction: Negative | Prediction: Positive |
|---|---:|---:|
| Actual: Negative | 3,635 | 30 |
| Actual: Positive | 300 | 3,365 |

Therefore:

```text
TP = 3,365
TN = 3,635
FP = 30
FN = 300
```

The low number of false positives indicates that the model was able to classify the negative class effectively, while the false negatives show that some positive/default cases remained difficult to identify. :contentReference[oaicite:23]{index=23}

---

# 📈 ROC Curve & AUC

ROC-AUC analysis was used to evaluate the classification capability of the models.

### AUC Results

| Model | AUC |
|---|---:|
| XGBoost Default | 0.983 |
| XGBoost Tuning 2 | **0.984** |
| XGBoost Tuning 3 | **0.986** |
| Random Forest Default | 0.985 |
| Random Forest Tuning 1 | 0.892 |

The highest AUC was achieved by:

```text
XGBoost Tuning 3
AUC = 0.986
```

XGBoost Tuning 2 also achieved a strong:

```text
AUC = 0.984
```

The results indicate strong classification capability for the evaluated models. :contentReference[oaicite:24]{index=24}

---

# 🔎 Feature Importance

Feature importance analysis was conducted to identify the variables that contributed most strongly to model predictions.

## XGBoost Tuning 2

The most influential features included:

| Feature | F Score |
|---|---:|
| `loan_to_income_ratio` | 1274 |
| `loan_int_rate` | 1193 |
| `int_rate_to_loan_amt_ratio` | 1191 |
| `person_age` | 910 |
| `person_emp_length` | 868 |

The most influential feature was:

```text
loan_to_income_ratio
```

This suggests that the relationship between the amount of the loan and the borrower's income is highly influential in the XGBoost model. :contentReference[oaicite:25]{index=25}

---

## Random Forest Tuning 3

The most influential features included:

```text
loan_amount_group_very_large
income_group_low-middle
loan_to_income_ratio
loan_int_rate
loan_percent_income
cb_person_cred_hist_length
```

The strongest feature was:

```text
loan_amount_group_very_large
Feature Importance ≈ 0.22
```

Other important features included:

```text
income_group_low-middle ≈ 0.15
loan_to_income_ratio ≈ 0.14
```

:contentReference[oaicite:26]{index=26}

---

# 🧠 Key Findings

Based on the experiments, several findings were obtained:

### 1. XGBoost Outperformed Random Forest

XGBoost produced higher test accuracy across the evaluated tuning configurations.

### 2. XGBoost Tuning 3 Achieved the Highest Accuracy

The highest test accuracy was:

```text
95.63%
```

achieved by **XGBoost Tuning 3**. :contentReference[oaicite:27]{index=27}

### 3. XGBoost Tuning 2 Was Selected for Stability

Despite having slightly lower accuracy than Tuning 3, XGBoost Tuning 2 was preferred because of its more consistent performance across training, validation, and test data. :contentReference[oaicite:28]{index=28}

### 4. Financial Features Were Highly Influential

Features related to:

- Loan amount
- Income
- Interest rate
- Loan-to-income ratio

showed strong contributions to the prediction process.

### 5. Model Performance Should Not Be Evaluated Using Accuracy Alone

Precision, recall, F1-score, confusion matrix, and ROC-AUC were also considered to obtain a broader understanding of model behavior.

---

# 🔬 Machine Learning Pipeline

```text
                    ┌─────────────────────┐
                    │   Credit Risk Data  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │        EDA          │
                    │                     │
                    │ • Statistics        │
                    │ • Missing Values    │
                    │ • Outliers          │
                    │ • Correlation       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Data Preprocessing  │
                    │                     │
                    │ • Imputation        │
                    │ • Outlier Handling  │
                    │ • Encoding          │
                    │ • Scaling           │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Feature Engineering │
                    │                     │
                    │ • Ratios            │
                    │ • Binning           │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Dataset Splitting  │
                    │                     │
                    │ Train 70%           │
                    │ Validation 15%      │
                    │ Test 15%            │
                    └──────────┬──────────┘
                               │
                ┌──────────────┴──────────────┐
                ▼                             ▼
       ┌─────────────────┐           ┌─────────────────┐
       │    XGBoost      │           │  Random Forest  │
       └────────┬────────┘           └────────┬────────┘
                │                             │
                ▼                             ▼
       ┌─────────────────┐           ┌─────────────────┐
       │ Hyperparameter  │           │ Hyperparameter  │
       │     Tuning      │           │     Tuning      │
       └────────┬────────┘           └────────┬────────┘
                │                             │
                └──────────────┬──────────────┘
                               ▼
                    ┌─────────────────────┐
                    │ Model Evaluation    │
                    │                     │
                    │ Accuracy            │
                    │ Precision           │
                    │ Recall              │
                    │ F1-Score            │
                    │ Confusion Matrix     │
                    │ ROC-AUC             │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Feature Importance  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Best Configuration  │
                    │                     │
                    │ XGBoost Tuning 2    │
                    └─────────────────────┘
```

---

# 🛠️ Tech Stack

The project was implemented using:

### Programming Language

- Python

### Data Processing

- Pandas
- NumPy

### Machine Learning

- Scikit-learn
- XGBoost

### Visualization

- Matplotlib
- Seaborn

### Development Environment

- Jupyter Notebook
- Google Colab

---

# 📂 Repository Structure

```text
credit-risk-prediction/
│
├── dataset/
│   └── credit_risk_dataset.csv
│
├── notebook/
│   └── credit_risk_prediction.ipynb
│
├── images/
│   ├── eda/
│   ├── correlation/
│   ├── feature-importance/
│   ├── confusion-matrix/
│   └── roc-curve/
│
├── report/
│   └── UAS_Laporan_Praktikum_Machine_Learning.pdf
│
├── requirements.txt
└── README.md
```

> Struktur folder di atas merupakan struktur yang direkomendasikan untuk repository. Sesuaikan nama file dengan struktur aktual repository.

---

# 🚀 Installation & Usage

## 1. Clone Repository

```bash
git clone https://github.com/USERNAME/credit-risk-prediction.git
```

Navigate into the project directory:

```bash
cd credit-risk-prediction
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Example dependencies:

```txt
numpy
pandas
scikit-learn
xgboost
matplotlib
seaborn
jupyter
```

---

## 4. Run Jupyter Notebook

```bash
jupyter notebook
```

Then open the project notebook:

```text
credit_risk_prediction.ipynb
```

---

# 📊 Recommended Visualizations

The repository can include the following visualizations generated during the experiment:

- Dataset distribution
- Missing value analysis
- Boxplot for outliers
- Correlation heatmap
- XGBoost feature importance
- Random Forest feature importance
- Confusion matrix
- Accuracy comparison
- ROC curve

Suggested structure:

```text
images/
│
├── eda/
│   ├── missing-values.png
│   ├── boxplot.png
│   └── distributions.png
│
├── correlation/
│   └── correlation-heatmap.png
│
├── feature-importance/
│   ├── xgboost-feature-importance.png
│   └── random-forest-feature-importance.png
│
├── confusion-matrix/
│   ├── xgboost.png
│   └── random-forest.png
│
└── roc-curve/
    └── roc-curve.png
```

---

# 📚 Methodology Summary

The methodology implemented in this project can be summarized as follows:

```text
1. Dataset Selection
        ↓
2. Exploratory Data Analysis
        ↓
3. Missing Value Handling
        ↓
4. Outlier Detection & Handling
        ↓
5. Feature Engineering
        ↓
6. Data Binning
        ↓
7. Categorical Encoding
        ↓
8. Feature Scaling
        ↓
9. Train / Validation / Test Split
        ↓
10. XGBoost Training
        ↓
11. Random Forest Training
        ↓
12. Hyperparameter Tuning
        ↓
13. Model Evaluation
        ↓
14. Feature Importance
        ↓
15. ROC-AUC Analysis
        ↓
16. Model Comparison
```

---

# 📌 Conclusion

This project demonstrates the implementation and comparison of **XGBoost and Random Forest** for credit risk classification.

Both algorithms were able to classify credit risk effectively. However, the experimental results showed that **XGBoost provided better overall testing performance and greater consistency across training, validation, and testing datasets**.

Although **XGBoost Tuning 3** achieved the highest test accuracy of **95.63%**, **XGBoost Tuning 2** was selected as the preferred model because it provided a stronger balance between performance and stability. :contentReference[oaicite:29]{index=29}

The feature importance analysis also demonstrated that financial characteristics, particularly the relationship between loan amount and income, interest rate, and loan-related ratios, played an important role in the prediction process. :contentReference[oaicite:30]{index=30}

---

# 🔮 Future Improvements

Potential future development for this project includes:

- More systematic hyperparameter optimization.
- Grid Search or Random Search for parameter optimization.
- More advanced handling of class imbalance.
- Additional ensemble learning approaches.
- Explainable AI techniques for deeper model interpretation.
- Deployment of the model as an interactive application.
- Integration with a prediction API.

These improvements are also aligned with the recommendations presented in the project report. :contentReference[oaicite:31]{index=31}

---

# 🎓 Academic Context

This project was developed as part of the:

**Machine Learning Practicum**

**Computer Systems Department**  
**Faculty of Computer Science**  
**Universitas Sriwijaya**

### Author

**Rafki Sahasika Riyuda**

2025

---

# 👨‍💻 Author

**Rafki Sahasika Riyuda**

Computer Systems Graduate  
AI / Machine Learning • Computer Vision • Data Analytics • Graphic Design

📧 Email: `rafkirafki551@gmail.com`

🔗 LinkedIn:  
https://www.linkedin.com/in/rafkiSahasikaRiyuda

---

## ⭐ Acknowledgement

This repository contains the implementation and documentation of a Machine Learning experiment comparing **XGBoost and Random Forest** for credit risk prediction.

If you find this project useful or interesting, consider giving the repository a ⭐.

---

<p align="center">
  <b>Built with Python & Machine Learning</b>
</p>
