
# 🔐 SIM Swap Fraud Detection in Kenya's M-Pesa Ecosystem

A supervised machine learning system that detects SIM swap fraud in Kenya's M-Pesa mobile money ecosystem in real time — built as a capstone project for the GoMyCode Data Science Bootcamp 2026.

🚀 **[Live Demo](https://m-pesa-sim-swap-fraud-detection-model-q3fgdzajmkzeuopqfza4sa.streamlit.app)**

---

## 📌 Problem Statement

SIM swap fraud is one of the most damaging threats to Kenya's mobile money ecosystem. By exploiting identity verification gaps in the telecommunications sector, fraudsters hijack a subscriber's phone number, bypass OTP authentication, drain M-Pesa wallets, and exhaust credit facilities such as Fuliza and M-Shwari — all within minutes.

Kenya's CBK Financial Sector Stability Report 2024 documented KSh 810 million in mobile banking losses, with fraud incidents more than doubling year-on-year. Kenya has been identified as the highest-risk mobile money market in Africa, with 51% of transactions flagged as suspicious by cybersecurity firm Evina.

This project builds a proactive, ML-powered behavioral detection system that flags high-risk transactions in real time — giving fintech operators an automated defense against a fraud pattern that reactive security measures consistently fail to match.

---

## 🎯 Project Objectives

- Detect fraudulent M-Pesa transactions that follow a SIM swap attack pattern
- Engineer behavioral proxy features that capture account-drain signals from transaction data
- Train a supervised classification model under realistic class imbalance conditions
- Deploy a real-time fraud scoring interface accessible to operators

---

## 🗂️ Project Structure

```
├── app.py                    # Streamlit web application
├── xgboost_fraud_model.pkl   # Trained XGBoost model
├── scaler.pkl                # Fitted StandardScaler
├── requirements.txt          # Python dependencies
└── README.md
```

---

## 🔬 Methodology

### Phase 1 — Data Exploration & Preparation
- Loaded and inspected 120,000 synthetic M-Pesa transaction records
- Confirmed zero missing values and zero duplicates
- Identified and removed 1 zero-amount transaction
- One-hot encoded categorical variables (transaction type, device type, region, day of week)
- Engineered 4 SIM swap behavioral proxy features:
  - `balance_depletion_rate` — how aggressively the sender's balance was consumed
  - `is_balance_wipeout` — flag for near-total account drain (balance < KES 100)
  - `sender_balance_ratio` — ratio of balance after to before the transaction
  - `is_high_value` — flag for transactions above the 90th percentile of amount

### Phase 2 — Data Visualisation & Model Selection
- Correlation heatmap confirmed `is_balance_wipeout` (0.78) and `sender_balance_ratio` (-0.62) as strongest fraud predictors
- Scatter plots confirmed no linear decision boundary exists — ruling out Logistic Regression as primary model
- Severe class imbalance confirmed: 97.1% legitimate vs 2.9% fraud (33:1 ratio)
- XGBoost selected as primary model based on visual and statistical evidence

### Phase 3 — Model Training & Testing
- Stratified 80/20 train/test split to preserve fraud ratio
- StandardScaler fitted on training data only — applied to both sets to prevent leakage
- Class imbalance handled via `scale_pos_weight=33` — confirmed superior to SMOTE for this dataset
- Hyperparameter tuning via RandomizedSearchCV (30 combinations, 5-fold CV)
- Three models trained and evaluated: Logistic Regression (baseline), XGBoost (primary), LightGBM (comparison)

### Phase 4 — Deployment & Monitoring
- Model and scaler saved as `.pkl` files using joblib
- Streamlit web application built with real-time feature engineering
- Deployed via GitHub to Streamlit Community Cloud

---

## 📊 Model Results

| Model | Precision | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|
| Logistic Regression (Baseline) | 0.52 | 0.66 | 0.58 | 0.83 |
| LightGBM (Comparison) | 0.96 | 0.65 | 0.78 | 0.83 |
| **XGBoost (Primary)** | **1.00** | **0.66** | **0.79** | **0.83** |

**XGBoost achieved perfect precision with zero false positives** — every fraud alert raised was genuine. This means zero legitimate customers incorrectly blocked, which is critical for maintaining customer trust in a production deployment.

> Note: Accuracy is excluded as a primary metric due to the 33:1 class imbalance making it statistically misleading.

---

## ⚙️ Engineered Features

| Feature | Correlation with Fraud | Description |
|---|---|---|
| `is_balance_wipeout` | 0.78 | Account nearly emptied after transaction |
| `sender_balance_ratio` | -0.62 | Ratio of balance after to before |
| `balance_depletion_rate` | 0.50 | Amount sent relative to available balance |
| `is_high_value` | 0.09 | Transaction above 90th percentile |

---

## 🚀 Live App

The fraud detection system is deployed at:

**[https://m-pesa-sim-swap-fraud-detection-model-q3fgdzajmkzeuopqfza4sa.streamlit.app](https://m-pesa-sim-swap-fraud-detection-model-q3fgdzajmkzeuopqfza4sa.streamlit.app)**

### How to use
1. Enter transaction details — amount, balances before and after, hour, device type, region
2. Click **Check Transaction**
3. The model returns a fraud probability score and recommended actions

### Test it yourself

**Fraud scenario:**
- Amount: `4900`, Sender Balance Before: `5000`, Sender Balance After: `50`
- Expected: 🚨 FRAUDULENT — ~93% probability

**Legitimate scenario:**
- Amount: `500`, Sender Balance Before: `25000`, Sender Balance After: `24500`
- Expected: ✅ LEGITIMATE — ~24% probability

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.12 | Core language |
| Pandas & NumPy | Data manipulation |
| Scikit-Learn | Preprocessing, evaluation, Logistic Regression |
| XGBoost | Primary fraud detection model |
| LightGBM | Comparison model |
| Matplotlib & Seaborn | Visualisation |
| ydata-profiling | Automated data profiling |
| Streamlit | Web application |
| joblib | Model serialisation |
| GitHub | Version control |
| Streamlit Community Cloud | Deployment |

---

## 📦 Installation & Local Setup

```bash
# Clone the repository
git clone https://github.com/Everlyne-Nasimiyu/M-pesa-Sim-Swap-Fraud-Detection-Model.git
cd M-pesa-Sim-Swap-Fraud-Detection-Model

# Install dependencies
pip install -r requirements.txt

# Run the app locally
streamlit run app.py
```

---

## 📋 Requirements

```
streamlit
xgboost
lightgbm
scikit-learn
pandas
numpy
joblib
```

---

## 🔭 Monitoring Plan

| Check | Frequency | Alert Threshold |
|---|---|---|
| Precision & Recall on new transactions | Weekly | Precision < 0.80 or Recall < 0.55 |
| Feature distribution drift | Monthly | Significant shift from training stats |
| Fraud pattern changes | Monthly | New attack vectors from CBK bulletins |
| Model retraining | Quarterly or on trigger | F1 < 0.70 or 3+ months of new data |

---

## ⚠️ Limitations

- The dataset is synthetic — real-world performance requires validation against actual M-Pesa transaction data
- The dataset lacks native SIM swap signals (IMEI changes, SIM swap timestamps) — behavioral proxy features are used instead
- Recall of 0.66 means approximately 34% of fraud cases are missed — a limitation of the synthetic dataset ceiling, not a pipeline error

