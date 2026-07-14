# 📉 Customer Churn Prediction — Telco Dataset

## Project Overview
End-to-end binary classification project predicting customer churn using the IBM/Kaggle Telco Customer Churn dataset. The goal was to identify which customers are at risk of leaving, understand *why* they churn, and build a model that a business could realistically act on — with a strong focus on the challenges of imbalanced classification.

This is **Project 4 of 10** in a Data Analysis & Machine Learning roadmap.

---

## 🛠️ Tools & Technologies
- **Python** — core analysis language
- **Pandas / NumPy** — data manipulation
- **Seaborn / Matplotlib** — visualization
- **Scikit-learn** — modeling, evaluation, preprocessing
- **imbalanced-learn (SMOTE)** — class imbalance handling
- **XGBoost** — gradient boosting comparison model
- **scikit-optimize (BayesSearchCV)** — Bayesian hyperparameter optimization
- **Streamlit** — interactive churn prediction app
- **Jupyter Lab** — development environment

---

## 📁 Project Structure
```
churn-prediction/
├── data/
│   └── data.csv                      ← raw Telco dataset
├── notebooks/
│   └── 01_churn_prediction.ipynb     ← full analysis notebook
├── app.py                            ← Streamlit churn predictor
├── model.pkl                         ← trained final model
├── model_columns.pkl                 ← expected feature column order
└── README.md
```

---

## 📥 Dataset
- **Source:** [Kaggle — Telco Customer Churn (IBM), by blastchar](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Description:** Customer-level account, service, and billing data for a telecom provider, with a binary churn label.
- **Size:** 7,043 rows × 21 columns

---

## 🧹 Data Cleaning

**The `TotalCharges` blank-string issue**
`TotalCharges` loaded as an `object` (text) column despite looking numeric. Investigation showed 11 rows failing numeric conversion — every one of them had `tenure = 0`. These were brand-new customers who hadn't completed a billing cycle yet, so the source system left the field blank rather than `0`. Since the correct value was logically known (a customer with 0 months tenure has accumulated $0 in charges), these were **filled with 0** rather than dropped — preserving 11 real customers instead of discarding them based on a data-entry quirk, not genuine missingness.

**Collapsing redundant categories**
Seven columns (`MultipleLines`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`) contained a third value like `"No phone service"` or `"No internet service"` — but this was purely redundant information already captured by `PhoneService`/`InternetService`. These were collapsed into a plain `"No"`, turning all 7 columns into clean binary features without losing any information or dropping any rows.

---

## 🔍 Exploratory Data Analysis

**Overall churn rate:** 26.5% of customers churned, 73.5% stayed — a moderate class imbalance that shaped every modeling decision downstream.

### Tenure
Customers who stayed had a median tenure of ~38 months; churned customers had a median of just ~10 months. The first several months of a customer relationship appear to be the highest-risk window for churn.

### Contract Type
The single strongest churn driver found in EDA:
| Contract | Churn Rate |
|---|---|
| Month-to-month | 42.7% |
| One year | 11.3% |
| Two year | 2.8% |

Month-to-month customers churn roughly **15x more often** than two-year contract customers — likely because there's no switching cost or penalty to leaving at any time.

### Internet Service Type
Fiber optic customers churned at **41.9%**, compared to 19.0% for DSL and 7.4% for customers with no internet service at all.

> **Domain insight:** Based on prior telecom industry experience, this pattern likely reflects more than just price. Fiber installations are often sold into areas where the underlying infrastructure can't fully support them, leading to a documented service degradation case (referred to internally as an "RSO"). Once an account is flagged this way, the customer is rarely able to revert to DSL, and most end up leaving rather than tolerating degraded service.

### Payment Method
The largest gap of any single categorical feature:
| Payment Method | Churn Rate |
|---|---|
| Electronic check | 45.3% |
| Mailed check | 19.1% |
| Bank transfer (automatic) | 16.7% |
| Credit card (automatic) | 15.2% |

> **Domain insight:** Electronic check payments are more prone to silent failures (checks lost or dropped before reaching the company). Tracing a missing payment across both company and bank systems can take two or more weeks, during which the customer's service may be suspended. From the customer's perspective, they already paid — so the resulting confusion and frustration is often directed at the company rather than the actual point of failure, escalating into churn.

### Monthly Charges
Churned customers had a median monthly bill of ~$79 versus ~$64 for retained customers — confirming price sensitivity plays a real role alongside contract flexibility and service type.

---

## ⚙️ Preprocessing
- Dropped `customerID` (non-predictive identifier)
- Label-encoded 13 binary Yes/No columns (explicit `{'No': 0, 'Yes': 1}` mapping for consistency and interpretability)
- One-hot encoded 3 multi-category columns (`InternetService`, `Contract`, `PaymentMethod`) with `drop_first=True` to avoid redundant dummy columns
- Stratified 80/20 train/test split, preserving the original 73.5%/26.5% churn ratio in both sets

---

## ⚖️ Handling Class Imbalance

A naive model that always predicted "no churn" would already score 73.5% accuracy — making plain accuracy a misleading metric on its own. Three separate imbalance-handling strategies were tested and compared:

| Model | Precision (Churn) | Recall (Churn) | F1 (Churn) | Accuracy | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (plain, no imbalance handling) | 0.65 | 0.55 | 0.59 | 80.1% | — |
| Logistic Regression (`class_weight='balanced'`) | 0.51 | **0.79** | 0.62 | 74.4% | 0.839 |
| Random Forest (`class_weight='balanced'`, default params) | 0.65 | 0.51 | 0.57 | 80.0% | — |
| Random Forest (`class_weight='balanced'`, Bayesian-tuned) | 0.50 | **0.80** | 0.62 | 74.0% | 0.839 |
| Logistic Regression + SMOTE (synthetic oversampling) | 0.53 | 0.66 | 0.59 | 76.4% | — |
| XGBoost (`scale_pos_weight`) | 0.55 | 0.68 | 0.61 | 77.0% | 0.821 |

**Key findings:**
- Recall for churners more than doubled between the plain baseline (0.55) and the best-balanced approaches (~0.79-0.80) — at the cost of precision and overall accuracy.
- More complex models (Random Forest, XGBoost) did **not** meaningfully outperform simple Logistic Regression — all four modeling approaches converged to a near-identical ROC-AUC ceiling (~0.82–0.84), suggesting the churn signal in this data is fairly linear and doesn't require complex non-linear modeling to capture.
- SMOTE, despite generating genuine synthetic minority-class examples rather than just reweighting, performed in between the extremes — a reminder that no single imbalance technique is universally best; it depends on the underlying structure of the data.

**Business framing:** for a telecom provider, the cost of a false alarm (an unnecessary retention offer to a customer who wasn't leaving) is small relative to the cost of a missed churner (losing a paying customer's full remaining lifetime value, plus the cost of acquiring a replacement customer). This asymmetry justifies favoring **recall over precision** — the final model was chosen accordingly.

---

## 🏆 Final Model: Logistic Regression (`class_weight='balanced'`)

Chosen over the marginally-tied tuned Random Forest because it achieves **identical practical performance** (same recall, F1, and ROC-AUC) while being fully interpretable via its coefficients — a meaningful advantage for both business communication and portfolio transparency.

### Coefficient Interpretation (Top Drivers)

**Increases churn risk:**
- `PaperlessBilling` (+0.50)
- `PaymentMethod: Electronic check` (+0.41)
- `InternetService: Fiber optic` (+0.36)

**Decreases churn risk:**
- `Contract: Two year` (−0.68) — the single strongest effect in the model
- `TechSupport` (−0.58)
- `OnlineSecurity` (−0.57)

**A note on tenure:** despite showing a strong relationship with churn in EDA, `tenure`'s coefficient in the final model is small (−0.05). This isn't a contradiction — it reflects **multicollinearity** between tenure and Contract type (long-tenure customers are disproportionately on 2-year contracts, and vice versa). Contract type appears to be the more direct driver, with tenure partly acting as a proxy for the same underlying relationship rather than contributing fully independent information.

---

## 📊 Deliverable: Interactive Streamlit App

A standalone app (`app.py`) allows a user to input a hypothetical customer's account details (tenure, contract type, services, billing method, etc.) and receive a live churn risk prediction with probability, using the final trained model.

To run locally:
```bash
streamlit run app.py
```

---

## 💡 Key Business Insights Summary

| # | Insight |
|---|---|
| 1 | Month-to-month contracts churn ~15x more than two-year contracts — the single strongest churn driver |
| 2 | The first ~10 months of a customer relationship is the highest-risk window for churn |
| 3 | Electronic check payments show the highest churn rate of any payment method, likely tied to payment-failure-driven service suspensions rather than customer dissatisfaction alone |
| 4 | Fiber optic customers churn over 2x more than DSL customers, plausibly linked to infrastructure limitations in some service areas |
| 5 | Model complexity did not improve predictive ceiling — Logistic Regression matched Random Forest and XGBoost on ROC-AUC (~0.82–0.84) |
| 6 | Business cost asymmetry (cheap retention offer vs. expensive customer loss) justifies prioritizing recall over precision |
| 7 | TechSupport and OnlineSecurity subscriptions are associated with meaningfully lower churn — potential retention levers worth testing |

---

## 🚀 How to Run
```bash
# Clone the repo
git clone https://github.com/ahmedhussien96/churn-prediction

# Install dependencies
pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn xgboost scikit-optimize streamlit joblib

# Launch Jupyter for the analysis notebook
cd churn-prediction
jupyter lab

# Or launch the interactive app directly
streamlit run app.py
```

---

## 👤 Author
Ahmed Hussien — Data Analysis & ML Roadmap, Project 4 of 10
