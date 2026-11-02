# AI-Based Customer Spending Prediction

A beginner-friendly machine learning project that predicts whether a customer belongs to the **Low**, **Medium** or **High** spending category using demographic and purchasing-behaviour data.

> Built as part of the **Veda Technology – AI & ML Internship**.

---

## Project Overview

Businesses want to understand which customers spend the most so they can plan offers, loyalty programs and marketing. This project builds an end-to-end ML workflow that classifies customers into three spending groups and compares two models.

## Objectives
- Load, clean and understand a customer dataset
- Explore patterns with 5 visualizations (EDA)
- Preprocess numerical and categorical features with a scikit-learn pipeline
- Train and compare **Logistic Regression** and **Random Forest**
- Evaluate with Accuracy, Precision, Recall, F1-score and a confusion matrix
- Find the most important features
- Predict the spending category of a new customer

## Tech Stack
Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · Jupyter Notebook / Google Colab

## Dataset
- File: `customer_spending_dataset.csv`
- 1,200 synthetic (generated) customer records, created for learning purposes only. It does not represent real customers.
- **Target:** `spending_category` (Low / Medium / High)
- **Features:** age, gender, annual_income, purchase_frequency, avg_order_value, total_orders, recency_days, discount_usage, category_diversity, online_orders_ratio, membership

## Results

| Model | Accuracy | Weighted F1-score |
|---|---|---|
| **Random Forest** | **0.783** | **0.785** |
| Logistic Regression | 0.750 | 0.754 |

- Random Forest performed best and was used for the sample prediction.
- The **Low** class was predicted best (F1 0.86). The **Medium** class was hardest (F1 0.69), as it sits between the other two.
- **Top features:** `avg_order_value`, `purchase_frequency`, `total_orders`, `annual_income`.
- Sample new customer (29 yrs, Gold member, 10 purchases/yr, avg order 110) → predicted **High** (probability 0.78).

The notebook includes all charts: category distribution, age distribution, order value by category, income vs order value, membership vs category, model comparison, confusion matrix and feature importance.

## Project Structure
```text
AI_Based_Customer_Spending_Prediction_Project/
├── AI_Based_Customer_Spending_Prediction_Project.ipynb
├── customer_spending_dataset.csv
├── requirements.txt
└── README.md
```

## How to Run

**Google Colab**
1. Open the `.ipynb` file in Google Colab.
2. Upload `customer_spending_dataset.csv` when asked (or via the Files panel).
3. Click **Runtime → Run all**.

**Locally**
```bash
pip install -r requirements.txt
jupyter notebook
```
Open the notebook and run all cells (keep the CSV in the same folder).

## Key Learnings
- End-to-end ML workflow: cleaning → EDA → preprocessing → modelling → evaluation
- Handling numerical and categorical data with `ColumnTransformer` pipelines
- Comparing models using weighted F1-score and a confusion matrix
- Interpreting a model with feature importance

## Business Insights
- Order value and purchase frequency are the strongest signals of high spenders.
- Predictions can help segment customers and plan targeted offers.

## Disclaimer
This is an educational prototype trained on synthetic data. It should not be used for real financial or business decisions without validation on real-world data.
