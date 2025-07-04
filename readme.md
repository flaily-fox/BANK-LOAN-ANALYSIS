# 🏦 Bank Loan Prediction using PCA & Naive Bayes in Python

This project predicts loan eligibility by applying Principal Component Analysis (PCA) for feature reduction and Naive Bayes classification for modeling. Built for real-time risk analysis, it helps banks target eligible customers and reduce lending risks using customer data from online applications.

## Objectives
- Predict loan approval based on customer details.
- Automate decision-making using machine learning.
- Reduce manual loan verification workload.
- Enhance prediction efficiency using PCA.

## Dataset Description
Includes 100+ customer records with:
- Gender  
- Marital Status  
- Age  
- Occupation  
- Income  
- Debts  
- Loan Type  
- Loan Amount  
- Loan Decision Status

Each entry represents a unique customer’s application history.

## Methodology
1. **Data Preprocessing**
   - Missing value handling
   - Label encoding & normalization
   - PCA to reduce dimensions

2. **Model Building**
   - Naive Bayes classifier with scikit-learn
   - Train/test split with 70–30 ratio
   - Cross-validation for performance tuning

3. **Evaluation**
   - Accuracy Score
   - Confusion Matrix
   - Precision, Recall, F1 Score

## Insights
- Automated loan eligibility screening
-  Reduced manual approval time
-  Improved targeting for approved loans

## 🛠 Requirements

# Libraries Required

- Pandas
- numpy
- scikit-learn
- matplotlib
