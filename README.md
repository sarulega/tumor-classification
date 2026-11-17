# Tumor Diagnostic Support System

A web dashboard that classifies tumors as benign or malignant using probabilistic machine learning (Naive Bayes), with patient management, probability analysis and PDF reports.

**Live app:** (link will be added after deployment)

## Features
1. Home dashboard with system statistics
2. Patient registration
3. Patient management (search and delete)
4. Tumor measurement input (30 features)
5. Benign or malignant classification
6. Probability analysis with decision bands
7. Model comparison (Naive Bayes, Logistic Regression, KNN)
8. Data visualisation
9. Prediction history
10. Downloadable PDF reports
11. Dataset analysis
12. Admin dashboard

## Dataset
Wisconsin Diagnostic Breast Cancer (569 patients, 30 features), loaded from scikit-learn.

## How it works
Naive Bayes applies Bayes' theorem with a Gaussian curve per feature and class. Cases with a malignant probability between 10% and 90% are flagged as uncertain and referred for specialist review.

## Run locally
pip install -r requirements.txt
streamlit run app.py

## Disclaimer
Educational project. Not for real clinical use or diagnosis.

## Author
Sarulega
