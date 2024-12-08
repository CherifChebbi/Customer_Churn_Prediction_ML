# Customer Churn Prediction using Machine Learning

## Overview
Customer churn prediction is crucial for businesses, especially in the telecom industry, as it helps identify customers who are likely to stop using services. By accurately predicting churn, companies can proactively implement retention strategies to reduce customer attrition and increase customer satisfaction.

This project focuses on building a machine learning model to predict whether a customer will churn based on various features such as demographic details, subscription services, and billing information.

## Project Highlights
- **Dataset**: The dataset used for this project contains customer information from a telecom company, including demographic details, service subscription types, and financial information.
- **Objective**: Predict whether a customer will churn (leave the company) based on their behavior and characteristics.
- **Techniques**:
  - Data preprocessing and cleaning
  - Feature engineering and encoding categorical variables
  - Imbalanced class handling with SMOTE (Synthetic Minority Oversampling Technique)
  - Model training with Decision Tree, Random Forest, and XGBoost classifiers
  - Model evaluation using accuracy, confusion matrix, and classification report
  - Streamlit web app for interactive customer churn prediction

## Dataset Columns
- **gender**: Customer gender (Male/Female)
- **SeniorCitizen**: Whether the customer is a senior citizen (0/1)
- **Partner**: Whether the customer has a partner (Yes/No)
- **Dependents**: Whether the customer has dependents (Yes/No)
- **tenure**: Number of months the customer has stayed with the company
- **PhoneService**: Whether the customer has phone service (Yes/No)
- **InternetService**: Type of internet service (DSL/Fiber optic/No)
- **Contract**: Type of contract (Month-to-month/One year/Two year)
- **PaperlessBilling**: Whether the customer has paperless billing (Yes/No)
- **PaymentMethod**: Method of payment (Electronic check/Mailed check/Bank transfer/Credit card)
- **MonthlyCharges**: Monthly charges for the customer
- **TotalCharges**: Total charges incurred by the customer

## Project Workflow
1. **Data Loading and Understanding**: Loading the dataset, checking its structure, and identifying missing or irrelevant data.
2. **Data Cleaning**: Handling missing values, encoding categorical variables, and preprocessing features.
3. **Exploratory Data Analysis (EDA)**: Analyzing the distribution of numerical and categorical variables, and visualizing class imbalance in the target variable.
4. **Modeling**: Training machine learning models (Decision Tree, Random Forest, XGBoost) on the processed data.
5. **Evaluation**: Assessing the model's performance using cross-validation and evaluating it with metrics like accuracy, confusion matrix, and classification report.
6. **Deployment**: Deploying the model using a Streamlit app, allowing users to input customer details and receive churn predictions in real-time.

## Technologies Used
- Python
- Pandas, NumPy, Scikit-learn, XGBoost
- Streamlit for creating the web app
- Matplotlib, Seaborn for data visualization
- SMOTE for handling class imbalance

## Usage
1. **Predicting Churn**: The Streamlit app allows users to input customer data and receive a churn prediction (Yes/No) with a probability score.
2. **Interactive Web Interface**: The app is easy to use, with a sidebar for entering customer details and a button to make predictions.
