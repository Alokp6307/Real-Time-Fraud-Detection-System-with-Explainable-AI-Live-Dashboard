# Customer Churn Prediction System

## Overview
The Customer Churn Prediction System is a Machine Learning based project developed to identify telecom customers who are likely to discontinue services. The primary goal of this project is to help businesses reduce customer loss by predicting churn behavior and enabling proactive customer retention strategies.

This project uses customer demographic information, service details, billing records, contract information, and account activity to analyze customer behavior patterns and predict churn probability. By leveraging Machine Learning algorithms, the system can classify customers into churn and non-churn categories with high accuracy.

---

## Problem Statement
Customer churn is one of the major challenges faced by telecom companies. Losing customers directly impacts revenue and business growth. Identifying customers at high risk of churn allows companies to take preventive actions such as targeted offers, customer support, and loyalty programs.

The objective of this project is to:
- Analyze customer behavior
- Identify important churn factors
- Build a predictive Machine Learning model
- Generate business-focused retention insights

---

## Features
- Data Cleaning and Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Machine Learning Model Training
- Churn Risk Prediction
- Customer Segmentation
- Business Insights and Recommendations
- Visualization of Churn Trends
- High-Risk Customer Identification

---

## Technologies Used

### Programming Language
- Python

### Libraries and Frameworks
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

### Machine Learning Algorithms
- Logistic Regression
- Random Forest
- Gradient Boosting

---

## Dataset Information
The project uses the Telco Customer Churn dataset containing:
- Customer demographics
- Contract details
- Billing information
- Internet and phone services
- Monthly and total charges
- Customer tenure
- Churn status

After preprocessing and cleaning, the final dataset contained 7,032 customer records.

---

## Data Preprocessing
The following preprocessing steps were performed:
- Handling missing values
- Removing blank records
- Encoding categorical variables
- Feature scaling
- Data transformation
- Splitting training and testing datasets

---

## Exploratory Data Analysis
EDA was performed to identify the major factors affecting customer churn. Key observations include:
- Customers with month-to-month contracts showed the highest churn rate.
- Short tenure customers were more likely to churn.
- Customers with higher monthly charges had higher churn probability.

Various charts and visualizations were created to understand customer behavior and churn distribution.

---

## Model Building
Multiple Machine Learning models were implemented and evaluated to identify the best-performing algorithm.

Models Used:
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

The Gradient Boosting model achieved the best performance with a tuned ROC-AUC score of 0.842.

---

## Results
- Best Model: Gradient Boosting
- ROC-AUC Score: 0.842
- Overall Churn Rate: 26.6%
- High-Risk Customers Successfully Identified

The model effectively ranked customers based on churn risk and supported retention prioritization.

---

## Business Recommendations
Based on the analysis, the following recommendations were generated:
- Focus retention campaigns on month-to-month customers
- Improve onboarding experience for new customers
- Provide loyalty benefits for high-risk customers
- Offer long-term contract incentives
- Monitor customer churn probability regularly

---

## Future Improvements
- Deploy the project using Streamlit or Flask
- Integrate real-time customer data
- Add deep learning models
- Improve prediction accuracy using advanced feature engineering
- Create automated retention dashboards

---

## Conclusion
This project demonstrates the practical implementation of Machine Learning for solving real-world business problems. The Customer Churn Prediction System helps telecom companies identify at-risk customers and improve retention strategies through predictive analytics and data-driven decision making.

---

## Author
Alok Pandey  
B.Tech CSE (Data Science)  
Noida Institute of Engineering and Technology
