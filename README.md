# Fraud Detection Using Machine Learning Algorithms

# Introduction
In this project, I explore the application of machine learning techniques to the problem of credit card fraud detection. Given the rapid increase in online transactions, fraud detection has become more important than ever, and machine learning offers powerful tools to tackle this challenge.

# Data Description
The dataset used in this project is the Credit Card Fraud Detection dataset from Kaggle. It contains transactions made by credit cards in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions. The data contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, the original features and more background information about the data are not provided. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. The feature 'Class' is the target variable and it takes value 1 in case of fraud and 0 otherwise.

# Methodology
I first performed exploratory data analysis to understand the distribution of classes, correlation among different features, and so on. Then I preprocessed the data by standardizing the 'Amount' and 'Time' features and split our data into training and test sets. More info can found in Data_Explorer.ipynb file in this repository. 

I trained two different models for the task of fraud detection: a logistic regression model and a random forest model. Given the class imbalance in our data, we used the Synthetic Minority Over-sampling Technique (SMOTE) when training our random forest model to balance out our classes.

# Results
Both models achieved high accuracy on the test set. However, considering the extreme class imbalance in the dataset, we also looked at the precision, recall, and F1 scores for the minority class (fraudulent transactions).

The logistic regression model had a precision of 0.86 and a recall of 0.58. This means that when this model predicted a transaction was fraudulent, it was correct 86% of the time, and it correctly identified 58% of all fraudulent transactions.

The random forest model with SMOTE had a precision of 0.89 and a recall of 0.85. This means that when this model predicted a transaction was fraudulent, it was correct 89% of the time, and it correctly identified 85% of all fraudulent transactions.

In the future, I plan to test more sophisticated techniques for dealing with class imbalance and compare more types of models.

