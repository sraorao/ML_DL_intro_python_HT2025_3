#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Machine Learning in Python with scikit-learn
Breast Cancer dataset analysis (classification problem)
Author: Irina Chelysheva

Date: March 10th, 2025
Breast Cancer Wisconsin (Diagnostic) Data Set extracted from UCI Machine Learning Repository
(https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
"""

# 1. Import necessary libraries and setup working
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score

import os
print("Current Working Directory:", os.getcwd())
os.chdir('ML_DL_intro_python_HT2025_3-main')

# 2. Load, explore and prepare the dataset
url = "https://raw.githubusercontent.com/selva86/datasets/master/BreastCancer.csv"
df = pd.read_csv(url)
print("Total data points",df.shape[0])
print("Total number of features(as number of columns) are ", df.shape[1])
df.describe()
df.head()
#Column Class contains the target variable - 0 (benign) and 1 (malignant)

# Check skewness of the data
skewness_before = df.select_dtypes(include=np.number).apply(lambda x: x.skew()).sort_values(ascending=False)
print("Skewness before transformation:")
print(skewness_before) #We don't include ID as it is categorical, so the rest is not too skewed

# Define feature matrix X and target variable y
X = df.drop(['Id', 'Class'], axis=1)  # Feature matrix without 'Id' and 'Class'
y = df['Class']  # Target variable
class_counts = df['Class'].value_counts() 
print(class_counts) #We need to handle imbalanced classes

#Check for null values
null_values = df.isnull().values.any()
if null_values == True:
    print("There are some missing values in data")
else:
    print("There are no missing values in the dataset")
    
# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# 3. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.3, random_state=1, stratify=y)

# 4. Optional example of feature selection using SelectKBest with chi-square
select_k_best = SelectKBest(f_classif, k='all')  # We will just use 'all' to select all features
X_train_kbest = select_k_best.fit_transform(X_train, y_train)
X_test_kbest = select_k_best.transform(X_test)
print("Selected features using SelectKBest:", X.columns[select_k_best.get_support(indices=True)])

# Optional example of Recursive Feature Elimination (RFE) using Logistic Regression
rfe = RFE(estimator=LogisticRegression(max_iter=500), n_features_to_select=9)
X_train_rfe = rfe.fit_transform(X_train, y_train)
X_test_rfe = rfe.transform(X_test)
print("Features selected by RFE:", X.columns[rfe.get_support(indices=True)])

# 5. Choosing the best classification model using selected features
# Subset data based on feature selection (using SelectKBest features)
X_a_train, X_a_test = X_train_kbest, X_test_kbest

# Define models for evaluation
models = [
    ('LR', LogisticRegression(max_iter=500)), 
    ('KNN', KNeighborsClassifier()), 
    ('DTC', DecisionTreeClassifier(random_state=1)), 
    ('SVM', SVC(gamma='auto', probability=True)),
    ('RF', RandomForestClassifier(random_state=1))
]

# Evaluate each model using 10-fold cross-validation
seed = 7
scoring = 'accuracy'
results, names = [], []
kfold = KFold(n_splits=10, random_state=seed, shuffle=True)

print("\nModel Evaluation Results:")
for name, model in models:
    cv_results = cross_val_score(model, X_a_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    print(f'{name}: Mean Accuracy = {cv_results.mean():.4f} (Std = {cv_results.std():.4f})')

# Plot the comparison of model performance
plt.boxplot(results, tick_labels=names)
plt.title('Algorithm Comparison')
plt.ylabel('Accuracy Score')
plt.show()

# 6. Train the best-performing model and evaluate it
best_model = LogisticRegression(random_state=1)
best_model.fit(X_a_train, y_train)  # Train on the training data
predictions = best_model.predict(X_a_test)  # Make predictions on the test data

# 7. Evaluate the model using various metrics
print("\nBest Model Evaluation on Test Set:")
print(f"Accuracy Score: {accuracy_score(y_test, predictions):.4f}")
print("\nClassification Report:\n", classification_report(y_test, predictions))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# 8. ROC Curve
# Obtain predicted probabilities for the positive class
y_pred_proba = best_model.predict_proba(X_a_test)[:, 1]  # Probability of the positive class
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Plot the ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# Feature Importance using coefficients
feature_importance = pd.Series(best_model.coef_[0], index=X.columns[select_k_best.get_support(indices=True)])
feature_importance.sort_values(ascending=False).plot(kind='bar', color='skyblue')
plt.title('Feature Importance')
plt.ylabel('Coefficient Value')
plt.show()
#Note: try it yourself - choose only important features using the feature selection above
#Rerun the predictions again - how did the reduction in number of features affected predictions?

# 9. Export the trained model
import joblib
joblib.dump(best_model, 'best_model.joblib')
print("Model exported as 'best_model.joblib'")
