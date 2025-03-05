#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Machine Learning in Python with scikit-learn
COVID dataset analysis (regression problem)
Author: Irina Chelysheva

Date: March 3rd, 2025
COVID dataset extracted from OurWorldInData (https://github.com/owid/covid-19-data)
"""

# 1. Import necessary libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 2. Set working directory, load, explore and prepare the dataset
os.chdir('/Users/irina/Downloads/ML_DL_intro_python_HT2025-main')
df = pd.read_csv('owid_covid_for_ML_short.csv')
print("Total data points",df.shape[0])
print("Total number of features(as number of columns) are ", df.shape[1])
df.describe()
df.head()

# Check skewness of the data
skewness_before = df.select_dtypes(include=np.number).apply(lambda x: x.skew()).sort_values(ascending=False)
print("Skewness before transformation:")
print(skewness_before)

#Check for null values
null_values = df.isnull().values.any()
if null_values == True:
    print("There are some missing values in data")
else:
    print("There are no missing values in the dataset")
    
# 3. Log-transform numeric features to reduce skewness and stabilize variance
#Note: ignore non-numeric columns
for column in df.select_dtypes(include=np.number).columns:
    df[column] = df[column].apply(lambda x: np.log10(x) if x > 0 else x)
    
# Check skewness after transformation
skewness_after = df.select_dtypes(include=np.number).apply(lambda x: x.skew()).sort_values(ascending=False)
print("\nSkewness after transformation:")
print(skewness_after)

# Define feature matrix X and target variable y
X = df.drop("deaths_per_million", axis=1)  # Feature Matrix
y = df["deaths_per_million"]  # Target Variable

# 4. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Combine X_train and y_train for correlation analysis
all_train = pd.concat([X_train, y_train], axis=1)

# 5. Feature selection using Pearson Correlation
cor = all_train.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.title('Correlation Matrix')
plt.show()

# Select features highly correlated with the target variable
cor_target = abs(cor["deaths_per_million"])
relevant_features = cor_target[cor_target > 0.5].index.tolist()
print(f"Highly correlated features with target: {relevant_features}")

# Check and remove multicollinear features
multicollinearity_pairs = [
    ("median_age", "life_expectancy"), 
    ("life_expectancy", "aged_65_older"),
    ("gdp_per_capita", "hdi"),          
    ("cases_per_million", "cases")
]

# Loop through the pairs and print the correlation
for pair in multicollinearity_pairs:
    print(f"Correlation between {pair[0]} and {pair[1]}:\n", all_train[[pair[0], pair[1]]].corr(), "\n")

# 6. Choosing the best regression model using selected features
# Subset data based on feature selection
X_a = X[["cases_per_million", "cases", "median_age", "gdp_per_capita"]]
Xa_train, Xa_test, ya_train, ya_test = train_test_split(X_a, y, test_size=0.3, random_state=1)

# Define models for evaluation
models = [
    ('LR', LinearRegression()), 
    ('R', Ridge()), 
    ('LASSO', Lasso()), 
    ('EN', ElasticNet()), 
    ('KN', KNeighborsRegressor()), 
    ('DTR', DecisionTreeRegressor(random_state=1)), 
    ('SVR', SVR(gamma='auto')),
    ('RF', RandomForestRegressor(random_state=1))
]

# Evaluate each model using 10-fold cross-validation
seed = 7
scoring = 'neg_mean_squared_error'
results, names = [], []
kfold = KFold(n_splits=10, random_state=seed, shuffle=True)

print("Model Evaluation Results:")
for name, model in models:
    cv_results = cross_val_score(model, Xa_train, ya_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    print(f'{name}: Mean MSE = {cv_results.mean():.4f} (Std = {cv_results.std():.4f})')

# Plot the comparison of model performance
plt.boxplot(results, tick_labels=names)
plt.title('Algorithm Comparison')
plt.ylabel('Negative Mean Squared Error')
plt.show()

# 7. Train the best-performing model and evaluate it (hint: you can also try Ridge on your own)
best_model = LinearRegression()
best_model.fit(Xa_train, ya_train)
predictions = best_model.predict(Xa_test)

# 8. Evaluate the model using various metrics
mse = mean_squared_error(ya_test, predictions)
mae = mean_absolute_error(ya_test, predictions)
r2 = r2_score(ya_test, predictions)

print("\nBest Model Evaluation on Test Set:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R^2 Score: {r2:.4f}")
print(f"Mean of ya_train: {ya_train.mean():.4f}")


# 9. Plot Actual vs. Predicted Values
# Reverse the log10 transformation for plotting
ya_test_exp = 10 ** ya_test
predictions_exp = 10 ** predictions

plt.figure(figsize=(8, 6))
plt.scatter(ya_test_exp, predictions_exp, color='blue', alpha=0.6)
plt.plot([min(ya_test_exp), max(ya_test_exp)], [min(ya_test_exp), max(ya_test_exp)], color='red', linestyle='--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted Values')
plt.grid(True)
plt.show()

#Further exploration

# Feature Importance
feature_importance = pd.Series(best_model.coef_, index=X_a.columns)
feature_importance.sort_values(ascending=False).plot(kind='bar', color='skyblue')
plt.title('Feature Importance')
plt.ylabel('Coefficient Magnitude')
plt.show()
#Note1: cases_per_million is the most significant, while total cases have less influence
#Note2: gdp_per_capita has a negative coefficient (higher GDP per capita - fewer deaths per million)

# Comparison Table
results_df = pd.DataFrame({'Actual': ya_test_exp, 'Predicted': predictions_exp})
results_df['Error'] = results_df['Actual'] - results_df['Predicted']
print(results_df.head(10))  # Display the first 10 results


# Histogram of Prediction Errors
plt.figure(figsize=(8, 6))
plt.hist(results_df['Error'], bins=20, color='orange', edgecolor='black')
plt.title('Distribution of Prediction Errors')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.show()

