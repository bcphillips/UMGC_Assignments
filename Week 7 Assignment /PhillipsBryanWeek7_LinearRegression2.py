"""
Week 7 Assignment - Exercise Model Building (Linear Regression)

Bryan Phillips
09/26/23
DATA 300/6381

LR2: Dataset: student_scores.csv
"""

# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Pandas display options
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.float_format', '{:.2f}'.format)


def display_scatterplotraw(data):
    """Displays scatterplot for raw data."""
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=data, x='Hours', y='Scores', alpha=0.7)
    plt.xlabel('Hours of Study')
    plt.ylabel('Scores')
    plt.title('Scores vs. Hours of Study')
    plt.grid(True)
    plt.show()


def display_regression(data, X_pred_values, y_pred_values):
    """Displays the original data along with the linear regression line."""
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=data, x='Hours', y='Scores', color='blue', alpha=0.7)
    plt.plot(X_pred_values, y_pred_values, color='red', linewidth=2)
    plt.title('Raw Data with Regression Line: Scores vs. Hours of Study')
    plt.xlabel('Hours of Study')
    plt.ylabel('Scores')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def display_actual_vs_predicted(y_test, y_pred):
    """Displays scatterplot comparing the actual scores to the predicted scores."""
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=y_test, y=y_pred, color='r', alpha=0.7)
    plt.xlabel('Actual Scores')
    plt.ylabel('Predicted Scores')
    plt.title('Actual vs. Predicted Scores')
    plt.grid(True)
    limits = [min(min(y_test), min(y_pred)), max(max(y_test), max(y_pred))]
    plt.plot(limits, limits, color='blue', linestyle='--', linewidth=2)
    plt.tight_layout()
    plt.show()


def display_error_plot(y_test, y_pred):
    """
    Display the errors in predictions.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values - y_pred, color='green', alpha=0.7)
    plt.xlabel('Index')
    plt.ylabel('Error')
    plt.title('Error in Predicted Scores')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.grid(True)
    plt.show()


def compute_regression_line_values(model, data):
    """Calculates regression line values for plotting."""
    X_pred_values = pd.DataFrame(np.linspace(min(data["Hours"]), max(data["Hours"]), 100), columns=["Hours"])
    y_pred_values = model.predict(X_pred_values)
    return X_pred_values, y_pred_values


def main():
    # Prepare the dataset
    data = pd.read_csv('student_scores.csv')
    print(data.head())

    # Display dataset statistics
    print(data.describe())

    # Display raw data scatterplot
    display_scatterplotraw(data)

    # Split data into features and target variable
    X = data[["Hours"]]
    y = data["Scores"]

    # Split the data for training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

    # Train linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict scores for the testing set
    y_pred = model.predict(X_test)

    # Model Results: Display the intercept and coefficient
    print("Intercept: ", round(model.intercept_, 2))
    print("Coefficient for Hours of Study: ", round(model.coef_[0], 2))

    # Display model metrics
    print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test, y_pred), 2))
    print('Mean Squared Error:', round(metrics.mean_squared_error(y_test, y_pred), 2))
    print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)), 2))
    print('R-squared:', round(metrics.r2_score(y_test, y_pred), 2))

    # Compute regression line data for plotting
    X_pred_values, y_pred_values = compute_regression_line_values(model, data)

    # Display plots
    display_regression(data, X_pred_values, y_pred_values)
    display_actual_vs_predicted(y_test, y_pred)
    display_error_plot(y_test, y_pred)


main()
