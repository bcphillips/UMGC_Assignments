"""
Week 7 Assignment - Exercise Model Building (Linear Regression)

Bryan Phillips
09/26/23
DATA 300/6381

LR2: Dataset: student_scores.csv
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Set Pandas display options
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.float_format', '{:.2f}'.format)


def display_scatterplotraw(data):
    """
    Display a scatter plot showing the relationship between study hours (x-axis) and scores (y-axis).
    Each dot represents a student's performance based on study hours.
    """
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=data, x='Hours', y='Scores', alpha=0.7)
    plt.xlabel('Hours Studied')
    plt.ylabel('Score')
    plt.show()


def display_regression(data, X_pred_values, y_pred_values):
    """
    Scatter plot showcasing original data along with the predicted regression line.
    The proximity of data points to the line helps gauge model accuracy.
    """
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=data, x="Hours", y="Scores", color='blue', alpha=0.7)
    plt.plot(X_pred_values, y_pred_values, color='red', linewidth=2)
    plt.title('Study Hours vs. Scores with Regression Line')
    plt.xlabel('Hours Studied')
    plt.ylabel('Score')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def display_actual_vs_predicted(y_test, y_pred):
    """
    Scatter plot to compare actual scores (x-axis) with model's predicted scores (y-axis).
    Points close to the blue-dashed line indicate accurate predictions.
    """
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


def display_error(y_test, y_pred):
    """
    Displays error in model's score predictions.
    Points above the dashed line indicate overestimation and points below
    indicate underestimation.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values - y_pred, color='green', alpha=0.7)
    plt.xlabel('Student Index')
    plt.ylabel('Prediction Error')
    plt.title('Error in Predicted Scores')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.grid(True)
    plt.show()


def compute_regression_line_values(model, data):
    """
    Calculate regression line points using hours studied as input feature.
    """
    X_pred_values = pd.DataFrame(np.linspace(min(data["Hours"]), max(data["Hours"]), 100),
                                 columns=["Hours"])
    y_pred_values = model.predict(X_pred_values)

    return X_pred_values, y_pred_values


def determine_outliers(data_series):
    Q1 = data_series.quantile(0.25)
    Q3 = data_series.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return any(data_series < lower_bound) or any(data_series > upper_bound)


def filterout_outliers(dataframe, column_name):
    Q1 = dataframe[column_name].quantile(0.25)
    Q3 = dataframe[column_name].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return dataframe[(dataframe[column_name] >= lower_bound) & (dataframe[column_name] <= upper_bound)]


def main():
    # Load data from student_scores.csv
    data = pd.read_csv('student_scores.csv')
    print(data.head())
    print("\r")  # Carriage return new line

    # Check for missing values
    missing_values = data.isnull().sum()
    print("Missing values by columns:")
    print(missing_values)
    print("\r")

    # Check for outliers in 'Hours'
    hours_outliers = determine_outliers(data['Hours'])
    print(f"Outliers in Hours: {'Yes' if hours_outliers else 'No'}")
    print("\r")

    # Filter outliers
    if hours_outliers:
        data = filterout_outliers(data, 'Hours')
        print("Outliers in 'Hours' have been filtered out.")
        print("\r")
    else:
        print("No outliers detected in 'Hours'.")
        print("\r")

    # Display dataset statistics
    print(data.describe())
    print("\r")  # Carriage return new line

    # Display raw data for Hours & Scores by calling display_scatterplotraw (with appropriate adjustments)
    display_scatterplotraw(data)

    # Use 'Hours' as X variable features and 'Scores' as y variable target
    X = data[["Hours"]]
    y = data["Scores"]

    # Split the data for training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

    # Initialize the linear regression model
    scores_model = LinearRegression()

    # Fit the model to the training data
    scores_model.fit(X_train, y_train)

    # Predict the target variable for the testing set
    y_pred = scores_model.predict(X_test)

    # Display the intercept and coefficient
    print("Intercept: ", round(scores_model.intercept_, 2))
    print("Coefficient for Hours: ", round(scores_model.coef_[0], 2))
    print("\r")  # Carriage return new line

    # Display model metrics
    print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test, y_pred), 2))
    print('Mean Squared Error:', round(metrics.mean_squared_error(y_test, y_pred), 2))
    print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)), 2))
    print('R-squared:', round(metrics.r2_score(y_test, y_pred), 2))
    print("\r")  # Carriage return new line

    # Compute regression line data for plotting
    X_pred_values, y_pred_values = compute_regression_line_values(scores_model, data)

    # Display scatterplot of raw data and the linear regression line
    display_regression(data, X_pred_values, y_pred_values)

    # Display scatterplot of actual vs. predicted data with regression line
    display_actual_vs_predicted(y_test, y_pred)

    # Display line graph showing the error
    display_error(y_test, y_pred)


main()
