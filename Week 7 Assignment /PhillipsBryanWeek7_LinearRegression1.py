"""
Week 7 Assignment - Exercise Model Building (Linear Regression)

Bryan Phillips
09/26/23
DATA 300/6381

LR1: Dataset: kc_house_data.csv
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
    """
    This functions displays the raw data and shows the relationship between the house prices (y-axis) and square
    footage area (x-axis).
    Each dot represents a different house found in the original dataset.
    Viewing the raw data allows one to see a direct correlation between house size and price
    """
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=data, x='sqft_living', y='price', alpha=0.7)
    plt.xlabel('Square Foot Living Area')
    plt.ylabel('House Price')
    plt.title('House Price vs. Square Foot Living Area')
    adjust_yticks()
    plt.show()


def display_regression(data, X_pred_values, y_pred_values):
    """
    This scatterplot shows the original data along with the linear regression line in red.
    The graph shows one how well the model supports the data.
    Datapoints closer to the red line indicate our model is a good fit for the dataset.
    """
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=data, x="sqft_living", y="price", color='blue', alpha=0.7)
    plt.plot(X_pred_values, y_pred_values, color='red', linewidth=2)
    plt.title('Raw Data with Regression Line: Price vs. Square Footage Living Area')
    plt.xlabel('Square Footage Living Area')
    plt.ylabel('Price')
    adjust_yticks()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def display_actual_vs_predicted(y_test, y_pred):
    """
    This function allows one to see and compare the actual house prices on the x-axis with the prices the train model
    predicted on the y-axis.
    Each point is a different house and one can see how the points are to blue dashed line.
    If the points follow the line, it means the prediction was accurate.
    Deviation from the line shows that our prediction model differs from the reality of the actual data.
    """
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=y_test, y=y_pred, color='r', alpha=0.7)
    plt.xlabel('Actual House Prices')
    plt.ylabel('Predicted House Prices')
    plt.title('Actual vs. Predicted House Prices')
    adjust_yticks()
    plt.grid(True)
    limits = [min(min(y_test), min(y_pred)), max(max(y_test), max(y_pred))]
    plt.plot(limits, limits, color='blue', linestyle='--', linewidth=2)
    plt.tight_layout()
    plt.show()


def display_error_plot(y_test, y_pred):
    """
    This graph will show one where the predictions of the model were too high (over the dashed line) or too low (below
    the dashed line) when compared to the actual house prices.
    The x-axis shows the houses while the y-axis shows the difference in predictions.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values - y_pred, color='green', alpha=0.7)
    plt.xlabel('Index')
    plt.ylabel('Error')
    plt.title('Error in Predicted House Prices')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.grid(True)
    plt.show()


def compute_regression_line_values(model, data):
    """
    This function calculates the starting and end point of a line to represent the house price predictions on a graph.
    It uses the house sizes for the x-axis and uses the trained model to predict the corresponding house prices for
    those house sizes.
    """
    X_pred_values = pd.DataFrame(np.linspace(min(data["sqft_living"]), max(data["sqft_living"]), 100),
                                 columns=["sqft_living"])
    y_pred_values = model.predict(X_pred_values)

    return X_pred_values, y_pred_values


def adjust_yticks():
    """
    This function changes the y-axis to millions of dollars for the above plots.
    """
    plt.ylim(0, 8000000)
    yticks = [0, 1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000]
    yticklabels = ['0', '$1M', '$2M', '$3M', '$4M', '$5M', '$6M', '$7M', '$8M']
    plt.yticks(yticks, yticklabels)


def main():
    # LR1 (kc_house_data.csv)
    # ------------------------------------------------------------------------------------------------------------------
    # Prepare the dataset
    data = pd.read_csv('kc_house_data.csv')
    print(data.head())
    print("\r")  # Carriage return new line

    # Display dataset statistics
    print(data.describe())
    print("\r")  # Carriage return new line

    # Display raw data for sqft_living & price by calling display_scatterplot
    display_scatterplotraw(data)

    # ------------------------------------------------------------------------------------------------------------------
    # Use 'sqft_living' as X variable features and 'price' as y variable target.
    # These will be used to build our linear
    # regression.
    """
    A linear regression is a way to predict and define an unknown value (i.e., the price of a house) based on a related
    known value (i.e., square footage/size of a house). Using the data one can plot a straight line that represents the
    relationship between these values and predict where future data may land. 
    """
    X = data[["sqft_living"]]
    y = data["price"]

    # Split the data for training and testing sets.
    """This section of the code is where the predictive model starts to take shape. The process of train_test_split 
    involves dividing the data into two parts: one to "train" or teach our model, and the other to "test" how well 
    the model's predictions align with actual outcomes. This step is crucial because it helps one understand how 
    accurate the model is and whether any adjustments are needed.
    
    By using the train_test_split function from the Python library scikit-learn, one can easily achieve this by 
    inputting X (square footage/size of a house) and y (price) into the function to get the training and testing 
    datasets."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

    # Initialize the linear regression model
    kc_house_model = LinearRegression()

    # Fit the model to the training data
    kc_house_model.fit(X_train, y_train)

    # Predict the target variable for the testing set
    y_pred = kc_house_model.predict(X_test)

    # Model Results: Display the intercept and coefficient
    """
    After training a linear regression one is provided with the intercept and coefficient for the input feature. The
    intercept shows how the predicted value did (house price) when the input feature (square footage) is set to zero.
    """
    print("Intercept: ", round(kc_house_model.intercept_, 2))
    print("Coefficient for sqft_living: ", round(kc_house_model.coef_[0], 2))
    print("\r")  # Carriage return new line

    # Display model metrics
    """
    Model metrics allow one to evaluate the performance of the linear regression model. 
    
    - Mean Absolute Error (MAE): Provides details for how far off our predictive model is using the average. Lower 
    numbers indicate that our model is working as it should. 
    
    - Mean Squared Error (MSE): Similar to MAE but one is squaring the differences.
    
    - Root Mean Squared Error (RMSE): This represents the average of the errors in the predictive model. It is a way of
    indicating how far off one can be before the model is not working. 
    
    - R-squared: This lets one know how well a predictive model shows trends in a dataset. A higher R2 means the model
    is doing well at predicting, while a lower score means that could be other factors influencing house prices. 
    """
    print('Mean Absolute Error:', round(metrics.mean_absolute_error(y_test, y_pred), 2))
    print('Mean Squared Error:', round(metrics.mean_squared_error(y_test, y_pred), 2))
    print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(y_test, y_pred)), 2))
    print('R-squared:', round(metrics.r2_score(y_test, y_pred), 2))
    print("\r")  # Carriage return new line

    # ------------------------------------------------------------------------------------------------------------------
    # Compute regression line data for plotting
    X_pred_values, y_pred_values = compute_regression_line_values(kc_house_model, data)

    # Display scatterplot of raw data and the linear regression line
    display_regression(data, X_pred_values, y_pred_values)

    # Display scatterplot of actual vs. predicted data with regression line
    display_actual_vs_predicted(y_test, y_pred)

    # Display line graph showing the error
    display_error_plot(y_test, y_pred)


main()
