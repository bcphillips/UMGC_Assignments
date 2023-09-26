"""
Week 7 Assignment - Exercise Model Building (Linear Regression)

Bryan Phillips
09/26/23
DATA 300/6381

LR1: Dataset: kc_house_data.csv
LR2: Dataset: student_scores.csv.
"""

# Import libraries for linear regression analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Pandas display options for optimal output
# pd.set_option('display.max_columns', None)  # Display all columns
# pd.set_option('display.max_rows', None)  # Display all rows
pd.set_option('display.width', None)  # Width of the display in characters
pd.set_option('display.max_colwidth', None)  # Display the full content of each column
pd.set_option('display.float_format', '{:.2f}'.format)  # Change exponential number display

# LR1
# ----------------------------------------------------------------------------------------------------------------------
# Prepare the dataset
data = pd.read_csv('kc_house_data.csv')
print(data.head())
print("\r")  # Carriage return new line

# Display dataset statistics
print(data.describe())
print("\r")  # Carriage return new line
print(data.info())
print("\r")  # Carriage return new line

# Use 'sqft_living' as X variable features and 'price' as y variable target. These will be used to build our linear
# regression.
"""
A linear regression is a way to predict and define an unknown value (i.e., the price of a house) based on a related
known value (i.e., square footage/size of a house). Using the data one can plot a straight line that represents the
relationship between these values and predict where future data may land. 
"""
X = data[['sqft_living']]
y = data['price']
print("Type of X:", type(X))
print("Type of y:", type(y))
print("\r")  # Carriage return new line

# Split the data for training and testing sets.
"""
This section of the code is where the predictive model starts to take shape. The process of train_test_split involves 
dividing the data into two parts: one to "train" or teach our model, and the other to "test" how well the model's 
predictions align with actual outcomes. This step is crucial because it helps one understand how accurate the model is 
and whether any adjustments are needed.

By using the train_test_split function from the Python library scikit-learn, one can easily achieve this by inputting  
X (square footage/size of a house) and y (price) into the function to get the training and testing datasets.
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

# Initialize the linear regression model
kc_house_model = LinearRegression()

# Fit the model to the training data
kc_house_model.fit(X_train, y_train)

# Predict the target variable for the testing set
y_pred = kc_house_model.predict(X_test)

# Model Results: Display the intercept and coefficient
print("Intercept: ", kc_house_model.intercept_)
print("Coefficient for sqft_living: ", kc_house_model.coef_)
print("\r")  # Carriage return new line

# Display model metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("\r")  # Carriage return new line

# Create a scatterplot of raw data for X('sqft_living') vs. y('price)
plt.figure(figsize=(15, 9))
plt.scatter(X, y, alpha=0.5, facecolors='none', edgecolors='blue', s=20)
plt.title('Price vs. Sqft Living')
plt.xlabel('Sqft Living')
plt.ylabel('Price')
plt.grid(True)

# Adjust y-axis limits and labels
plt.ylim(0, 8000000)
yticks = [0, 1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000]
yticklabels = ['0', '$1M', '$2M', '$3M', '$4M', '$5M', '$6M', '$7M', '$8M']
plt.yticks(yticks, yticklabels)

plt.tight_layout()
plt.show()
