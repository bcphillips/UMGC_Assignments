"""
Week 4 Assignment - Data Exploration & Wrangling with Pandas

Bryan Phillips
08/25/23
DATA 300/6381
"""

import pandas as pd

pd.set_option('display.max_columns', None)  # Display all columns
pd.set_option('display.max_rows', None)  # Display all rows
# pd.set_option('display.width', None)  # Width of the display in characters
# pd.set_option('display.max_colwidth', None)  # Display the full content of each column
# pd.set_option('display.float_format', '{:.2f}'.format)  # Change exponential number display

dataFrame = pd.read_csv("6153237444115dat.csv", na_values=['*', '**', '***', '****', '*****', '******']
                        )
# Display the total number of rows in the DataFrame
print("How many rows are there in the data?: ", len(dataFrame))

# Create a list of column names
column_names = dataFrame.columns.tolist()

# Print the column names separated by commas
print("\r")  # Carriage return new line
print("What are the column names?:", ", ".join(column_names))

# Display mean Fahrenheit temperature of column TEMP
print("\r")  # Carriage return new line
mean_ColumnTemp = dataFrame["TEMP"].mean()
print("What is the mean Fahrenheit temperature in the data? (TEMP Column): ", mean_ColumnTemp)

# Display standard deviation of column MAX
print("\r")  # Carriage return new line
sd_ColumnMax = dataFrame["MAX"].std()
print("What is the standard deviation of the Maximum temperature? (MAX Column): ", sd_ColumnMax)

# Display unique station IDs of column USAF
print("\r")  # Carriage return new line
uni_ColumnUSAF = len(dataFrame["USAF"].unique())
print("How many unique stations exists in the data? (USAF Column): ", uni_ColumnUSAF)
