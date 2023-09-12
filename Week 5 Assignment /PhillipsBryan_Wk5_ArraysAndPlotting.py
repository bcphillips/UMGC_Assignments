"""
Week 5 Assignment - NumPy Arrays and matplotlib Plotting

Bryan Phillips
09/12/23
DATA 300/6381

Part A - looping_arrays.py
Part B - matplotlib Plotting
"""

# Part A
# ----------------------------------------------------------------------------------------------------------------------
# Import Numpy library
import numpy as np

# Initialize 3D array with each array containing three elements
arr3 = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

# Display the 3D array
print(arr3)

# Display the shape of the 3D array
print(np.shape(arr3))

# Three nested loops to access all the elements of the 3D array
for ii in range(np.shape(arr3)[0]):
    for jj in range(np.shape(arr3)[1]):
        for kk in range(np.shape(arr3)[2]):
            print(arr3[ii, jj, kk])  # Display all the elements


# Part B
# ----------------------------------------------------------------------------------------------------------------------
# Import Pandas and matplotlib libraries
import pandas as pd
import matplotlib.pyplot as plt

# Import Pandas and set display options to view all data without truncation
pd.set_option('display.max_columns', None)  # Display all columns
pd.set_option('display.max_rows', None)  # Display all rows
pd.set_option('display.width', None)  # Width of the display in characters
pd.set_option('display.max_colwidth', None)  # Display the full content of each column

# Create DataFrame from .csv file and use country as index
data = pd.read_csv("gapminder_gdp_asia.csv", index_col="country")

# Create variable to store "gdpPercap_xxxx" year data and extract only year data as string
years = data.columns.str.strip("gdpPercap_")

# Select GDP data for China & India.
gdp_china = data.loc["China"]
gdp_india = data.loc["India"]

# Plot data with differently colored lines
plt.plot(years, gdp_china, "r--", label="China")
plt.plot(years, gdp_india, "g-", label="India")

# Create title, legend, x-axis & y-axis label
plt.title("Comparison of GDP for China & India: 1952-2007")
plt.legend(loc="lower right")  # Show legend in the lower right-hand corner of the plot
plt.xlabel("Year")
plt.ylabel("GDP per capita")

# Display line plot
plt.show()

# Part B2
# ----------------------------------------------------------------------------------------------------------------------
# # Create DataFrame from .csv file and use country as index
data_all = pd.read_csv("gapminder_all.csv", index_col="country")

# Creat scatter plot that show the correlation between GDP and life expectancy for 2007
data_all.plot(kind='scatter', x='gdpPercap_2007', y='lifeExp_2007',
              s=data_all['pop_2007']/1e6)

# Create title for graph/plot and display scatter plot
plt.title("Correlation Between GDP & Life Expectancy for 2007")

# Display the scatter plot
plt.show()
