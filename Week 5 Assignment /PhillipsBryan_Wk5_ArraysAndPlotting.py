"""
Week 5 Assignment - NumPy Arrays and matplotlib Plotting

Bryan Phillips
09/12/23
DATA 300/6381

Part A - looping_arrays.py
Part B - matplotlib Plotting
"""

# Imports
# ----------------------------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set Pandas Display Options
# ----------------------------------------------------------------------------------------------------------------------
pd.set_option('display.max_columns', None)  # Display all columns
pd.set_option('display.max_rows', None)  # Display all rows
pd.set_option('display.width', None)  # Width of the display in characters
pd.set_option('display.max_colwidth', None)  # Display the full content of each column

# Part A: Looping through 3D Array
# ----------------------------------------------------------------------------------------------------------------------
arr3 = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
print(arr3)
shape = arr3.shape

for ii in range(shape[0]):
    for jj in range(shape[1]):
        for kk in range(shape[2]):
            print(arr3[ii, jj, kk])

# Part B1: Comparison of GDP for China & India
# ----------------------------------------------------------------------------------------------------------------------
data = pd.read_csv("gapminder_gdp_asia.csv", index_col="country")
years = data.columns.str.strip("gdpPercap_")
gdp_china = data.loc["China"]
gdp_india = data.loc["India"]

# Plotting
plt.plot(years, gdp_china, "r--", label="China")
plt.plot(years, gdp_india, "g-", label="India")
plt.title("Comparison of GDP for China & India: 1952-2007")
plt.legend(loc="lower right")
plt.xlabel("Year")
plt.ylabel("GDP per capita")
plt.show()

# Part B2: Correlation Between GDP & Life Expectancy for 2007
# ----------------------------------------------------------------------------------------------------------------------
data_all = pd.read_csv("gapminder_all.csv", index_col="country")
data_all.plot(kind='scatter', x='gdpPercap_2007', y='lifeExp_2007', s=data_all['pop_2007']/1e6)
plt.title("Correlation Between GDP & Life Expectancy for 2007")
plt.show()
