"""
Week 4 Assignment - Data Exploration & Wrangling with Pandas

Bryan Phillips
08/25/23
DATA 300/6381
"""

# Import Pandas and set display options to view all data without truncation
import pandas as pd
pd.set_option('display.max_columns', None)  # Display all columns
pd.set_option('display.max_rows', None)  # Display all rows
pd.set_option('display.width', None)  # Width of the display in characters
pd.set_option('display.max_colwidth', None)  # Display the full content of each column


# Part 1
# ----------------------------------------------------------------------------------------------------------------------
# Create variable for new DataFrame containing data from 6153237444115dat.csv file
# Assign attributes to na_values
data = pd.read_csv("6153237444115dat.csv", na_values=['*', '**', '***', '****', '*****', '******'])

# Display the total number of rows in the DataFrame
print("How many rows are there in the data?: ", len(data))

# Create a list of column names
column_names = data.columns.tolist()

# Display the column names separated by commas
print("\r")  # Carriage return new line
print("What are the column names?:", ", ".join(column_names))

# Display the datatype for each column
print("\r")  # Carriage return new line
print("What are the datatypes of the columns?:")
print(data.dtypes)

# Display mean Fahrenheit temperature of column TEMP
print("\r")  # Carriage return new line
mean_ColumnTemp = data["TEMP"].mean()
print("What is the mean Fahrenheit temperature in the data? (TEMP Column): ", mean_ColumnTemp)

# Display standard deviation of column MAX
print("\r")  # Carriage return new line
sd_ColumnMax = data["MAX"].std()
print("What is the standard deviation of the Maximum temperature? (MAX Column): ", sd_ColumnMax)

# Display unique station IDs of column USAF
print("\r")  # Carriage return new line
uni_ColumnUSAF = len(data["USAF"].unique())
print("How many unique stations exists in the data? (USAF Column): ", uni_ColumnUSAF)


# Part 2
# ----------------------------------------------------------------------------------------------------------------------
# Create variable using data from only columns USAF, YR--MODAHRMN, TEMP, MAX & MIN
selected = data[["USAF", "YR--MODAHRMN", "TEMP", "MAX", "MIN"]]

# Use dropna() function to clean selected columns of N/A data and create nondestructive copy
clean_SelectedColumns = selected.dropna(subset=["TEMP"]).copy()

# Create new column Celsius to store converted temperatures from TEMP column using Fahrenheit to Celsius formula
clean_SelectedColumns["Celsius"] = (clean_SelectedColumns["TEMP"] - 32) * 5/9

# Round the conversions to one decimal place without creating a new column
clean_SelectedColumns["Celsius"] = clean_SelectedColumns["Celsius"].round(1)


# Part 3
# ----------------------------------------------------------------------------------------------------------------------
# Create variable to store Kumpula weather data using USAF code 29980
kumpula = clean_SelectedColumns[clean_SelectedColumns["USAF"] == 29980]

# Create variable to store Rovaniemi weather data using USAF code 28450
rovaniemi = clean_SelectedColumns[clean_SelectedColumns["USAF"] == 28450]

# Save Kumpula data to new .csv file separated with commas, and one decimal point floating point numbers
kumpula.to_csv("Kumpula_temps_May_Aug_2017.csv", index=False, float_format="%.1f", sep=",")

# Save Rovaniemi data to new .csv file separated with commas, and one decimal point floating point numbers
rovaniemi.to_csv("Rovaniemi_temps_May_Aug_2017.csv", index=False, float_format="%.1f", sep=",")


# Part 4a
# ----------------------------------------------------------------------------------------------------------------------
# Create variable for new DataFrame containing data from Kumpula_temps_May_Aug_2017.csv file
kumpula_data = pd.read_csv("Kumpula_temps_May_Aug_2017.csv")

# Create variable for new DataFrame containing data from Rovaniemi_temps_May_Aug_2017.csv file
rovaniemi_data = pd.read_csv("Rovaniemi_temps_May_Aug_2017.csv")

# Display the median temperature for Helsinki Kumpula in Celsius
print("\r")  # Carriage return new line
print("Median temperature in Helsinki Kumpula: ", kumpula_data["Celsius"].median())

# Display the median temperature for Rovaniemi in Celsius
print("\r")  # Carriage return new line
print("Median temperature in Rovaniemi: ", rovaniemi_data["Celsius"].median())


# Part 4b
# ----------------------------------------------------------------------------------------------------------------------
# Create variable to store Helsinki Kumpula & Rovaniemi May temperatures
kumpula_may = kumpula_data.loc[(kumpula_data
                                ["YR--MODAHRMN"] >= 201705010000) & (kumpula_data["YR--MODAHRMN"] < 201706010000)]

rovaniemi_may = rovaniemi_data.loc[(rovaniemi_data
                                    ["YR--MODAHRMN"] >= 201705010000) & (rovaniemi_data["YR--MODAHRMN"] < 201706010000)]


# Create variable to store Helsinki Kumpula & Rovaniemi June temperatures
kumpula_june = kumpula_data.loc[(kumpula_data
                                 ["YR--MODAHRMN"] >= 201706010000) & (kumpula_data["YR--MODAHRMN"] < 201707010000)]

rovaniemi_june = rovaniemi_data.loc[(rovaniemi_data
                                    ["YR--MODAHRMN"] >= 201706010000) & (rovaniemi_data["YR--MODAHRMN"] < 201707010000)]

# Create variable for statistical analysis of Helsinki Kumpula mean, min and max temperatures for May & June
kumpula_may_mean = kumpula_may["Celsius"].mean().round(1)

# Test output
print("\r")  # Carriage return new line
print(f"Helsinki Kumpula mean temperature for the month of may is {kumpula_may_mean} degrees Celsius.")
