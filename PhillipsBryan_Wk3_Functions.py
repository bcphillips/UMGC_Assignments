"""
Week 3 Assignment - Data Wrangling

Bryan Phillips
08/25/23
DATA 300/6381




"""


# Function fahrToCelsius converts a temperature in Fahrenheit to degrees Celsius.
# It contains one input parameter called tempFahrenheit and returns the variable convertedTemp back to the user.
def fahrToCelsius(tempFahrenheit):
    converterTemp = (tempFahrenheit - 32) * 5 / 9
    return converterTemp


# Test Case - Update number between parenthesis in order to convert Fahrenheit temperature to degrees Celsius
print("32 degrees Fahrenheit in Celsius is: ", fahrToCelsius(32))
