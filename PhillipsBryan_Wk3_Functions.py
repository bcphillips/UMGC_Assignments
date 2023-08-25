"""
Week 3 Assignment - Data Wrangling

Bryan Phillips
08/25/23
DATA 300/6381




"""


def fahrToCelsius(tempFahrenheit):
    converterTemp = (tempFahrenheit - 32) * 5 / 9
    return converterTemp


# Test Case
print("32 degrees Fahrenheit in Celsius is: ", fahrToCelsius(32))
