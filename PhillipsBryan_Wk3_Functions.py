"""
Week 3 Assignment - Data Wrangling

Bryan Phillips
08/25/23
DATA 300/6381

Part 1 - Simple Temperature Calculator
Part 2 - Temperature Classifier
"""


# Part #1


def fahrToCelsius(tempFahrenheit):
    # Function fahrToCelsius converts a temperature in degrees Fahrenheit to degrees Celsius.
    # It contains one input parameter called tempFahrenheit and returns the variable convertedTemp back to the user.

    converterTemp = (tempFahrenheit - 32) * 5 / 9
    return converterTemp


# Test Case: Update number between parenthesis in order to convert Fahrenheit temperature to degrees Celsius
print("32 degrees Fahrenheit in Celsius is: ", fahrToCelsius(32))


# Part #2


def tempClassifier(tempCelsius):
    # Function tempClassifier uses the function fahrToCelsius and classifies that Celsius temperature on a scale from
    # 0-3 as cold, slippery, comfortable or warm.
    # The function contains one input parameter tempeCelsius
    # and returns a tuple based on the classified temperature.

    if tempCelsius < -2:
        return 0, "cold"
    elif -2 <= tempCelsius < 2:
        return 1, "slippery"
    elif 2 <= tempCelsius < 15:
        return 2, "comfortable"
    else:
        return 3, "warm"


classify_temperature = fahrToCelsius(32)

print("\r")  # Carriage return new line
print(tempClassifier(classify_temperature))
