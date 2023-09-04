"""
Week 3 Assignment - Data Wrangling

Bryan Phillips
09/04/23
DATA 300/6381

Part 1 - Simple Temperature Calculator
Part 2 - Temperature Classifier
"""


# Part #1


def fahrToCelsius(tempFahrenheit):
    # Function fahrToCelsius converts a temperature in degrees Fahrenheit to degrees Celsius.
    # It contains one input parameter called tempFahrenheit and returns the variable convertedTemp back to the user.

    convertedTemp = (tempFahrenheit - 32) * 5 / 9
    return convertedTemp


# Test Case: Update number between parenthesis in order to convert Fahrenheit temperature to degrees Celsius
print("32 degrees Fahrenheit in Celsius is: ", fahrToCelsius(32))


# Part #2

def tempClassifier(tempCelsius):
    # Classifies a Celsius temperature on a scale from 0-3 as cold(0), slippery(1), comfortable(2), or warm(3).
    # The function contains one input parameter tempCelsius
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
