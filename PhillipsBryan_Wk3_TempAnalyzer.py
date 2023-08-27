"""
Week 3 Assignment - Data Wrangling

Bryan Phillips
08/25/23
DATA 300/6381

Part 3 - Temp Analyzer
"""

# List that contains Fahrenheit temperature data to be analyzed
tempData = [19, 21, 21, 21, 23, 23, 23, 21, 19, 21, 19, 21, 23, 27, 27, 28, 30, 30, 32, 32, 32, 32, 34, 34,
            34, 36, 36, 36, 36, 36, 36, 34, 34, 34, 34, 34, 34, 32, 30, 30, 30, 28, 28, 27, 27, 27, 23, 23,
            21, 21, 21, 19, 19, 19, 18, 18, 21, 27, 28, 30, 32, 34, 36, 37, 37, 37, 39, 39, 39, 39, 39, 39,
            41, 41, 41, 41, 41, 39, 39, 37, 37, 36, 36, 34, 34, 32, 30, 30, 28, 27, 27, 25, 23, 23, 21, 21,
            19, 19, 19, 18, 18, 18, 21, 25, 27, 28, 34, 34, 41, 37, 37, 39, 39, 39, 39, 41, 41, 39, 39, 39,
            39, 39, 41, 39, 39, 39, 37, 36, 34, 32, 28, 28, 27, 25, 25, 25, 23, 23, 23, 23, 21, 21, 21, 21,
            19, 21, 19, 21, 21, 19, 21, 27, 28, 32, 36, 36, 37, 39, 39, 39, 39, 39, 41, 41, 41, 41, 41, 41,
            41, 41, 41, 39, 37, 36, 36, 34, 32, 30, 28, 28, 27, 27, 25, 25, 23, 23, 23, 21, 21, 21, 19, 19,
            19, 19, 19, 19, 21, 23, 23, 23, 25, 27, 30, 36, 37, 37, 39, 39, 41, 41, 41, 39, 39, 41, 43, 43,
            43, 43, 43, 43, 43, 43, 43, 39, 37, 37, 37, 36, 36, 36, 36, 34, 32, 32, 32, 32, 30, 30, 28, 28,
            28, 27, 27, 27, 27, 25, 27, 27, 27, 28, 28, 28, 30, 32, 32, 32, 34, 34, 36, 36, 36, 37, 37, 37,
            37, 37, 37, 37, 37, 37, 36, 34, 30, 30, 27, 27, 25, 25, 23, 21, 21, 21, 21, 19, 19, 19, 19, 19,
            18, 18, 18, 18, 18, 19, 23, 27, 30, 32, 32, 32, 32, 32, 32, 34, 34, 34, 34, 34, 36, 36, 36, 36,
            36, 32, 32, 32, 32, 32, 32, 32, 32, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 28, 28]


def fahrToCelsius(tempFahrenheit):
    # Function fahrToCelsius converts a temperature in degrees Fahrenheit to degrees Celsius.
    # It contains one input parameter called tempFahrenheit and returns the variable convertedTemp back to the user.

    converterTemp = (tempFahrenheit - 32) * 5 / 9
    return converterTemp


def tempClassifier(tempCelsius):
    # Classifies a Celsius temperature on a scale from 0-3 as cold(0), slippery(1), comfortable(2), or warm(3).
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


def main():
    # Empty list to store temperature class information.
    tempClasses = []

    # for loop to iterate through tempData and convert temperatures using fahrtoCelsius function and place converted
    # temps in appropriate classifier using function tempClassifier
    for tempFahr in tempData:
        tempCel = fahrToCelsius(tempFahr)
        tempClass = tempClassifier(tempCel)
        tempClasses.append(tempClass)

    # Output converted temperatures in tuple format.
    for item in tempClasses:
        print(item)

    # Counter dictionary to store how many temperatures belong to which class.
    class_counter = {0: 0, 1: 0, 2: 0, 3: 0}

    # Initiate counter and populate class_counter variable
    for tempClass in tempClasses:
        class_counter[tempClass[0]] += 1

    # Output and display readable formatted string.
    print("\r")  # Carriage return new line
    format_output = (f"There are {class_counter[0]}, 0 degree Celsius, {class_counter[1]} one degree Celsius, "
                     f"{class_counter[2]} two degree Celsius and {class_counter[3]} three degree Celsius.")
    print(format_output)


main()
