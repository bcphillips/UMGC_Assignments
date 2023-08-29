"""
Week 2 Assignment - Data Types

Bryan Phillips
08/25/23
DATA 300/6381
"""

# Initialized list with mixed data types
list3 = list(('hello', 1, 'you', 6.3, 'yes', 7, 2.3))

# For loop that iterates through list3

for i in range(len(list3)):

    # if statement to check if the list element is type str
    if str(type(list3[i])) == "<class 'str'>":
        list3[i] = list3[i] + " of course"

    # elif statement to check if the list element is type float
    elif str(type(list3[i])) == "<class 'float'>":
        # Add 5 to the original values of float in list3
        list3[i] = list3[i] + 5

# Output list3 and modified elements in list3
print('list3 = ', list3)
