"""
Week 2 Assignment - Data Types

Bryan Phillips
08/22/23
DATA 300/6381

NOTE: you will be graded on the following:

Exercise on Python Data Types:

1. Read, study and execute PhillipsBryan_Wk2Assignment_DataTypes.py; you can copy the entire code from the file or from
the Word document or parts of it into code blocks in a Jupyter workbook and execute it there. You can also copy the code
into a Python IDE, like Spyder (from anaconda) and execute it there.
2. Modify the following code, such that if the elements are of type float, then add 5 to the previous value:
This code loops through all items of list3 and if they are of type str, then it appends 'of course'
3. Notice that when we indent statements we should hit tab
4. Before you modify the code below try to guess what your output will be...
5. Add at least two comments describing functioning of the code below

"""

# Original/unmodified script
# ----------------------------------------------------------------------------------------------------------------------

# list3 = list(('hello', 1, 'you', 6.3, 'yes', 7, 2.3))
#
# for i in range(len(list3)):
#     if str(type(list3[i])) == "<class 'str'>":
#         list3[i] = list3[i] + " of course"
# print('list3=', list3)

# Modified/updated script
# ----------------------------------------------------------------------------------------------------------------------

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
