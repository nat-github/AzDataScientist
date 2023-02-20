import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

'''
    Range: The difference between the maximum and minimum. There's no built-in function for this, 
    but it's easy to calculate using the min and max functions.
    Variance: The average of the squared difference from the mean. You can use the built-in var function to find this.
    Standard Deviation: The square root of the variance. You can use the built-in std function to find this.
'''

df_students_csv = pd.read_csv('grades.csv', delimiter=',', header='infer')
df_students_csv.head()
# dropping the values which has NaN
df_students_csv = df_students_csv.dropna(axis=0, how='any')
# Get students who studied for the mean or more hours
passes = pd.Series(df_students_csv['Grade'] >= 60)
df_students_csv = pd.concat([df_students_csv, passes.rename("Pass")], axis=1)
#print("Result", df_students_csv)
for col_name in ['Grade', 'StudyHours']:
    col = df_students_csv[col_name]
    rng = col.max() - col.min()
    var = col.var()
    std = col.std()
    print('\n{}:\n - Range: {:.2f}\n - Variance: {:.2f}\n - Std.Dev: {:.2f}'.format(col_name, rng, var, std))
