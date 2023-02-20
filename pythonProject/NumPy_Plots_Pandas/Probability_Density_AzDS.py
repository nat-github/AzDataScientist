import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

df_students_csv = pd.read_csv('grades.csv', delimiter=',', header='infer')
df_students_csv.head()
# dropping the values which has NaN
df_students_csv = df_students_csv.dropna(axis=0, how='any')
# Get students who studied for the mean or more hours
passes = pd.Series(df_students_csv['Grade'] >= 60)
df_students_csv = pd.concat([df_students_csv, passes.rename("Pass")], axis=1)
# print("Result", df_students_csv)
# Get the variable to examine
var_data = df_students_csv['Grade']
# Create a Figure
fig = plt.figure(figsize=(10, 4))

# Plot a density
var_data.plot.density()

# Add titles and labels
plt.title('Data Density')
# Show the mean, median, and mode
plt.axvline(x=var_data.mean(), color='cyan', linestyle='dashed', linewidth=2)
plt.axvline(x=var_data.median(), color='red', linestyle='dashed', linewidth=2)
plt.axvline(x=var_data.mode()[0], color='yellow', linestyle='dashed', linewidth=2)
# Show the figure
plt.show()
