import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

df_students_csv = pd.read_csv('grades.csv', delimiter=',', header='infer');
df_students_csv.head()
# dropping the values which has NaN
df_students_csv = df_students_csv.dropna(axis=0, how='any')
# Get students who studied for the mean or more hours
passes = pd.Series(df_students_csv['Grade'] >= 60)
df_students_csv = pd.concat([df_students_csv, passes.rename("Pass")], axis=1)
print("Result", df_students_csv)
# Create a Figure
fig = plt.figure(figsize=(8, 3))
# Create a bar plot of name vs grade
plt.bar(x=df_students_csv.Name, height=df_students_csv.Grade, color='orange')
# Customize the chart
plt.title('Student Grades')
plt.xlabel('Student')
plt.ylabel('Grade')
plt.grid(color='#95a5a6', linestyle='--', linewidth=2, axis='y', alpha=0.7)
plt.xticks(rotation=90)

# Display the plot
plt.show()
# Display the plot
plt.show()
