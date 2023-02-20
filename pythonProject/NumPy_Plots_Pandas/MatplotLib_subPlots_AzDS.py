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
# Create a figure for 2 subplots (1 row, 2 columns)
fig, ax = plt.subplots(1, 2, figsize = (10,4))

# Create a bar plot of name vs grade on the first axis
ax[0].bar(x=df_students_csv.Name, height=df_students_csv.Grade, color='orange')
ax[0].set_title('Grades')
ax[0].set_xticklabels(df_students_csv.Name, rotation=90)

# Create a pie chart of pass counts on the second axis
pass_counts = df_students_csv['Pass'].value_counts()
ax[1].pie(pass_counts, labels=pass_counts)
ax[1].set_title('Passing Grades')
ax[1].legend(pass_counts.keys().tolist())

# Add a title to the Figure
fig.suptitle('Student Data')

# Show the figure
fig.show()
plt.show()