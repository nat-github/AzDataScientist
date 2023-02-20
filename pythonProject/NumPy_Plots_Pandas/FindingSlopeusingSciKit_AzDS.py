import pandas as pd
from matplotlib import pyplot as plt
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler

df_students_csv = pd.read_csv('grades.csv', delimiter=',', header='infer')
df_students_csv.head()
# dropping the values which has NaN
df_students_csv = df_students_csv.dropna(axis=0, how='any')
# Get students who studied for the mean or more hours
passes = pd.Series(df_students_csv['Grade'] >= 60)
df_students_csv = pd.concat([df_students_csv, passes.rename("Pass")], axis=1)
print(df_students_csv.describe())
# print("Result", df_students_csv)
df_sample = df_students_csv[df_students_csv['StudyHours'] > 1]

df_regression = df_sample[['Grade', 'StudyHours']].copy()
# Get the regression slope and intercept
m, b, r, p, se = stats.linregress(df_regression['StudyHours'], df_regression['Grade'])
print('slope: {:.4f}\ny-intercept: {:.4f}'.format(m, b))
print('so...\n f(x) = {:.4f}x + {:.4f}'.format(m, b))
# Use the function (mx + b) to calculate f(x) for each x (StudyHours) value
df_regression['fx'] = (m * df_regression['StudyHours']) + b

# Calculate the error between f(x) and the actual y (Grade) value
df_regression['error'] = df_regression['fx'] - df_regression['Grade']

# Show the original x,y values, the f(x) value, and the error
print(df_regression[['StudyHours', 'Grade', 'fx', 'error']])

# Create a scatter plot of Grade vs StudyHours
df_regression.plot.scatter(x='StudyHours', y='Grade')

# Plot the regression line
plt.plot(df_regression['StudyHours'], df_regression['fx'], color='cyan')

# Display the plot
plt.show()

