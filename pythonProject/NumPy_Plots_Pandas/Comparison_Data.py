import pandas as pd
from matplotlib import pyplot as plt
import scipy.stats as stats

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
print(df_sample)
#df_sample.boxplot(column='StudyHours', by='Pass', figsize=(8, 5))
df_sample.plot(x='Name', y=['Grade','StudyHours'], kind='bar', figsize=(8, 5))
plt.show()

