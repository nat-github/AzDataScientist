import numpy as np
import pandas as pd

data = [50, 40, 47, 97, 49, 3, 53, 42, 26, 74, 82, 62, 37, 15, 70, 27, 36, 35, 48, 52, 63, 64];
study_hours = [10.0, 11.5, 9.0, 16.0, 9.25, 1.0, 11.5, 9.0, 8.5, 14.5, 15.5,
               13.75, 9.0, 8.0, 15.5, 8.0, 9.0, 6.0, 10.0, 12.0, 12.5, 12.0];
grades = np.array(data);
# Create a 2D array (an array of arrays)
student_data = np.array([study_hours, grades]);
df_students = pd.DataFrame({'Name': ['Dan', 'Joann', 'Pedro', 'Rosie', 'Ethan', 'Vicky', 'Frederic', 'Jimmie',
                                     'Rhonda', 'Giovanni', 'Francesca', 'Rajab', 'Naiyana', 'Kian', 'Jenny',
                                     'Jakeem', 'Helena', 'Ismat', 'Anila', 'Skye', 'Daniel', 'Aisha'],
                            'StudyHours': student_data[0],
                            'Grade': student_data[1]})

print(df_students.loc[5])
# from 0 to 5 - loc
print('Using loc', df_students.loc[0:5])
# using iloc
print('Using iloc', df_students.iloc[0:5])
print(df_students.iloc[0, [1, 2]])
print(df_students.loc[0, 'Grade'])
print(df_students.loc[df_students['Name'] == 'Aisha']);
print(df_students[df_students['Name'] == 'Aisha'])
print(df_students.query('Name=="Aisha"'))
print(df_students[df_students.Name == 'Aisha'])
df_students_csv = pd.read_csv('grades.csv', delimiter=',', header='infer');
df_students_csv.head()
# Print rows which are null in dataframe
print("Data with null", df_students_csv.isnull())
print("Data with sum of null\n", df_students_csv.isnull().sum())
print(df_students_csv[df_students_csv.isnull().any(axis=1)])
# Fill the null vales with average hours
df_students_csv.StudyHours = df_students_csv.StudyHours.fillna(df_students_csv.StudyHours.mean())
print("Filling out with default values for null", df_students_csv)
# dropping the values which has NaN
df_students_csv = df_students_csv.dropna(axis=0, how='any')
print("Dropping values with null\n", df_students_csv)
mean_study = df_students_csv['StudyHours'].mean();
mean_grade = df_students_csv.Grade.mean()
# Get students who studied for the mean or more hours
print("Get students who studied for the mean or more hours\n", df_students_csv[df_students_csv.StudyHours > mean_study])
passes = pd.Series(df_students_csv['Grade'] >= 60)
df_students_csv = pd.concat([df_students, passes.rename("Pass")], axis=1)
print("Passes series\n", df_students_csv)
print("group By", df_students_csv.groupby(df_students_csv.Pass).Name.count())
print("Group By aggregating multiple fields",
      df_students_csv.groupby(df_students_csv.Pass)['StudyHours', 'Grade'].mean())
df_students_csv = df_students_csv.sort_values('Grade', ascending=False)
print("Sorted values", df_students_csv)