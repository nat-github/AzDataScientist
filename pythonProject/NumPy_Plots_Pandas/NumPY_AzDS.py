import numpy as np

data = [50, 50, 47, 97, 49, 3, 53, 42, 26, 74, 82, 62, 37, 15, 70, 27, 36, 35, 48, 52, 63, 64];
study_hours = [10.0, 11.5, 9.0, 16.0, 9.25, 1.0, 11.5, 9.0, 8.5, 14.5, 15.5,
               13.75, 9.0, 8.0, 15.5, 8.0, 9.0, 6.0, 10.0, 12.0, 12.5, 12.0];
grades = np.array(data);
# print(grades)
print(type(data), 'x 2:', data * 2);
print('---');
print(type(grades), 'x 2:', grades * 2);
print("Dimesnsions of Arrays", grades.shape);
# Mean - Grades
print('Mean', grades.mean());
# Create a 2D array (an array of arrays)
student_data = np.array([study_hours, grades]);
print('Specific elements in grades', grades[4]);
print('specific element in student data', student_data[0][1]);
avg_grade = student_data[0].mean();
avg_hours = student_data[1].mean();
print('Average study hours: {:.2f}\nAverage grade: {:.2f}'.format(avg_grade, avg_hours));