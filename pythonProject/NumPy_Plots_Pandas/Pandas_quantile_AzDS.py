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
print("Result", df_students_csv)
# Get the variable to examine
# calculate the 0.01th percentile
q01 = df_students_csv.StudyHours.quantile(0.01)
col = df_students_csv[df_students_csv.StudyHours>q01]
print(col)
var_data = df_students_csv['Grade']


def distribution(var_data):
    min_val = var_data.min()
    max_val = var_data.max()
    mean_val = var_data.mean()
    med_val = var_data.median()
    mod_val = var_data.mode()[0]

    print('Minimum:{:.2f}\nMean:{:.2f}\nMedian:{:.2f}\nMode:{:.2f}\nMaximum:{:.2f}\n'.format(min_val,
                                                                                             mean_val,
                                                                                             med_val,
                                                                                             mod_val,
                                                                                             max_val))

    # Create a Figure
    fig = plt.figure(figsize=(10, 4))

    # Plot a histogram
    plt.hist(var_data)

    # Plot a Box chart
    # plt.box(var_data)

    # Add lines for the statistics
    plt.axvline(x=min_val, color='gray', linestyle='dashed', linewidth=2)
    plt.axvline(x=mean_val, color='cyan', linestyle='dashed', linewidth=2)
    plt.axvline(x=med_val, color='red', linestyle='dashed', linewidth=2)
    plt.axvline(x=mod_val, color='yellow', linestyle='dashed', linewidth=2)
    plt.axvline(x=max_val, color='gray', linestyle='dashed', linewidth=2)

    # Add titles and labels
    plt.title('Data Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # Show the figure
    fig.show()
    plt.show()


distribution(var_data)
