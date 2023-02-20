import pandas as pd
from matplotlib import pyplot as plt
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler

'''
Refer - FindingSlopeusingScikit_AzDS.py for coefficient values - m and b
'''


def f(x):
    m = 6.3134
    b = -17.9164
    return m * x + b


study_time = 14

# Get f(x) for study time
prediction = f(study_time)

# Grade can't be less than 0 or more than 100
expected_grade = max(0, min(100, prediction))

# Print the estimated grade
print('Studying for {} hours per week may result in a grade of {:.0f}'.format(study_time, expected_grade))
