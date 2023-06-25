import os
import pandas as pd
import numpy as np
import random

# List of subjects
subjects = ['语文', '数学', '英语', '物理', '化学', '生物', '历史']

# List of grades
grades = ['高一', '高二', '高三']

# List of ranks
ranks = ['甲', '乙', '丙', '丁']

# Empty dataframe
df = pd.DataFrame(columns = ['年级'] + subjects + ['综合', '等级'])

# Generate data randomly
num_students = 5000

for i in range(num_students):
    grade = random.choice(grades)
    subject_scores = [random.randint(50, 100) for i in range(len(subjects))]
    # Calculate the median score
    median_score = np.median(subject_scores).astype(int)
    # Calculate the rank
    if median_score >= 90:
        rank = '甲'
    elif median_score >= 85:
        rank = '乙'
    elif median_score >= 60:
        rank = '丙'
    else:
        rank = '丁'

    # Add the data to the dataframe
    df.loc[i] = [grade] + subject_scores + [median_score, rank]

# Save the dataframe to a csv file
print(df.head())
df.to_csv('data.csv', index = False, encoding = 'utf-8')






