import sys, matplotlib as mpl, numpy as np
import sklearn.linear_model
import pandas as pd

students = pd.read_csv("./data/studentInfo.csv")
X = students[['gender', 'region', 'highest_education', 'imd_band', 'age_band', 'num_of_prev_attempts', 'studied_credits', 'disability']][:-1]
y = students[['final_result']][:-1]

model = sklearn.linear_model.LinearRegression()
model.fit(X, y)  # Doesn't work because they aren't all numbers (gender is a string etc.)
