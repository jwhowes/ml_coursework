import sys, matplotlib as mpl, numpy as np, statistics
import sklearn.linear_model
import pandas as pd

def gender_to_int(g):
    if g == 'M':
        return 1
    return 0

def education_to_int(e):
    if e == 'Lower Than A Level':
        return 1
    if e == 'A Level or Equivalent':
        return 2
    if e == 'HE Qualification':
        return 3
    if e == 'Post Graduate Qualification':
        return 4

def percentage_range_to_int(p):  # Currently just takes the average of their range
    if type(p) == str:
        p = p[:-1]
        ps = p.split("-")
        return statistics.mean([int(i) for i in ps])
    else:
        return 0

def disability_to_int(d):
    if d == 'Y':
        return 1
    return 0

def result_to_int(r):
    if r == 'Pass':
        return 1
    return 0

students = pd.read_csv("./data/studentInfo.csv")
X = students[['gender', 'highest_education', 'imd_band', 'num_of_prev_attempts', 'studied_credits', 'disability']][:-1]
y = students[['final_result']][:-1]
X['gender'] = X['gender'].map(gender_to_int)
X['highest_education'] = X['highest_education'].map(education_to_int)
X['imd_band'] = X['imd_band'].map(percentage_range_to_int)
X['disability'] = X['disability'].map(disability_to_int)
y['final_result'] = y['final_result'].map(result_to_int)

model = sklearn.linear_model.LinearRegression()
model.fit(X, y)  # Doesn't work and I don't know why
