import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import StratifiedShuffleSplit
import sklearn.pipeline
import sklearn.impute
import sklearn.compose
import sklearn.preprocessing
import sklearn.linear_model
import numpy as np
import pandas as pd

show_corr = False
show_hist = False

def result_to_int(result):
    if result == "Distinction":
        return 3
    elif result == "Pass":
        return 2
    elif result == "Fail":
        return 1
    else:
        return 0

def int_to_pass_fail(num):
    if num == 3 or num == 2:
        return 1
    return 0

students = pd.read_csv("./data/studentInfo.csv")  # Data from studentInfo
registration = pd.read_csv("./data/studentRegistration.csv")
assessment = pd.read_csv("./data/studentAssessment.csv")

students = students.dropna(subset=["imd_band"])

#################################
#### Creating Aggregate data ####
#################################

assessment_agg = assessment.groupby('id_student').agg(
    avg_score=pd.NamedAgg(column='score', aggfunc=np.mean)
)
students = pd.merge(students, assessment_agg, on='id_student')  # Adding average score for each student

registration_agg = registration.drop(['code_presentation', 'date_registration', 'date_unregistration'], axis=1)
registration_agg = registration_agg.groupby('id_student').count()
registration_agg = registration_agg.rename(columns={
    'id_student': 'id_student',
    'code_module': 'num_modules'
})
students = pd.merge(students, registration_agg, on='id_student')  # Adding the number of modules a student is taking

# Histograms (pg 55):
hist = False
if show_hist:
    students.hist(bins=50, figsize=(20, 15))
    plt.show()

# What to do stratified sampling on:
# Correlation matrix (pg 64):
students["final_result"] = students["final_result"].map(result_to_int)

if show_corr:
    corr_matrix = students.corr()
    print(corr_matrix["final_result"].sort_values(ascending=False))  # It's clear that avg_score has the highest absolute correlation (0.47, the next highest is 0.16)

students["avg_score_cat"] = np.ceil(students["avg_score"]/20)
students["avg_score_cat"].where(students["avg_score_cat"] < 5, 5.0, inplace=True)

# There are two options to deal with the 4 columns:
#   1. split the data into 2 groups with subgroups
#       - i.e. have a pass/fail table, a pass/distinction table and a fail/withdrawn table
#   2. Remove withdrawn rows
#       - Still need to split data into pass/fail table and pass/distinction table
#   A third option is to remove withdrawn rows and convert distinctions to simple passes
#       However, I feel this will remove too much information and make the classifier less useful

pass_fail = students.copy()
pass_fail["final_result"] = pass_fail["final_result"].map(int_to_pass_fail)


pass_distinction = students.loc[students["final_result"].isin([3, 2])].copy()
pass_distinction["final_result"] = pass_distinction["final_result"].map(lambda x: x - 2)  # 1: distinction, 0: pass

fail_withdrawn = students.loc[students["final_result"].isin([1, 0])].copy()  # 1: Fail, 0: Withdrawn

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)  # Remove random_state before submitting
for train_index, test_index in split.split(pass_fail, pass_fail["avg_score_cat"]):
    pass_fail_train_set = pass_fail.iloc[train_index]
    pass_fail_test_set = pass_fail.iloc[test_index]

for train_index, test_index in split.split(pass_distinction, pass_distinction["avg_score_cat"]):
    pass_distinction_train_set = pass_distinction.iloc[train_index]
    pass_distinction_test_set = pass_distinction.iloc[test_index]

for train_index, test_index in split.split(fail_withdrawn, fail_withdrawn["avg_score_cat"]):
    fail_withdrawn_train_set = fail_withdrawn.iloc[train_index]
    fail_withdrawn_test_set = fail_withdrawn.iloc[test_index]

for set_ in (pass_fail_train_set, pass_fail_test_set, pass_distinction_train_set, pass_distinction_test_set, fail_withdrawn_train_set, fail_withdrawn_test_set):
    set_.drop("avg_score_cat", axis=1, inplace=True)

pass_fail = pass_fail_train_set.drop("final_result", axis=1)
pass_fail_labels = pass_fail_train_set["final_result"].copy()

pass_distinction = pass_distinction_train_set.drop("final_result", axis=1)
pass_distinction_labels = pass_distinction_train_set["final_result"].copy()

fail_withdrawn = fail_withdrawn_train_set.drop("final_result", axis=1)
fail_withdrawn_labels = fail_withdrawn_train_set["final_result"].copy()

#######################
#### Data Cleaning ####
#######################

num_attribs = ["num_of_prev_attempts", "studied_credits", "num_modules", "avg_score"]
cat_attribs = ["code_module", "code_presentation", "id_student", "gender", "region", "highest_education", "imd_band", "age_band", "disability"]

num_pipeline = sklearn.pipeline.Pipeline([
    ('imputer', sklearn.impute.SimpleImputer(strategy="median")),  # Replace null values with the median
    ('std_scaler', sklearn.preprocessing.StandardScaler())  # Standardises data ranges
])

full_pipeline = sklearn.compose.ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", sklearn.preprocessing.OneHotEncoder(), cat_attribs)
])

pass_fail = full_pipeline.fit_transform(pass_fail)
pass_distinction = full_pipeline.fit_transform(pass_distinction)
fail_withdrawn = full_pipeline.fit_transform(fail_withdrawn)

###############################
#### Training the model(s) ####
###############################

# We will start with the first option of splitting the categories into two groups
# and those groups into two subgroups each

pass_fail_reg = sklearn.linear_model.LinearRegression()  # This is very good on the training set!
pass_fail_reg.fit(pass_fail, pass_fail_labels)

pass_distinction_reg = sklearn.linear_model.LinearRegression()
pass_distinction_reg.fit(pass_distinction, pass_distinction_labels)

fail_withdrawn_reg = sklearn.linear_model.LinearRegression()
fail_withdrawn_reg.fit(fail_withdrawn, fail_withdrawn_labels)

# Trying pass_fail on a 5 instances from the training set:
some_data = pass_fail[:5]
some_labels = pass_fail_labels[:5]
print("Predictions:", pass_fail_reg.predict(some_data))
print("Labels:", list(some_labels))