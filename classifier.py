import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import StratifiedShuffleSplit
import sklearn.pipeline
import sklearn.impute
import sklearn.compose
import sklearn.preprocessing
import numpy as np
import pandas as pd

show_corr = False
show_hist = False

def result_to_int(result):
    if result == "Distinction":
        return 2
    elif result == "Pass":
        return 1
    else:
        return 0

students = pd.read_csv("./data/studentInfo.csv")  # Data from studentInfo
registration = pd.read_csv("./data/studentRegistration.csv")
assessment = pd.read_csv("./data/studentAssessment.csv")

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

students["avg_score_cat"] = np.ceil(students["avg_score"]/10)
students["avg_score_cat"].where(students["avg_score_cat"] < 8, 8.0, inplace=True)
#students["avg_score_cat"].hist()
#plt.show()

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)  # Remove random_state before submitting
for train_index, test_index in split.split(students, students["avg_score_cat"]):
    strat_train_set = students.loc[train_index]
    strat_test_set = students.loc[test_index]

for set_ in (strat_train_set, strat_test_set):
    set_.drop("avg_score_cat", axis=1, inplace=True)

students = strat_train_set.drop("final_result", axis=1)
students_labels = strat_train_set["final_result"].copy()

#######################
#### Data Cleaning ####
#######################

students_num = students.drop(["code_module", "code_presentation", "id_student", "gender", "region", "highest_education", "imd_band", "age_band", "disability"], axis=1)
students_cat = students.drop(["num_of_prev_attempts", "studied_credits", "num_modules", "avg_score", "imd_band"], axis=1)  # For some reason this doesn't work with imd_band

num_attribs = list(students_num)
cat_attribs = list(students_cat)

num_pipeline = sklearn.pipeline.Pipeline([
    ('imputer', sklearn.impute.SimpleImputer(strategy="median")),  # Replace null values with the median
    ('std_scaler', sklearn.preprocessing.StandardScaler())  # Standardises data ranges
])

full_pipeline = sklearn.compose.ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", sklearn.preprocessing.OneHotEncoder(), cat_attribs)
])

students_prepared = full_pipeline.fit_transform(students)
print(students_prepared)