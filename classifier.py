import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import StratifiedShuffleSplit
import sklearn.pipeline
import sklearn.impute
import sklearn.compose
import sklearn.preprocessing
import sklearn.linear_model
import sklearn.ensemble
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import numpy as np
import pandas as pd

#########################################################################################
#### How to run:                                                                     ####
####    - Ensure matplotlib, sklearn, numpy, and pandas are installed and up-to-date ####
####    - Run via the commmand line or through other means                           ####
####        - e.g. python classifier.py                                              ####
####    - All data files should be stored as .csv in a subfolder called 'data'       ####
#########################################################################################

show_corr = False
show_hist = False

def result_to_int(result):
    if result == "Distinction":
        return 2
    elif result == "Pass":
        return 1
    elif result == "Fail":
        return 0

def int_to_pass_fail(num):  # If we're including withdrawn rows
    if num == 2 or num == 1:
        return 1
    return 0

def plot_roc_curve(X, y, classifier):
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y, classifier.predict(X))
    plt.figure(figsize=(4,4))
    plt.xlabel("FPR", fontsize=14)
    plt.ylabel("TPR", fontsize=14)
    plt.title("ROC Curve", fontsize=14)
    plt.plot(fpr, tpr)
    plt.show()

def show_metrics(X, y, classifier, multiclass=False):
    if multiclass:
        print("Precision:", precision_score(y, classifier.predict(X), average="macro"))
        print("Recall:", recall_score(y, classifier.predict(X), average="macro"))
    else:
        print("Precision:", precision_score(y, classifier.predict(X)))
        print("Recall:", recall_score(y, classifier.predict(X)))
    print("Accuracy:", accuracy_score(y, classifier.predict(X)))
    if multiclass:
        print("F1:", f1_score(y, classifier.predict(X), average="macro"))
    else:
    	print("F1:", f1_score(y, classifier.predict(X)))

def show_corr_matrix(data, label):
    corr_matrix = data.corr()
    print(corr_matrix[label].sort_values(ascending=False))

def plot_scatter_matrices(data, attributes):
    pd.plotting.scatter_matrix(students[attributes], figsize=(12,8))
    plt.show()

def plot_logistic_classifier(X_unprepared, X_prepared, x, labels, log_classifier):
    plt.figure(figsize=(4,4))
    plt.xlabel(x, fontsize=14)
    plt.ylabel("final_result", fontsize=14)
    plt.scatter(X_unprepared[x], labels)
    plt.scatter(X_unprepared[x], log_classifier.predict_proba(X_prepared)[:,1], s=0.1)
    plt.show()

students = pd.read_csv("./data/studentInfo.csv")  # Data from studentInfo
registration = pd.read_csv("./data/studentRegistration.csv")
assessment = pd.read_csv("./data/studentAssessment.csv")

students = students.dropna(subset=["imd_band"])  # Drop students with null imd_band attribute

students = students[students["final_result"] != "Withdrawn"]  # The correlations are much higher when we remove withdrawn values

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
    show_corr_matrix(students, "final_result")
    plot_scatter_matrices(students, ["num_of_prev_attempts", "studied_credits", "num_modules", "avg_score", "final_result"])

students["avg_score_cat"] = np.ceil(students["avg_score"]/20)
students["avg_score_cat"].where(students["avg_score_cat"] < 5, 5.0, inplace=True)

############################
#### Stratifed Sampling ####
############################
pass_fail = students.copy()
pass_fail["final_result"] = pass_fail["final_result"].map(int_to_pass_fail)

pass_distinction = students.loc[students["final_result"].isin([2, 1])].copy()
pass_distinction["final_result"] = pass_distinction["final_result"].map(lambda x: x - 1)  # 1: distinction, 0: pass

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)  # Remove random_state before submitting
for train_index, test_index in split.split(students, students["avg_score_cat"]):
    students_train_set = students.iloc[train_index]
    students_test_set = students.iloc[test_index]

for train_index, test_index in split.split(pass_fail, pass_fail["avg_score_cat"]):
    pass_fail_train_set = pass_fail.iloc[train_index]
    pass_fail_test_set = pass_fail.iloc[test_index]

for train_index, test_index in split.split(pass_distinction, pass_distinction["avg_score_cat"]):
    pass_distinction_train_set = pass_distinction.iloc[train_index]
    pass_distinction_test_set = pass_distinction.iloc[test_index]

for set_ in (students_train_set, students_test_set, pass_fail_train_set, pass_fail_test_set, pass_distinction_train_set, pass_distinction_test_set):
    set_.drop("avg_score_cat", axis=1, inplace=True)
    set_.drop("id_student", axis=1, inplace=True)

students = students_train_set.drop("final_result", axis=1)
students_labels = students_train_set["final_result"].copy()

pass_fail = pass_fail_train_set.drop("final_result", axis=1)
pass_fail_labels = pass_fail_train_set["final_result"].copy()

pass_distinction = pass_distinction_train_set.drop("final_result", axis=1)
pass_distinction_labels = pass_distinction_train_set["final_result"].copy()

#######################
#### Data Cleaning ####
#######################
num_attribs = ["num_of_prev_attempts", "studied_credits", "num_modules", "avg_score"]
cat_attribs = ["code_module", "code_presentation", "gender", "region", "highest_education", "imd_band", "age_band", "disability"]

num_pipeline = sklearn.pipeline.Pipeline([
    ('imputer', sklearn.impute.SimpleImputer(strategy="median")),  # Replace null values with the median
    ('std_scaler', sklearn.preprocessing.StandardScaler())  # Standardises data ranges
])

full_pipeline = sklearn.compose.ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", sklearn.preprocessing.OneHotEncoder(), cat_attribs)
])

students_prepared = full_pipeline.fit_transform(students)
pass_fail_prepared = full_pipeline.transform(pass_fail)
pass_distinction_prepared = full_pipeline.transform(pass_distinction)

###############################
#### Training the model(s) ####
###############################
# We will start with the first option of splitting the categories into two groups
# and those groups into two subgroups each

students_forest = sklearn.ensemble.RandomForestClassifier(random_state=42)
students_forest.fit(students_prepared, students_labels)

#pass_fail_log = sklearn.linear_model.LogisticRegressionCV()  # This is very good on the training set!
#pass_fail_log.fit(pass_fail_prepared, pass_fail_labels)

pass_distinction_log = sklearn.linear_model.LogisticRegressionCV()
pass_distinction_log.fit(pass_distinction_prepared, pass_distinction_labels)

#################################
#### Preparing the test sets ####
#################################
students_test_set_labels = students_test_set["final_result"].copy()
students_test_set = students_test_set.drop("final_result", axis=1)

pass_fail_test_set_labels = pass_fail_test_set["final_result"].copy()
pass_fail_test_set = pass_fail_test_set.drop("final_result", axis=1)

pass_distinction_test_set_labels = pass_distinction_test_set["final_result"].copy()
pass_distinction_test_set = pass_distinction_test_set.drop("final_result", axis=1)

students_test_prepared = full_pipeline.transform(students_test_set)
pass_fail_test_prepared = full_pipeline.transform(pass_fail_test_set)
pass_distinction_test_prepared = full_pipeline.transform(pass_distinction_test_set)

###############################
#### Evaluating the models ####
###############################

#students.plot(kind="scatter", x="avg_score", y="final_result")
#plt.show()