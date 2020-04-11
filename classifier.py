import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import StratifiedShuffleSplit
import sklearn.pipeline
import sklearn.impute
import sklearn.compose
import sklearn.preprocessing
import sklearn.linear_model
import sklearn.ensemble
import sklearn.tree
import sklearn.base
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import numpy as np
import pandas as pd
import collections

#########################################################################################
#### How to run:																	 ####
####	- Ensure matplotlib, sklearn, numpy, and pandas are installed and up-to-date ####
####	- Run via the commmand line or through other means							 ####
####		- e.g. python classifier.py												 ####
####	- All data files should be stored as .csv in a subfolder called 'data'		 ####
#########################################################################################

show_corr = False
show_hist = False
show_strat_hist = False
do_grid_search = False

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

def show_metrics(X, y, classifier, multiclass=False):
	pred_y = classifier.predict(X)
	if multiclass:
		print("Precision:", precision_score(y, pred_y, average="macro"))
		print("Recall:", recall_score(y, pred_y, average="macro"))
	else:
		print("Precision:", precision_score(y, pred_y))
		print("Recall:", recall_score(y, pred_y))
	print("Accuracy:", accuracy_score(y, pred_y))
	if multiclass:
		print("F1:", f1_score(y, pred_y, average="macro"))
	else:
		print("F1:", f1_score(y, pred_y))

def show_corr_matrix(data, label):
	corr_matrix = data.corr()
	print(corr_matrix[label].sort_values(ascending=False))

def plot_scatter_matrices(data, attributes):
	matrix = pd.plotting.scatter_matrix(students[attributes], figsize=(15,15), )
	for ax in matrix.ravel():
		ax.set_xlabel(ax.get_xlabel(), fontsize=15, rotation=0)
		ax.set_ylabel(ax.get_ylabel(), fontsize=15, rotation=90)
	plt.savefig("scatter_matrix.png")

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

#students = students.dropna(subset=["imd_band"])  # Drop students with null imd_band attribute (not used anymore)

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

if show_strat_hist:
	students["avg_score_cat"].hist()
	plt.show()

############################
#### Stratifed Sampling ####
############################
pass_fail = students.copy()
pass_fail["final_result"] = pass_fail["final_result"].map(int_to_pass_fail)

pass_distinction = students.loc[students["final_result"].isin([2, 1])].copy()
pass_distinction["final_result"] = pass_distinction["final_result"].map(lambda x: x - 1)  # 1: distinction, 0: pass

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)  # Remove random_state before submitting
for train_index, test_index in split.split(students, students["final_result"]):
	students_train_set = students.iloc[train_index]
	students_test_set = students.iloc[test_index]

for train_index, test_index in split.split(pass_fail, pass_fail["final_result"]):
	pass_fail_train_set = pass_fail.iloc[train_index]
	pass_fail_test_set = pass_fail.iloc[test_index]

for train_index, test_index in split.split(pass_distinction, pass_distinction["final_result"]):
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

cat_pipeline = sklearn.pipeline.Pipeline([
	('imputer', sklearn.impute.SimpleImputer(strategy="most_frequent")),
	('one_hot', sklearn.preprocessing.OneHotEncoder())
])

full_pipeline = sklearn.compose.ColumnTransformer([
	("num", num_pipeline, num_attribs),
	("cat", cat_pipeline, cat_attribs)
])

students_prepared = full_pipeline.fit_transform(students)
pass_fail_prepared = full_pipeline.transform(pass_fail)
pass_distinction_prepared = full_pipeline.transform(pass_distinction)

###############################
#### Training the model(s) ####
###############################

students_forest = sklearn.ensemble.RandomForestClassifier(random_state=42, oob_score=True)
pass_fail_log = sklearn.linear_model.LogisticRegressionCV()  # sklearn's CV classifiers contain builit-in cross-validation for hyper paramter tuning.
pass_distinction_log = sklearn.linear_model.LogisticRegressionCV()

if do_grid_search:  # Very slow for random forests
	param_grid = [{'n_estimators': [50, 75, 100], 'max_features': ["auto", 6, 8]}]
	grid_search = sklearn.model_selection.GridSearchCV(students_forest, param_grid, cv=5, scoring='roc_auc_ovr', return_train_score=True)
	grid_search.fit(students_prepared, students_labels)
	print("Random forest:")
	print(grid_search.best_params_)
	param_grid = [{'Cs': [5, 10, 15], 'max_iter': [100, 150]}]
	grid_search = sklearn.model_selection.GridSearchCV(pass_fail_log, param_grid, cv=5, scoring='roc_auc', return_train_score=True)
	grid_search.fit(pass_fail_prepared, pass_fail_labels)
	print("Pass fail log:")
	print(grid_search.best_params_)
	grid_search = sklearn.model_selection.GridSearchCV(pass_distinction_log, param_grid, cv=5, scoring='roc_auc', return_train_score=True)
	grid_search.fit(pass_distinction_prepared, pass_distinction_labels)
	print("Pass distinction log:")
	print(grid_search.best_params_)

students_forest.fit(students_prepared, students_labels)
pass_fail_log.fit(pass_fail_prepared, pass_fail_labels)
pass_distinction_log.fit(pass_distinction_prepared, pass_distinction_labels)

class Combine_logs(object):
	def __init__(self, log1, log2):
		self.log1 = log1
		self.log2 = log2
	def predict(self, X):
		y = self.log1.predict(X)
		for i in range(len(y)):
			if y[i] == 1:
				y[i] = self.log2.predict(X[i])[0] + 1
		return y

students_log = Combine_logs(pass_fail_log, pass_distinction_log)

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

print("here")
#students_log = sklearn.linear_model.LogisticRegressionCV(multi_class="multinomial")
#students_log.fit(students_prepared, students_labels)

#show_metrics(students_test_prepared, students_test_set_labels, students_log, multiclass=True)
#print("!!!!!!!!!!!!!!!!!!!!!")
#show_metrics(students_test_prepared, students_test_set_labels, students_forest, multiclass=True)
show_metrics(students_prepared, students_labels, students_log, multiclass=True)
print("!!!!!!!!!!!!!!!!!!!")
show_metrics(students_prepared, students_labels, students_forest, multiclass=True)
print("Out of bag:", students_forest.oob_score_)

sklearn.metrics.plot_roc_curve(pass_fail_log, pass_fail_prepared, pass_fail_labels)
plt.show()
sklearn.metrics.plot_roc_curve(pass_distinction_log, pass_distinction_prepared, pass_distinction_labels)
plt.show()