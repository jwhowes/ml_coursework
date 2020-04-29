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
#### How to run:																	 ####
####	- Ensure matplotlib, sklearn, numpy, and pandas are installed and up-to-date ####
####	- Run via the commmand line or through other means							 ####
####		- e.g. $ python classifier.py											 ####
####	- All data files should be stored as .csv in a subfolder called 'data'		 ####
#########################################################################################

show_corr = False
show_hist = False
do_grid_search = False

def result_to_int(result):
	"""Converts final_result attribute to an integer"""
	if result == "Distinction":
		return 2
	elif result == "Pass":
		return 1
	elif result == "Fail":
		return 0

def int_to_pass_fail(num):
	"""Converts result to pass/fail (pass = 1, fail = 0)"""
	if num == 2 or num == 1:
		return 1
	return 0

def show_metrics(X, y, classifier, multiclass=False):
	"""Displays metrics for a classifier acting on data X with true labels y"""
	pred_y = classifier.predict(X)
	if multiclass:
		print("\tPrecision:", precision_score(y, pred_y, average="macro"))  # Uses macro averaging for multiclass classifiers
		print("\tRecall:", recall_score(y, pred_y, average="macro"))
	else:
		print("\tPrecision:", precision_score(y, pred_y))
		print("\tRecall:", recall_score(y, pred_y))
	print("\tAccuracy:", accuracy_score(y, pred_y))
	if multiclass:
		print("\tF1:", f1_score(y, pred_y, average="macro"))
	else:
		print("\tF1:", f1_score(y, pred_y))

def show_corr_matrix(data, label):
	"""Displays the correlation matrix for data with a given label"""
	corr_matrix = data.corr()
	print(corr_matrix[label].sort_values(ascending=False))

def plot_scatter_matrices(data, attributes):
	"""Plots the scatter matrix for dataset 'data' and attribute names 'attributes'"""
	matrix = pd.plotting.scatter_matrix(students[attributes], figsize=(15,15), )
	for ax in matrix.ravel():
		ax.set_xlabel(ax.get_xlabel(), fontsize=15, rotation=0)
		ax.set_ylabel(ax.get_ylabel(), fontsize=15, rotation=90)
	plt.show()

def plot_logistic_classifier(X_unprepared, X_prepared, x, labels, log_classifier):
	"""Displays a scatter chart showing the distribution of a binary label and the predicted probability of each example against a certain attribute 'x'"""
	plt.figure(figsize=(4,4))
	plt.xlabel(x, fontsize=14)
	plt.ylabel("final_result", fontsize=14)
	plt.scatter(X_unprepared[x], labels)
	plt.scatter(X_unprepared[x], log_classifier.predict_proba(X_prepared)[:,1], s=0.1)

# import data
students = pd.read_csv("./data/studentInfo.csv")
registration = pd.read_csv("./data/studentRegistration.csv")
assessment = pd.read_csv("./data/studentAssessment.csv")

students = students[students["final_result"] != "Withdrawn"]  # Removes withdrawn students

#################################
#### Creating Aggregate data ####
#################################

assessment_agg = assessment.groupby('id_student').agg(  # Creates avg_score aggregate field
	avg_score=pd.NamedAgg(column='score', aggfunc=np.mean)
)
students = pd.merge(students, assessment_agg, on='id_student')  # Merges students data to add avg_score field to dataset

# Create and merge num_modules aggregate field
registration_agg = registration.drop(['code_presentation', 'date_registration', 'date_unregistration'], axis=1)
registration_agg = registration_agg.groupby('id_student').count()
registration_agg = registration_agg.rename(columns={
	'id_student': 'id_student',
	'code_module': 'num_modules'
})
students = pd.merge(students, registration_agg, on='id_student')  # Adding the number of modules a student is taking

# Displays histograms for numerical student data
if show_hist:
	students.hist(bins=50, figsize=(20, 15))
	plt.show()

students["final_result"] = students["final_result"].map(result_to_int)  # Convert final_result field to integer

# Displays correlation data for students
if show_corr:
	show_corr_matrix(students, "final_result")
	plot_scatter_matrices(students, ["num_of_prev_attempts", "studied_credits", "num_modules", "avg_score", "final_result"])

############################
#### Stratifed Sampling ####
############################
pass_fail = students.copy()
pass_fail["final_result"] = pass_fail["final_result"].map(int_to_pass_fail)  # Create pass_fail dataset: 1=pass, 0=fail

pass_distinction = students.loc[students["final_result"].isin([2, 1])].copy()
pass_distinction["final_result"] = pass_distinction["final_result"].map(lambda x: x - 1)  # Create pass_distinction dataset: 1=distinction, 0=pass

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)  # Begin stratified sampling
for train_index, test_index in split.split(students, students["final_result"]):  # Create student test and training sets
	students_train_set = students.iloc[train_index]
	students_test_set = students.iloc[test_index]

for train_index, test_index in split.split(pass_fail, pass_fail["final_result"]):  # Create pass/fail test and training sets
	pass_fail_train_set = pass_fail.iloc[train_index]
	pass_fail_test_set = pass_fail.iloc[test_index]

for train_index, test_index in split.split(pass_distinction, pass_distinction["final_result"]):  # Create pass/distinction test and training sets
	pass_distinction_train_set = pass_distinction.iloc[train_index]
	pass_distinction_test_set = pass_distinction.iloc[test_index]

# Separate labels from training sets
students = students_train_set.drop("final_result", axis=1)
students_labels = students_train_set["final_result"].copy()

pass_fail = pass_fail_train_set.drop("final_result", axis=1)
pass_fail_labels = pass_fail_train_set["final_result"].copy()

pass_distinction = pass_distinction_train_set.drop("final_result", axis=1)
pass_distinction_labels = pass_distinction_train_set["final_result"].copy()

#######################
#### Data Cleaning ####
#######################

# Separating numerical and categorical attributes
num_attribs = ["num_of_prev_attempts", "studied_credits", "num_modules", "avg_score"]
cat_attribs = ["code_module", "code_presentation", "gender", "region", "highest_education", "imd_band", "age_band", "disability"]

# Numerical pipeline
num_pipeline = sklearn.pipeline.Pipeline([
	('imputer', sklearn.impute.SimpleImputer(strategy="median")),  # Replace null values with the median
	('std_scaler', sklearn.preprocessing.StandardScaler())  # Standardises data ranges
])

# Categorical pipeline
cat_pipeline = sklearn.pipeline.Pipeline([
	('imputer', sklearn.impute.SimpleImputer(strategy="most_frequent")),  # Replace null values with most common for that attribute
	('one_hot', sklearn.preprocessing.OneHotEncoder())  # One-hot encoding
])

# Combine numerical and categorical pipelines
full_pipeline = sklearn.compose.ColumnTransformer([
	('num', num_pipeline, num_attribs),
	('cat', cat_pipeline, cat_attribs)
])

# Create prepared training sets
students_prepared = full_pipeline.fit_transform(students)  # Imputers etc. trained against full training set (fail, pass, and distinctions included)
pass_fail_prepared = full_pipeline.transform(pass_fail)
pass_distinction_prepared = full_pipeline.transform(pass_distinction)

###############################
#### Training the model(s) ####
###############################

# Create untrained models
students_forest = sklearn.ensemble.RandomForestClassifier(random_state=42, oob_score=True)
pass_fail_log = sklearn.linear_model.LogisticRegressionCV()  # sklearn's CV classifiers contain builit-in cross-validation for hyper paramter tuning.
pass_distinction_log = sklearn.linear_model.LogisticRegressionCV()

# Perform grid search for hyperparameter tuning (quite slow)
if do_grid_search:
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

# Train models against training data
students_forest.fit(students_prepared, students_labels)
pass_fail_log.fit(pass_fail_prepared, pass_fail_labels)
pass_distinction_log.fit(pass_distinction_prepared, pass_distinction_labels)

# Combine pass/fail and pass/distinction logistic regressors into one regressor
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

print("\nMetrics for logistic regressors on training data:")
print("Pass/fail:")
show_metrics(pass_fail_prepared, pass_fail_labels, pass_fail_log)
print("Pass/distinction:")
show_metrics(pass_distinction_prepared, pass_distinction_labels, pass_distinction_log)
print("Combined:")
show_metrics(students_prepared, students_labels, students_log, multiclass=True)
print("Combined confusion matrix:")
print(sklearn.metrics.confusion_matrix(students_labels, students_log.predict(students_prepared)))

print("\nMetrics for logistic regressors on test data:")
print("Pass/fail:")
show_metrics(pass_fail_test_prepared, pass_fail_test_set_labels, pass_fail_log)
print("Pass/distinction:")
show_metrics(pass_distinction_test_prepared, pass_distinction_test_set_labels, pass_distinction_log)
print("Combined:")
show_metrics(students_test_prepared, students_test_set_labels, students_log, multiclass=True)
print("Combined confusion matrix:")
print(sklearn.metrics.confusion_matrix(students_test_set_labels, students_log.predict(students_test_prepared)))

# Plotting predicted probability based on average score against final result
plot_logistic_classifier(pass_fail, pass_fail_prepared, "avg_score", pass_fail_labels, pass_fail_log)
plot_logistic_classifier(pass_distinction, pass_distinction_prepared, "avg_score", pass_distinction_labels, pass_distinction_log)
plt.show()

# Plotting ROC curves
print("Plotting pass/fail ROC")
sklearn.metrics.plot_roc_curve(pass_fail_log, pass_fail_test_prepared, pass_fail_test_set_labels)
plt.show()
print("Plotting pass/distinction ROC")
sklearn.metrics.plot_roc_curve(pass_distinction_log, pass_distinction_test_prepared, pass_distinction_test_set_labels)
plt.show()

print("\nMetrics for random forest on training data:")
show_metrics(students_prepared, students_labels, students_forest, multiclass=True)
print("\tOut of bag:", students_forest.oob_score_)
print("Confusion matrix:")
print(sklearn.metrics.confusion_matrix(students_labels, students_forest.predict(students_prepared)))

print("\nMetrics for random forest on test data:")
show_metrics(students_test_prepared, students_test_set_labels, students_forest, multiclass=True)
print("Confusion Matrix:")
print(sklearn.metrics.confusion_matrix(students_test_set_labels, students_forest.predict(students_test_prepared)))