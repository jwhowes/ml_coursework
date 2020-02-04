import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

students = pd.read_csv("./data/studentInfo.csv")

#students.hist(bins=50, figsize=(20,15))
#plt.show()

# Generic random sampling:
train_set, test_set = train_test_split(students, test_size=0.2)

# Stratified sampling based on studied_credits (is that the best field for it?):
students["credits_cat"] = np.ceil(students["studied_credits"]/100) # Only works with 100 (I don't know why).
students["credits_cat"].where(students["credits_cat"] < 300, 300.0, inplace=True)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
for train_index, test_index in split.split(students, students["credits_cat"]):
    strat_train_set = students.loc[train_index]
    strat_test_set = students.loc[test_index]

for set_ in (strat_train_set, strat_test_set):
    set_.drop("credits_cat", axis=1, inplace=True)

# Produce clean training set:
students = strat_train_set.drop("final_result", axis=1)
students_labels = strat_train_set["final_result"].copy()

# Cleaning the data:

# Replace null values with the median: (doesn't work because they're not integers)
#median = students["imd_band"].median()
#students["imd_band"].fillna(median, inplace=True)

imputer = SimpleImputer(strategy="median")

students_num = students.drop(["code_module", "code_presentation", "id_student", "gender", "region", "highest_education", "imd_band", "age_band", "disability"], axis=1)
imputer.fit(students_num)

X = imputer.transform(students_num)
students_tr = pd.DataFrame(X, columns=students_num.columns)

# With one-hot encoding (example):
one_hot = sklearn.preprocessing.OneHotEncoder()
students_cat = students[["highest_education"]]
students_cat_1hot = one_hot.fit_transform(students_cat)

#one-hot encoding:
one_hot = sklearn.preprocessing.OneHotEncoder()
students_nan = students[["gender", "region", "highest_education", "age_band", "disability"]]  # Doesn't work with imd_band (probably because there's missing values)
students_nan = one_hot.fit_transform(students_nan)

#Combine the two transforms:
students = pd.concat([students_tr, students_nan])  # Doesn't work (students_nan needs to be a dataframe object)

print(students)