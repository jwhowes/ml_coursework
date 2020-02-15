import matplotlib as mpl
import sklearn as skl
import numpy as np
import pandas as pd

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