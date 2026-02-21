import pandas as pd
import zipfile
import os

class Avatar_Model:
    stress = None
    mental_health = pd.read_csv("MentalHealthSurvey.csv")
    sleep_health = pd.read_csv("expanded_sleep_health_dataset.csv")
    student_stress = pd.read_csv("Student Stress Factors.csv")
    