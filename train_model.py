import pandas as pd
import zipfile
import os

class Avatar_Model:
    stress = None
    mental_health = pd.read_csv("C:\Users\Ashle\Downloads\Student_Mental_Health_Survey", compression = "zip")
    student_stress = pd.read_csv("C:\Users\Ashle\Downloads\Student_Stress_Analysis", compression = "zip")
    sleep_health = pd.read_csv("C:\Users\Ashle\Downloads\Sleep_Health_and_Lifestyle_Dataset", compression = "zip")
    