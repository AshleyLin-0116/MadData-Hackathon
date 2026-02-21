import pandas as pd
import zipfile
import os

class Avatar_Model:
    stress = None
    mental_health = pd.read_csv("MentalHealthSurvey.csv")
    sleep_health = pd.read_csv("expanded_sleep_health_dataset.csv")
    student_stress = pd.read_csv("Student Stress Factors.csv")
    mental_health_clean = mental_health[[
        'gender', 
        'age', 
        'average_sleep', 
        'study_satisfaction',
        'academic_workload ', 
        'academic_pressure', 
        'financial_concerns',
        'social_relationships', 
        'depression', 
        'anxiety', 
        'isolation',
        'future_insecurity'
    ]]
    sleep_health_clean = sleep_health[[
        'Gender', 
        'Age', 
        'Sleep Duration',
        'Quality of Sleep', 
        'Physical Activity Level', 
        'Stress Level', 
        'Blood Pressure', 
        'Heart Rate', 
        'Daily Steps'
    ]]
    sleep_health_clean = sleep_health_clean.rename(columns = {
        'Gender' : 'gender', 
        'Age' : 'age', 
        'Sleep Duration' : 'sleep_duration',
        'Quality of Sleep' : 'sleep_quality', 
        'Physical Activity Level' : 'activity_level', 
        'Stress Level' : 'stress_level', 
        'Blood Pressure' : 'blood_pressure', 
        'Heart Rate' : 'heart_level', 
        'Daily Steps' : 'daily_steps'
    })
    student_stress_clean = student_stress.rename(columns = {
        'Kindly Rate your Sleep Quality 😴' : 'sleep_quality',
        'How many times a week do you suffer headaches 🤕?' : 'num_headaches',
        'How would you rate you academic performance 👩‍🎓?' : 'academic_performance',
        'how would you rate your study load?' : 'study_load',
        'How many times a week you practice extracurricular activities 🎾?' : 'num_extracurricular',
        'How would you rate your stress levels?' : 'stress_level'
    })
    mental_health_clean = mental_health_clean.dropna()
    sleep_health_clean = sleep_health_clean.dropna()
    student_stress_clean = student_stress_clean.dropna()
    