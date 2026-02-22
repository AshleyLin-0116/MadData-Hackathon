import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error

class Train_Model:
    def __init__(self):
        # 1. Load only the two available datasets
        print("Loading datasets...")
        self.sleep_health = pd.read_csv("MadData-Hackathon\data\expanded_sleep_health_dataset.csv")
        self.student_stress = pd.read_csv("MadData-Hackathon\data\Student Stress Factors.csv")
        
        # 2. Clean Sleep Health Data
        self.sleep_health_clean = self.sleep_health[[
            'Sleep Duration', 'Quality of Sleep', 
            'Physical Activity Level', 'Stress Level', 
            'Blood Pressure', 'Heart Rate', 'Daily Steps', 'Occupation'
        ]].copy()
        
        self.sleep_health_clean = self.sleep_health_clean.rename(columns={
            'Sleep Duration': 'sleep_duration',
            'Quality of Sleep': 'sleep_quality',
            'Physical Activity Level': 'activity_level',
            'Stress Level': 'stress_level',
            'Blood Pressure': 'blood_pressure',
            'Heart Rate': 'heart_level',
            'Daily Steps': 'daily_steps',
            'Occupation': 'occupation'
        })
        
        # Parse Blood Pressure (taking the systolic/top number)
        self.sleep_health_clean['blood_pressure'] = self.sleep_health_clean['blood_pressure'].apply(
            lambda x: float(str(x).split('/')[0]) if '/' in str(x) else 120.0
        )
        
        # 3. Clean Student Stress Data
        self.student_stress_clean = self.student_stress.rename(columns={
            'Kindly Rate your Sleep Quality 😴': 'sleep_quality',
            'How many times a week do you suffer headaches 🤕?': 'num_headaches',
            'How would you rate you academic performance 👩‍🎓?': 'academic_performance',
            'how would you rate your study load?': 'study_load',
            'How many times a week you practice extracurricular activities 🎾?': 'num_extracurricular',
            'How would you rate your stress levels?': 'stress_level'
        })
        
        # Drop any missing values
        self.sleep_health_clean.dropna(inplace=True)
        self.student_stress_clean.dropna(inplace=True)

    def train_regression(self, dataset='sleep'):
        if dataset == 'sleep':
            df = self.sleep_health_clean.copy()
            df = pd.get_dummies(df, columns=['occupation'], prefix='occ')
            occ_cols = [c for c in df.columns if c.startswith('occ_')]
            feature_cols = [
                'sleep_duration',
                'sleep_quality',
                'activity_level',
                'blood_pressure',
                'heart_level',
                'daily_steps'
            ] + occ_cols
            target = 'stress_level'
        else:
            df = self.student_stress_clean
            feature_cols = ['sleep_quality', 'num_headaches', 'academic_performance', 'study_load', 'num_extracurricular']
            
        X = df[feature_cols]
        y = df['stress_level']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        model = Ridge(alpha=1.0) # Using Ridge for better stability
        model.fit(X_train_s, y_train)
        
        print(f"Model trained for {dataset}. R2 Score: {r2_score(y_test, model.predict(X_test_s)):.4f}")
        return model, scaler, feature_cols

    def save_bundle(self, model, scaler, features, path='stress_model.pkl'):
        bundle = {'model': model, 'scaler': scaler, 'feature_cols': features}
        joblib.dump(bundle, path)
        print(f"Bundle saved to {path}")

if __name__ == "__main__":
    tm = Train_Model()
    # For the MVP, we will prioritize the 'sleep' dataset for the physical health model
    model, scaler, features = tm.train_regression(dataset='sleep')
    tm.save_bundle(model, scaler, features)