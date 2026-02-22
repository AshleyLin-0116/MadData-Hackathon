import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error


class Train_Model:
    stress = None
    # Importing datasets
    mental_health = pd.read_csv("MentalHealthSurvey.csv")
    sleep_health = pd.read_csv("expanded_sleep_health_dataset.csv")
    student_stress = pd.read_csv("Student Stress Factors.csv")
    # Cleaning datasets
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
    mental_health_clean['gender'] = mental_health_clean['gender'].map({
        'Male': 0,
        'Female': 1
    })
    mental_health_clean['average_sleep'] = (mental_health_clean['average_sleep'].str.split('-').str[0].astype(int) + 1)
    sleep_health_clean['gender'] = sleep_health_clean['gender'].map({
        'Male': 0,
        'Female': 1
    })
    sleep_health_clean['blood_pressure'] = (sleep_health_clean['blood_pressure'].str.split('/').str[0].astype(float))
    # Training the Model
    def train_regression(self, dataset='sleep'):
        print(f"\n{'='*55}")
        print(f"  REGRESSION TRAINING  --  dataset: {dataset}")
        print(f"{'='*55}")
        if dataset == 'sleep':
            df = self.sleep_health_clean.copy()
            feature_cols = [
                'sleep_duration',
                'sleep_quality',
                'activity_level',
                'blood_pressure',
                'heart_level',
                'daily_steps'
            ]
            target = 'stress_level'
        elif dataset == 'student':
            df = self.student_stress_clean.copy()
            feature_cols = [
                'sleep_quality',
                'num_headaches',
                'academic_performance',
                'study_load',
                'num_extracurricular'
            ]
            target = 'stress_level'
        else:
            raise ValueError("dataset must be 'sleep' or 'student'")
        X = df[feature_cols]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge (a=1.0)'    : Ridge(alpha=1.0),
            'Lasso (a=0.1)'    : Lasso(alpha=0.1),
        }
        results = {}
        for name, model in models.items():
            model.fit(X_train_s, y_train)
            preds = model.predict(X_test_s)
            r2   = r2_score(y_test, preds)
            mse  = mean_squared_error(y_test, preds)
            rmse = np.sqrt(mse)
            results[name] = {'model': model, 'r2': r2, 'mse': mse, 'rmse': rmse}
            print(f"\n-- {name} --")
            print(f"  R2   : {r2:.4f}")
            print(f"  MSE  : {mse:.4f}")
            print(f"  RMSE : {rmse:.4f}")
            print("  Coefficients:")
            for feat, coef in zip(feature_cols, model.coef_):
                direction = "up stress" if coef > 0 else "down stress"
                print(f"    {feat:<25} {coef:+.4f}  ({direction})")
        best_name  = max(results, key=lambda k: results[k]['r2'])
        best_model = results[best_name]['model']
        print(f"\nBest model: {best_name}  (R2={results[best_name]['r2']:.4f})")
        return best_model, scaler, feature_cols
    # Save Model
    def save_model(self, model, scaler, feature_cols, path='stress_model.pkl'):
        import joblib
        bundle = {
            'model'       : model,
            'scaler'      : scaler,
            'feature_cols': feature_cols
        }
        joblib.dump(bundle, path)
        print(f"\nModel saved -> {path}")

if __name__ == "__main__":
    tm = Train_Model()
    model, scaler, features = tm.train_regression(dataset='sleep')
    tm.save_model(model, scaler, features, path='stress_model.pkl')