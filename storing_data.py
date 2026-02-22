import json
import os
import joblib
import numpy as np
from datetime import datetime
from avatar_display import get_avatar_state, display_avatar_for_log

# Represents one daily log entry for a user
class UserLog: 
    def __init__(
        self,
        name            : str,
        date            : str   = None,
        hours_slept     : float = None,
        sleep_quality   : int   = None,
        activity_level  : int   = None,
        blood_pressure  : float = None,
        avg_heart_rate  : int   = None,
        highest_heart_rate : int = None,
        study_load      : int   = None,
        daily_steps     : int   = None,
        predicted_stress: float = None
    ):
        self.avatar             = None
        self.name               = name
        self.date               = date or datetime.today().strftime('%Y-%m-%d')
        self.hours_slept        = hours_slept
        self.sleep_quality      = sleep_quality
        self.activity_level     = activity_level
        self.blood_pressure     = blood_pressure
        self.avg_heart_rate     = avg_heart_rate
        self.highest_heart_rate = highest_heart_rate
        self.study_load         = study_load
        self.daily_steps        = daily_steps
        self.predicted_stress   = predicted_stress


    def to_dict(self) -> dict:
        return {
            "name"               : self.name,
            "date"               : self.date,
            "hours_slept"        : self.hours_slept,
            "sleep_quality"      : self.sleep_quality,
            "activity_level"     : self.activity_level,
            "blood_pressure"     : self.blood_pressure,
            "avg_heart_rate"     : self.avg_heart_rate,
            "highest_heart_rate" : self.highest_heart_rate,
            "study_load"         : self.study_load,
            "daily_steps"        : self.daily_steps,
            "predicted_stress"   : self.predicted_stress,
            "avatar"             : self.avatar
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UserLog":
        return cls(**{k: v for k, v in data.items() if k not in ('avatar',)})

    def __repr__(self):
        return (
            f"UserLog({self.name} | {self.date} | "
            f"stress={self.predicted_stress} | avatar={self.avatar if self.avatar else 'N/A'})"
        )

# Saves and loads user log entries to a local JSON file
class UserDataStore:
    def __init__(self, filepath: str = "user_logs.json"):
        self.filepath = filepath
        if not os.path.exists(self.filepath):
            with open(self.filepath, 'w') as f:
                json.dump({}, f)

    def _load_all(self) -> dict:
        with open(self.filepath, 'r') as f:
            return json.load(f)

    def _save_all(self, data: dict):
        with open(self.filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def save(self, log: UserLog):
        data = self._load_all()
        if log.name not in data:
            data[log.name] = {}
        data[log.name][log.date] = log.to_dict()
        self._save_all(data)
        print(f"Saved log for {log.name} on {log.date}")

    def get_history(self, name: str) -> list[UserLog]:
        data = self._load_all()
        if name not in data:
            print(f"No logs found for {name}")
            return []
        return [
            UserLog.from_dict(entry)
            for entry in sorted(data[name].values(), key=lambda x: x['date'])
        ]

    def get_latest(self, name: str) -> UserLog | None:
        history = self.get_history(name)
        return history[-1] if history else None

    def get_by_date(self, name: str, date: str) -> UserLog | None:
        data = self._load_all()
        entry = data.get(name, {}).get(date)
        return UserLog.from_dict(entry) if entry else None

    def get_stress_trend(self, name: str) -> list[dict]:
        history = self.get_history(name)
        return [
            {"date": log.date, "stress_score": log.predicted_stress}
            for log in history
            if log.predicted_stress is not None
        ]

    def all_users(self) -> list[str]:
        return list(self._load_all().keys())

# Give feeback to user based on input
def build_feature_vector(user_input: dict, scaler, feature_cols: list) -> np.ndarray:
    alias_map = {
        'hours_slept'     : 'sleep_duration',
        'sleep_quality'   : 'sleep_quality',
        'activity_level'  : 'activity_level',
        'blood_pressure'  : 'blood_pressure',
        'avg_heart_rate'  : 'heart_level',
        'daily_steps'     : 'daily_steps',
        'hoursSlept'      : 'sleep_duration',
        'sleepQuality'    : 'sleep_quality',
        'activityLevel'   : 'activity_level',
        'bloodPressure'   : 'blood_pressure',
        'avgHeartRate'    : 'heart_level',
        'dailySteps'      : 'daily_steps',
        'sleep_duration'  : 'sleep_duration',
        'heart_level'     : 'heart_level',
    }
    normalized = {}
    for key, value in user_input.items():
        model_key = alias_map.get(key)
        if model_key:
            normalized[model_key] = value
    missing = [f for f in feature_cols if f not in normalized]
    if missing:
        raise ValueError(
            f"Missing required fields: {missing}\n"
            f"Received fields (after mapping): {list(normalized.keys())}\n"
            f"Expected fields: {feature_cols}"
        )
    raw = np.array([[normalized[f] for f in feature_cols]])
    scaled = scaler.transform(raw)
    return scaled

def get_feedback(
    stress_score       : float,
    hours_slept        : float,
    sleep_quality      : int,
    activity_level     : int,
    study_load         : int,
    avg_heart_rate     : int,
    daily_steps        : int,
    highest_heart_rate : int = None
) -> list:
    feedback = []
    if hours_slept < 6:
        feedback.append({
            "priority": 1,
            "category": "sleep",
            "tip": f"You only slept {hours_slept} hours. Aim for at least 7-8 hours to reduce stress significantly."
        })
    elif hours_slept < 7:
        feedback.append({
            "priority": 2,
            "category": "sleep",
            "tip": f"You slept {hours_slept} hours. Getting closer to 8 hours could help lower your stress."
        })
    if sleep_quality <= 3:
        feedback.append({
            "priority": 1,
            "category": "sleep_quality",
            "tip": "Your sleep quality was poor. Try avoiding screens 30 minutes before bed."
        })
    if activity_level < 20:
        feedback.append({
            "priority": 2,
            "category": "activity",
            "tip": "Low activity today. Even a 20-minute walk can reduce stress hormones."
        })
    if daily_steps < 5000:
        feedback.append({
            "priority": 3,
            "category": "steps",
            "tip": f"Only {daily_steps} steps today. Try to hit 7,000-10,000 steps for better mental health."
        })
    if study_load >= 4:
        feedback.append({
            "priority": 2,
            "category": "study",
            "tip": "Heavy study load detected. Try the Pomodoro technique — 25 min focused work, 5 min break."
        })
    if avg_heart_rate > 90:
        feedback.append({
            "priority": 1,
            "category": "heart_rate",
            "tip": f"Your average heart rate of {avg_heart_rate} bpm is elevated. Try a 5-minute breathing exercise."
        })
    if highest_heart_rate and highest_heart_rate > 130:
        feedback.append({
            "priority": 2,
            "category": "heart_rate",
            "tip": f"Your peak heart rate hit {highest_heart_rate} bpm. Make sure high intensity was intentional exercise and not anxiety."
        })
    if stress_score > 8:
        feedback.append({
            "priority": 1,
            "category": "general",
            "tip": "Your overall stress is very high today. Prioritize rest above everything else tonight."
        })
    if stress_score <= 4 and hours_slept >= 7 and activity_level >= 30:
        feedback.append({
            "priority": 0,
            "category": "positive",
            "tip": "Excellent day! Good sleep and activity are keeping your stress low. Keep it up!"
        })
    feedback.sort(key=lambda x: x['priority'])
    if not feedback:
        feedback.append({
            "priority": 0,
            "category": "general",
            "tip": "You're doing well overall. Stay consistent with your healthy habits!"
        })
    return feedback

def get_feedback_for_log(log) -> list:
    return get_feedback(
        stress_score       = log.predicted_stress,
        hours_slept        = log.hours_slept,
        sleep_quality      = log.sleep_quality,
        activity_level     = log.activity_level,
        study_load         = log.study_load,
        avg_heart_rate     = log.avg_heart_rate,
        daily_steps        = log.daily_steps,
        highest_heart_rate = log.highest_heart_rate
    )

# Predicts stress level based on user input
def predict_stress_for_log(log: UserLog, model_path: str = "stress_model.pkl") -> float:
    bundle = joblib.load(model_path)

    user_input = {
        "hours_slept"   : log.hours_slept,
        "sleep_quality" : log.sleep_quality,
        "activity_level": log.activity_level,
        "blood_pressure": log.blood_pressure,
        "avg_heart_rate": log.avg_heart_rate,
        "daily_steps"   : log.daily_steps
    }
    scaled = build_feature_vector(user_input, bundle['scaler'], bundle['feature_cols'])
    score  = bundle['model'].predict(scaled)[0]
    return float(np.clip(round(score, 1), 1.0, 10.0))

def log_user_entry(
    name            : str,
    hours_slept     : float,
    sleep_quality   : int,
    activity_level  : int,
    blood_pressure  : float,
    avg_heart_rate  : int,
    highest_heart_rate : int,
    study_load      : int,
    daily_steps     : int,
    date            : str = None,
    model_path      : str = "stress_model.pkl",
    store           : UserDataStore = None
) -> UserLog:
    if store is None:
        store = UserDataStore()
    log = UserLog(
        name=name, date=date,
        hours_slept=hours_slept, sleep_quality=sleep_quality,
        activity_level=activity_level, blood_pressure=blood_pressure,
        avg_heart_rate=avg_heart_rate, highest_heart_rate=highest_heart_rate,
        study_load=study_load, daily_steps=daily_steps
    )
    log.predicted_stress = predict_stress_for_log(log, model_path)
    log.avatar = get_avatar_state(log.predicted_stress)
    tips = get_feedback_for_log(log)
    for tip in tips:
        print(f"  [{tip['category']}] {tip['tip']}")
    store.save(log)
    display_avatar_for_log(log)
    return log
