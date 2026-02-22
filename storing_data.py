import json
import os
import joblib
import numpy as np
from datetime import datetime

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
        self.avatar             = get_avatar_state(predicted_stress) if predicted_stress is not None else None

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
            f"stress={self.predicted_stress} | avatar={self.avatar['state'] if self.avatar else 'N/A'})"
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

# Predicts stress level based on user input
def predict_stress_for_log(log: UserLog, model_path: str = "stress_model.pkl") -> float:
    bundle  = joblib.load(model_path)
    model   = bundle['model']
    scaler  = bundle['scaler']
    features = bundle['feature_cols']

    field_map = {
        'sleep_duration' : log.hours_slept,
        'sleep_quality'  : log.sleep_quality,
        'activity_level' : log.activity_level,
        'blood_pressure' : log.blood_pressure,
        'heart_level'    : log.avg_heart_rate,
        'daily_steps'    : log.daily_steps
    }

    raw = np.array([[field_map[f] for f in features]])
    scaled = scaler.transform(raw)
    score = model.predict(scaled)[0]

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
    store.save(log)
    print(f"\nResult for {name}:")
    print(f"  Predicted stress : {log.predicted_stress}")
    print(f"  Avatar state     : {log.avatar['state']}")
    print(f"  Message          : {log.avatar['message']}")
    return log