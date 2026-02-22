# test_engine.py
from model.storing_data import log_user_entry, UserDataStore

# 1. Initialize the storage
store = UserDataStore("test_logs.json")

print("--- Testing Sleep Nerd Engine ---")

# 2. Simulate a user entry (Using real data values from your CSV logic)
# Inputs: name, sleep_hours, quality, activity, bp, avg_hr, peak_hr, study_load, steps
try:
    log = log_user_entry(
        name="Ben",
        hours_slept=6.5,
        sleep_quality=3,
        activity_level=15,      # Low exercise
        blood_pressure=135.0,   # Slightly high
        avg_heart_rate=82,
        highest_heart_rate=140,
        study_load=4,           # High stress factor
        daily_steps=3200,       # Low steps
        store=store
    )

    print(f"\nSUCCESS!")
    print(f"Predicted Stress: {log.predicted_stress}/10")
    print(f"Current Avatar State: {log.avatar}")
    print(f"History Saved to: {store.filepath}")

except Exception as e:
    print(f"\nERROR: Something went wrong with the integration.")
    print(f"Details: {e}")