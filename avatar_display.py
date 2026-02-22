import os
from PIL import Image

def get_avatar_state(stress_score: float) -> str:
    if stress_score <= 3.5:
        return "happy"
    elif stress_score <= 6:
        return "tired"
    elif stress_score <= 8:
        return "stressed"
    else:
        return "overwhelmed"
    
def display_avatar(stress_score: float, avatar_dir: str = "avatars"):
    state    = get_avatar_state(stress_score)
    filename = f"avatar_{state}.png"
    filepath = os.path.join(avatar_dir, filename)
    print(f"\nStress score : {stress_score}")
    print(f"Avatar state : {state.upper()}")
    print(f"Loading      : {filepath}")
    if not os.path.exists(filepath):
        print(f"ERROR: Could not find {filepath}")
        print(f"Make sure your PNG files are in the '{avatar_dir}' folder")
        print(f"Expected files:")
        print(f"  avatar_happy.png")
        print(f"  avatar_tired.png")
        print(f"  avatar_stressed.png")
        print(f"  avatar_overwhelmed.png")
        return
    img = Image.open(filepath)
    img.show()
    print(f"Displaying avatar: {filename}")

def display_avatar_for_log(log, avatar_dir: str = "avatars"):
    if log.predicted_stress is None:
        print("No stress score found on this log entry.")
        return
    print(f"--- DISPLAYING: {log.avatar} ---")
    display_avatar(log.predicted_stress, avatar_dir)