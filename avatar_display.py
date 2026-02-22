# avatar_display.py (Placeholder for designer's work)

def get_avatar_state(stress_score):
    """Returns a description of the avatar based on stress."""
    if stress_score > 7:
        return "Tired/Stressed Avatar"
    elif stress_score > 4:
        return "Neutral Avatar"
    else:
        return "Energetic/Happy Avatar"

def display_avatar_for_log(log):
    """Prints the avatar state to console (to be replaced with UI code later)."""
    print(f"--- DISPLAYING: {log.avatar} ---") 