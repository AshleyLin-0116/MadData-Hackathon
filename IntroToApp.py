
    import tkinter as tk
from tkinter import Tk, messagebox

# Function to handle the "Make my pet" button click
def make_pet():
    # Hide the welcome frame
    welcome_frame.pack_forget()
    # Show the pet setup frame
    pet_frame.pack(pady=20)
    
# Function to handle pet setup actions (placeholder)
def setup_pet_tasks():
    messagebox.showinfo("Setup", "Here you can set up various tasks for your pet!")

# Create the main window
root = Tk.Tk()
root.title("Time Management Pet App")  # Window title
root.geometry("500x300")  # Set window size

# ----- Welcome Frame -----
welcome_frame = tk.Frame(root)

# Welcome message
welcome_label = tk.Label(
    welcome_frame, 
    text="Hello! Welcome to TimeBuddy!\n\n"
         "At TimeBuddy our goal is to help our users learn to maximize their time "
         "and reap the benefits of having a balanced work and sleep schedule…\n\n"
         "But first let's introduce you to your new pet!",
    wraplength=400,
    justify="center"
)
welcome_label.pack(pady=20)

# "Make my pet" button
make_pet_button = tk.Button(
    welcome_frame, 
    text="Make my pet", 
    command=make_pet
)
make_pet_button.pack(pady=10)

welcome_frame.pack(pady=20)

# ----- Pet Setup Frame -----
pet_frame = tk.Frame(root)

# Setup label
setup_label = tk.Label(
    pet_frame,
    text="Set up various tasks for your pet below!",
    wraplength=400,
    justify="center"
)
setup_label.pack(pady=10)

# Button to simulate task setup
setup_button = tk.Button(
    pet_frame,
    text="Setup Tasks",
    command=setup_pet_tasks
)
setup_button.pack(pady=10)

# Run the Tkinter event loop

root.mainloop()
