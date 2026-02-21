import tkinter as tk
from tkinter import messagebox, ttk
import json
import os
from datetime import date

# ── Data file to store tasks and sleep goal ──────────────────────────────────
DATA_FILE = "app_data.json"

def load_data():
    """Load saved tasks and sleep goal from file, or return defaults."""
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return {"tasks": [], "sleep_goal": 8, "user_name": ""}

def save_data(data):
    """Save tasks and sleep goal to file so they persist between sessions."""
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

# ── Main App Class ────────────────────────────────────────────────────────────
class PetSleepApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ZzzPet – Your Sleep & Task Companion")
        self.geometry("480x640")
        self.resizable(False, False)
        self.configure(bg="#1e1e2e")  # dark background

        # Load existing data
        self.data = load_data()

        # Show the welcome screen first
        self.show_welcome()

    # ── Color / style helpers ─────────────────────────────────────────────────
    def clr(self):
        """Remove all widgets from the window."""
        for w in self.winfo_children():
            w.destroy()

    def btn(self, parent, text, cmd, color="#7c3aed"):
        """Helper to create a styled button."""
        return tk.Button(
            parent, text=text, command=cmd,
            bg=color, fg="white", font=("Helvetica", 13, "bold"),
            relief="flat", padx=18, pady=10, cursor="hand2",
            activebackground="#6d28d9", activeforeground="white"
        )

    def label(self, parent, text, size=13, bold=False, color="#e0e0f0", wrap=400):
        """Helper to create a styled label."""
        weight = "bold" if bold else "normal"
        return tk.Label(
            parent, text=text, bg="#1e1e2e", fg=color,
            font=("Helvetica", size, weight), wraplength=wrap, justify="center"
        )

    # ── Screen 1: Welcome ─────────────────────────────────────────────────────
    def show_welcome(self):
        self.clr()
        self.label(self, "👋  Hello! Welcome to ZzzPet!", size=20, bold=True, color="#a78bfa").pack(pady=(40, 10))
        
        msg = (
            "At ZzzPet, our goal is to help our users learn to\n"
            "maximize their time and reap the benefits of having\n"
            "a balanced work and sleep schedule…\n\n"
            "But first, let's introduce you to your new pet! 🐾"
        )
        self.label(self, msg, size=12).pack(pady=10)

        # Big cute mascot emoji as placeholder avatar
        self.label(self, "🦉", size=64).pack(pady=10)

        self.btn(self, "✨  Make my pet", self.show_setup).pack(pady=20)

    # ── Screen 2: User Setup ──────────────────────────────────────────────────
    def show_setup(self):
        self.clr()
        self.label(self, "Let's set you up!", size=18, bold=True, color="#a78bfa").pack(pady=(30, 5))
        self.label(self, "Tell us a little about yourself so your pet\ncan give you personalized sleep tips.", size=11).pack(pady=5)

        form = tk.Frame(self, bg="#1e1e2e")
        form.pack(pady=15, padx=40, fill="x")

        def field(label_text, var):
            tk.Label(form, text=label_text, bg="#1e1e2e", fg="#a78bfa",
                     font=("Helvetica", 11, "bold"), anchor="w").pack(fill="x", pady=(8,1))
            entry = tk.Entry(form, textvariable=var, bg="#2d2d44", fg="white",
                             insertbackground="white", font=("Helvetica", 12),
                             relief="flat", bd=6)
            entry.pack(fill="x", ipady=4)

        self.name_var = tk.StringVar()
        self.age_var  = tk.StringVar()
        self.career_var = tk.StringVar()

        field("Your Name", self.name_var)
        field("Age", self.age_var)
        field("Career / Occupation", self.career_var)

        # Stress level slider
        tk.Label(form, text="Stress Level (1 = chill, 10 = max stress)",
                 bg="#1e1e2e", fg="#a78bfa", font=("Helvetica", 11, "bold"), anchor="w").pack(fill="x", pady=(8,1))
        self.stress_var = tk.IntVar(value=5)
        tk.Scale(form, variable=self.stress_var, from_=1, to=10, orient="horizontal",
                 bg="#2d2d44", fg="white", troughcolor="#7c3aed",
                 highlightthickness=0, font=("Helvetica", 10)).pack(fill="x")

        self.btn(self, "Continue →", self.finish_setup).pack(pady=20)

    def finish_setup(self):
        """Save the user's name and move to the main dashboard."""
        name = self.name_var.get().strip() or "Friend"
        self.data["user_name"] = name
        self.data["age"] = self.age_var.get()
        self.data["career"] = self.career_var.get()
        self.data["stress"] = self.stress_var.get()
        save_data(self.data)
        self.show_dashboard()

    # ── Screen 3: Main Dashboard ──────────────────────────────────────────────
    def show_dashboard(self):
        self.clr()
        name = self.data.get("user_name", "Friend")
        stress = self.data.get("stress", 5)

        # Mascot tip based on stress level
        if stress >= 7:
            tip = "Your stress is high! 😓 Try to wind down\n1 hour before bed — no screens!"
        elif stress >= 4:
            tip = "Moderate stress detected. 🌙\nA consistent bedtime can really help!"
        else:
            tip = "You're doing great! 😄\nKeep up that healthy sleep routine!"

        self.label(self, f"Hey {name}! 👋", size=18, bold=True, color="#a78bfa").pack(pady=(20, 2))
        self.label(self, f"Today: {date.today().strftime('%B %d, %Y')}", size=10, color="#888").pack()

        # Owl mascot + speech bubble
        bubble = tk.Frame(self, bg="#2d2d44", bd=0)
        bubble.pack(pady=10, padx=30, fill="x")
        tk.Label(bubble, text="🦉", bg="#2d2d44", font=("Helvetica", 36)).pack(side="left", padx=10, pady=8)
        tk.Label(bubble, text=tip, bg="#2d2d44", fg="#e0e0f0",
                 font=("Helvetica", 11), wraplength=300, justify="left").pack(side="left", padx=5)

        # Sleep goal display
        goal = self.data.get("sleep_goal", 8)
        self.label(self, f"🎯 Sleep Goal: {goal} hours/night", size=12, color="#34d399", bold=True).pack(pady=(10, 0))
        self.btn(self, "Edit Sleep Goal", self.show_sleep_goal, color="#059669").pack(pady=5)

        # Navigation buttons
        self.btn(self, "📋  To-Do List", self.show_todos).pack(pady=5)
        self.btn(self, "⚙️  Edit Profile", self.show_setup, color="#4b5563").pack(pady=5)

    # ── Screen 4: Sleep Goal ──────────────────────────────────────────────────
    def show_sleep_goal(self):
        self.clr()
        self.label(self, "🌙 Set Your Sleep Goal", size=18, bold=True, color="#a78bfa").pack(pady=(30, 10))

        # Fun sleep fact
        facts = [
            "Adults need 7–9 hours of sleep per night.",
            "Consistent sleep times boost your mood & memory.",
            "Sleep deprivation can feel like being drunk!",
            "Your brain clears toxins during deep sleep.",
        ]
        import random
        self.label(self, f"💡 Did you know?\n{random.choice(facts)}", size=11, color="#93c5fd").pack(pady=10)

        self.label(self, "How many hours of sleep per night?", size=12).pack(pady=10)
        self.goal_var = tk.IntVar(value=self.data.get("sleep_goal", 8))
        tk.Scale(self, variable=self.goal_var, from_=4, to=12, orient="horizontal",
                 bg="#1e1e2e", fg="white", troughcolor="#7c3aed",
                 highlightthickness=0, font=("Helvetica", 11),
                 length=300, label="hours").pack()

        def save_goal():
            self.data["sleep_goal"] = self.goal_var.get()
            save_data(self.data)
            messagebox.showinfo("Saved!", f"Sleep goal set to {self.goal_var.get()} hours! 🌙")
            self.show_dashboard()

        self.btn(self, "Save Goal", save_goal).pack(pady=15)
        self.btn(self, "← Back", self.show_dashboard, color="#4b5563").pack(pady=5)

    # ── Screen 5: To-Do List ──────────────────────────────────────────────────
    def show_todos(self):
        self.clr()
        self.label(self, "📋 Your To-Do List", size=18, bold=True, color="#a78bfa").pack(pady=(20, 5))

        # Scrollable task area
        frame = tk.Frame(self, bg="#1e1e2e")
        frame.pack(fill="both", expand=True, padx=20)

        canvas = tk.Canvas(frame, bg="#1e1e2e", highlightthickness=0)
        scrollbar = tk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        self.task_frame = tk.Frame(canvas, bg="#1e1e2e")

        self.task_frame.bind("<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.task_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.render_tasks()

        # Add new task input area
        add_frame = tk.Frame(self, bg="#1e1e2e")
        add_frame.pack(fill="x", padx=20, pady=8)
        self.task_entry = tk.Entry(add_frame, bg="#2d2d44", fg="white",
                                   insertbackground="white", font=("Helvetica", 12),
                                   relief="flat", bd=6)
        self.task_entry.pack(side="left", fill="x", expand=True, ipady=5, padx=(0,6))
        self.btn(add_frame, "+ Add", self.add_task, color="#7c3aed").pack(side="right")

        self.btn(self, "← Back", self.show_dashboard, color="#4b5563").pack(pady=(0,12))

    def render_tasks(self):
        """Draw all tasks with checkboxes and delete buttons."""
        for w in self.task_frame.winfo_children():
            w.destroy()

        if not self.data["tasks"]:
            tk.Label(self.task_frame, text="No tasks yet — add one below! 🎉",
                     bg="#1e1e2e", fg="#888", font=("Helvetica", 11)).pack(pady=20)
            return

        for i, task in enumerate(self.data["tasks"]):
            row = tk.Frame(self.task_frame, bg="#2d2d44", pady=4)
            row.pack(fill="x", pady=3, padx=2)

            # Checkbox to mark done
            done_var = tk.BooleanVar(value=task.get("done", False))
            def toggle(v=done_var, idx=i):
                self.data["tasks"][idx]["done"] = v.get()
                save_data(self.data)
                self.render_tasks()

            chk = tk.Checkbutton(row, variable=done_var, command=toggle,
                                  bg="#2d2d44", activebackground="#2d2d44",
                                  selectcolor="#7c3aed")
            chk.pack(side="left", padx=6)

            # Task text (strike-through style via color)
            color = "#555" if task.get("done") else "#e0e0f0"
            text = f"✓ {task['text']}" if task.get("done") else task["text"]
            tk.Label(row, text=text, bg="#2d2d44", fg=color,
                     font=("Helvetica", 11), anchor="w").pack(side="left", fill="x", expand=True)

            # Delete button
            def delete(idx=i):
                self.data["tasks"].pop(idx)
                save_data(self.data)
                self.render_tasks()

            tk.Button(row, text="✕", command=delete, bg="#991b1b", fg="white",
                      font=("Helvetica", 10, "bold"), relief="flat", padx=6,
                      cursor="hand2").pack(side="right", padx=6)

    def add_task(self):
        """Add a new task from the entry box."""
        text = self.task_entry.get().strip()
        if not text:
            messagebox.showwarning("Oops!", "Please type a task first.")
            return
        self.data["tasks"].append({"text": text, "done": False})
        save_data(self.data)
        self.task_entry.delete(0, tk.END)
        self.render_tasks()


# ── Run the app ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = PetSleepApp()
    app.mainloop()
