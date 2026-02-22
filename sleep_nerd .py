"""
Sleep Nerd  —  Predictive Sleep Health App
==========================================
Single file, zero dependencies beyond Python's stdlib.
Run:  python sleep_nerd.py

Screens
  1. Intake Form  — name, age, occupation (searchable), stress slider
  2. Results      — mock risk dashboard (backend stub)
"""

import tkinter as tk
from tkinter import ttk
import math, json, os, random, threading, time
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
#  DATA
# ─────────────────────────────────────────────────────────────────────────────

CAREERS = [
    ("Air Traffic Controller",   5),
    ("Surgeon",                  5),
    ("Physician / Doctor",       5),
    ("Nurse (RN / NP)",          5),
    ("Police Officer",           5),
    ("Firefighter",              5),
    ("EMT / Paramedic",          5),
    ("Military Personnel",       5),
    ("Trader / Broker",          5),
    ("Attorney / Lawyer",        4),
    ("Psychologist / Therapist", 4),
    ("Social Worker",            4),
    ("Chef / Cook",              4),
    ("Journalist / Reporter",    4),
    ("Flight Attendant",         4),
    ("Accountant",               3),
    ("Architect",                3),
    ("Civil Engineer",           3),
    ("Data Scientist",           3),
    ("Financial Advisor",        3),
    ("Healthcare Admin",         3),
    ("Teacher (K-12)",           3),
    ("Programmer / Dev",         3),
    ("Professor / Academic",     3),
    ("Real Estate Agent",        3),
    ("Veterinarian",             3),
    ("Marketing / PR",           3),
    ("Pharmacist",               3),
    ("Student",                  3),
    ("Graphic / UX Designer",    2),
    ("Retail / Sales",           2),
    ("Warehouse / Logistics",    2),
    ("Farmer",                   2),
    ("Carpenter / Trades",       2),
    ("Other / Not Listed",       3),
]
CAREER_NAMES = [c[0] for c in CAREERS]
CAREER_STRESS = {c[0]: c[1] for c in CAREERS}

DATA_FILE = "sleepnerd_profile.json"

def load_profile():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE) as f:
                return json.load(f)
        except:
            pass
    return {}

def save_profile(d):
    with open(DATA_FILE, "w") as f:
        json.dump(d, f, indent=2)

# ─────────────────────────────────────────────────────────────────────────────
#  MOCK BACKEND  (replace with real API call later)
# ─────────────────────────────────────────────────────────────────────────────

RISK_DATA = [
    # condition, category, min_age, max_age, min_stress, base_risk, severity
    ("Anxiety Disorder",        "Mental",   18, 99,  4, 0.40, "high"),
    ("Major Depression",        "Mental",   18, 99,  5, 0.36, "high"),
    ("Burnout Syndrome",        "Mental",   22, 60,  5, 0.44, "high"),
    ("Chronic Stress",          "Mental",   18, 99,  4, 0.50, "moderate"),
    ("Cognitive Decline",       "Mental",   40, 99,  3, 0.30, "moderate"),
    ("Substance Use Risk",      "Mental",   18, 65,  5, 0.28, "moderate"),
    ("Cardiovascular Disease",  "Physical", 30, 99,  3, 0.45, "high"),
    ("Type 2 Diabetes",         "Physical", 28, 99,  4, 0.34, "high"),
    ("Hypertension",            "Physical", 25, 70,  5, 0.38, "high"),
    ("Obesity / Metabolic",     "Physical", 18, 99,  3, 0.30, "moderate"),
    ("Impaired Immunity",       "Physical", 18, 99,  2, 0.25, "moderate"),
    ("Alzheimer's Risk",        "Physical", 50, 99,  2, 0.48, "critical"),
    ("Workplace Injury Risk",   "Physical", 18, 65,  6, 0.32, "moderate"),
]

def mock_backend(profile):
    """
    Calls the real data-driven backend (backend.py).
    backend.py trains scikit-learn Random Forest models on
    expanded_sleep_health_dataset.csv (1,500 real records) at first call,
    then caches them for all subsequent predictions.
    Returns a normalised dict compatible with the UI.
    """
    try:
        import backend as _be
        # Map UI profile keys to backend keys
        be_profile = {
            "user_name":    profile.get("name", profile.get("user_name", "Friend")),
            "age":          int(profile.get("age", 30)),
            "occupation":   profile.get("occupation", "Other / Not Listed"),
            "stress":       int(profile.get("stress", 5)),
            "sleep_hours":  float(profile.get("sleep_hours", 7.0)),
            "sleep_quality":int(profile.get("sleep_quality", 3)),
            "exercise_days":int(profile.get("exercise_days", 3)),
            "caffeine_cups":int(profile.get("caffeine_cups", 2)),
            "screen_mins":  int(profile.get("screen_mins", 30)),
        }
        raw = _be.submit_profile(be_profile)

        # Normalise predictions: add "risk" key = likelihood for backward compat
        preds = raw.get("predictions", [])
        for p in preds:
            p["risk"] = p.get("likelihood", 0.0)

        return {
            "results":      preds,
            "sleep_score":  raw.get("sleep_score",  50),
            "health_score": raw.get("health_score", 50),
            "name":         be_profile["user_name"],
            "occupation":   be_profile["occupation"],
            "age":          be_profile["age"],
            "stress":       be_profile["stress"],
            "summary":      raw.get("summary",    ""),
            "advice":       raw.get("advice",     []),
            "dataset_n":    raw.get("dataset_n",   0),
            "model_type":   raw.get("model_type", ""),
        }
    except FileNotFoundError as e:
        # Dataset missing — return clear error
        return {
            "results": [], "sleep_score": 0, "health_score": 0,
            "name": profile.get("name","Friend"),
            "occupation": "", "age": 0, "stress": 0,
            "summary": str(e), "advice": [], "dataset_n": 0,
            "model_type": "ERROR",
        }
    except Exception as e:
        import traceback; traceback.print_exc()
        return {
            "results": [], "sleep_score": 0, "health_score": 0,
            "name": profile.get("name","Friend"),
            "occupation": "", "age": 0, "stress": 0,
            "summary": f"Backend error: {e}", "advice": [], "dataset_n": 0,
            "model_type": "ERROR",
        }

# ─────────────────────────────────────────────────────────────────────────────
#  COLOUR TOKENS
# ─────────────────────────────────────────────────────────────────────────────

BG       = "#0d0d1a"   # deepest background
BG2      = "#11111f"   # page background
CARD     = "#161625"   # card surface
CARD2    = "#1c1c30"   # elevated / input surface
BORDER   = "#222235"   # subtle border
BORDER2  = "#2e2e50"   # hover/active border
TEXT     = "#e4e4f0"   # primary
TEXT2    = "#8888aa"   # secondary
TEXT3    = "#44445a"   # muted
ACCENT   = "#4d9fff"   # blue accent
MOON_COL = "#c8dfff"   # moon face
MOON_GLW = "#2255cc"   # moon glow
ERR      = "#ff5566"
SEV = {
    "critical": "#ef4444",
    "high":     "#f97316",
    "moderate": "#f59e0b",
    "low":      "#22c55e",
}

# ─────────────────────────────────────────────────────────────────────────────
#  ANIMATED MOON
# ─────────────────────────────────────────────────────────────────────────────

class Moon(tk.Canvas):
    def __init__(self, parent, size=130, **kw):
        kw.setdefault("bg", BG2)
        super().__init__(parent, width=size, height=size,
                         highlightthickness=0, **kw)
        self._s = size
        self._t = 0.0
        self._on = True
        self._tick()

    def _tick(self):
        if not self._on: return
        self.delete("all")
        s = self._s
        cx = cy = s / 2
        t = self._t

        # Soft outer ambient haze
        for i in range(5):
            r = 52 + i * 9
            a = 0.06 - i * 0.010
            col = _blend(BG2, MOON_GLW, a)
            self.create_oval(cx-r, cy-r, cx+r, cy+r, fill=col, outline="")

        # Animated pulse ring
        pulse = 0.5 + 0.5 * math.sin(t * 1.3)
        pr    = 42 + pulse * 6
        pc    = _blend(BG2, MOON_GLW, 0.25 + pulse * 0.18)
        self.create_oval(cx-pr, cy-pr, cx+pr, cy+pr,
                         fill="", outline=pc, width=2)
        # second fainter ring
        pr2 = pr + 10 + pulse * 4
        pc2 = _blend(BG2, MOON_GLW, 0.09 + pulse * 0.06)
        self.create_oval(cx-pr2, cy-pr2, cx+pr2, cy+pr2,
                         fill="", outline=pc2, width=1)

        # Inner glow layers
        for gr, ga in [(36,0.45),(28,0.62),(20,0.80),(12,0.95)]:
            gc = _blend(BG2, MOON_GLW, ga)
            self.create_oval(cx-gr, cy-gr, cx+gr, cy+gr, fill=gc, outline="")

        # Moon disc
        rm = 26
        self.create_oval(cx-rm, cy-rm, cx+rm, cy+rm,
                         fill=MOON_COL, outline="")

        # Shadow bite — offset left to carve crescent
        ox, rs = 16, 24
        self.create_oval(cx-ox-rs, cy-rs, cx-ox+rs, cy+rs,
                         fill=CARD, outline="")

        # Limb highlight
        self.create_oval(cx+rm-5, cy-5, cx+rm+1, cy+5,
                         fill="#ddefff", outline="")

        # Stars
        for sx, sy, ph in [(cx+40,cy-20,0.0),(cx+50,cy+8,1.2),(cx-40,cy-28,0.6)]:
            a = 0.35 + 0.65 * abs(math.sin(t * 1.9 + ph))
            sc = _blend(BG2, "#ffffff", a)
            self.create_oval(sx-1.5, sy-1.5, sx+1.5, sy+1.5, fill=sc, outline="")

        self._t += 0.04
        self.after(40, self._tick)

    def stop(self): self._on = False


def _blend(a, b, t):
    t = max(0.0, min(1.0, t))
    ar,ag,ab = int(a[1:3],16), int(a[3:5],16), int(a[5:7],16)
    br,bg,bb = int(b[1:3],16), int(b[3:5],16), int(b[5:7],16)
    return f"#{int(ar+(br-ar)*t):02x}{int(ag+(bg-ag)*t):02x}{int(ab+(bb-ab)*t):02x}"


# ─────────────────────────────────────────────────────────────────────────────
#  SCORE RING
# ─────────────────────────────────────────────────────────────────────────────

class ScoreRing(tk.Canvas):
    def __init__(self, parent, score, size=170, **kw):
        kw.setdefault("bg", CARD)
        super().__init__(parent, width=size, height=size,
                         highlightthickness=0, **kw)
        self._size = size
        self._tgt  = score
        self._cur  = 0.0
        self._on   = True
        self._anim()

    def _anim(self):
        if not self._on: return
        self._cur += (self._tgt - self._cur) * 0.09
        if abs(self._tgt - self._cur) < 0.4:
            self._cur = self._tgt
        self._draw()
        if self._cur < self._tgt - 0.3:
            self.after(16, self._anim)

    def _draw(self):
        self.delete("all")
        s = self._size; cx = cy = s/2; r = s/2 - 16
        self.create_oval(cx-r, cy-r, cx+r, cy+r, outline=BORDER2, width=9)
        ext = (self._cur/100)*270
        col = "#22c55e" if self._cur>=70 else "#f59e0b" if self._cur>=45 else "#ef4444"
        if ext > 0:
            self.create_arc(cx-r, cy-r, cx+r, cy+r,
                            start=135, extent=-ext,
                            outline=col, width=9, style="arc")
        self.create_text(cx, cy-8,  text=str(int(self._cur)),
                         fill=col, font=("Segoe UI",32,"bold"))
        self.create_text(cx, cy+16, text="/ 100",
                         fill=TEXT2, font=("Segoe UI",9))

    def stop(self): self._on = False


# ─────────────────────────────────────────────────────────────────────────────
#  RISK BAR
# ─────────────────────────────────────────────────────────────────────────────

class RiskBar(tk.Canvas):
    def __init__(self, parent, value, severity, w=280, **kw):
        kw.setdefault("bg", CARD)
        super().__init__(parent, width=w, height=8,
                         highlightthickness=0, **kw)
        self._w   = w
        self._tgt = value
        self._cur = 0.0
        self._col = SEV.get(severity, SEV["moderate"])
        self._on  = True
        self._anim()

    def _anim(self):
        if not self._on: return
        self._cur += (self._tgt - self._cur) * 0.11
        if abs(self._tgt - self._cur) < 0.003: self._cur = self._tgt
        self._draw()
        if self._cur < self._tgt - 0.003:
            self.after(16, self._anim)

    def _draw(self):
        self.delete("all")
        self.create_rectangle(0,3, self._w,5, fill=BORDER2, outline="")
        fw = int(self._w * self._cur)
        if fw > 2:
            self.create_rectangle(0,2, fw,6, fill=self._col, outline="")

    def stop(self): self._on = False


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN APP
# ─────────────────────────────────────────────────────────────────────────────

class SleepNerd(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sleep Nerd")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.minsize(880, 620)

        try:
            self.state("zoomed")
        except tk.TclError:
            try: self.attributes("-zoomed", True)
            except: self.geometry("1280x800")

        self.bind("<F11>", lambda e: self.attributes("-fullscreen",
                  not self.attributes("-fullscreen")))
        self.bind("<Escape>", lambda e: self.attributes("-fullscreen", False))

        self._moons = []
        self._bars  = []
        self.profile = load_profile()
        self._show_form()

    # ── clear screen ─────────────────────────────────────────────────────────
    def _clear(self):
        for m in self._moons:
            try: m.stop()
            except: pass
        for b in self._bars:
            try: b.stop()
            except: pass
        self._moons = []
        self._bars  = []
        for w in self.winfo_children():
            w.destroy()

    # ── scrollable page ───────────────────────────────────────────────────────
    def _page(self):
        outer = tk.Frame(self, bg=BG2)
        outer.place(relwidth=1, relheight=1)
        cv  = tk.Canvas(outer, bg=BG2, highlightthickness=0)
        vsb = tk.Scrollbar(outer, orient="vertical", command=cv.yview)
        cv.configure(yscrollcommand=vsb.set)
        cv.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        body = tk.Frame(cv, bg=BG2)
        body.bind("<Configure>",
            lambda e: cv.configure(scrollregion=cv.bbox("all")))
        win = cv.create_window((0,0), window=body, anchor="nw")
        cv.bind("<Configure>", lambda e: cv.itemconfig(win, width=e.width))
        for w in (cv, body):
            w.bind("<MouseWheel>",
                lambda e, c=cv: c.yview_scroll(int(-1*(e.delta/120)), "units"))
        return body

    # ── nav bar ───────────────────────────────────────────────────────────────
    def _nav(self, parent, show_back=False):
        bar = tk.Frame(parent, bg=BG2)
        bar.pack(fill="x", padx=52, pady=(32,0))
        tk.Label(bar, text="Sleep Nerd", bg=BG2, fg=ACCENT,
                 font=("Segoe UI",13,"bold")).pack(side="left")
        if show_back:
            tk.Button(bar, text="← New Assessment", command=self._show_form,
                      bg=CARD2, fg=TEXT2, font=("Segoe UI",9),
                      relief="flat", padx=10, pady=4, cursor="hand2",
                      bd=0, activebackground=BORDER,
                      activeforeground=TEXT).pack(side="right")
        tk.Frame(parent, bg=BORDER, height=1).pack(
            fill="x", padx=52, pady=(12,0))

    # ══════════════════════════════════════════════════════════════════════════
    #  SCREEN 1 — INTAKE FORM
    # ══════════════════════════════════════════════════════════════════════════
    def _show_form(self):
        self._clear()
        body = self._page()
        self._nav(body)

        # ── Hero ─────────────────────────────────────────────────────────────
        hero = tk.Frame(body, bg=BG2)
        hero.pack(pady=(44,0))

        moon = Moon(hero, size=140, bg=BG2)
        moon.pack()
        self._moons.append(moon)

        tk.Label(hero, text="Sleep Nerd",
                 bg=BG2, fg=TEXT, font=("Segoe UI",36,"bold")).pack(pady=(16,4))
        tk.Label(hero,
                 text="Enter your profile to receive a personalised sleep health risk analysis.",
                 bg=BG2, fg=TEXT2, font=("Segoe UI",11),
                 wraplength=560, justify="center").pack()

        # ── Card ─────────────────────────────────────────────────────────────
        card_wrap = tk.Frame(body, bg=BG2)
        card_wrap.pack(fill="x", padx=80, pady=32)

        card = tk.Frame(card_wrap, bg=CARD,
                        highlightthickness=1, highlightbackground=BORDER)
        card.pack(fill="x")

        f = tk.Frame(card, bg=CARD)
        f.pack(fill="x", padx=40, pady=36)

        tk.Label(f, text="Your Profile", bg=CARD, fg=TEXT,
                 font=("Segoe UI",15,"bold")).pack(anchor="w", pady=(0,22))

        # ── Row: Name + Age ───────────────────────────────────────────────────
        row1 = tk.Frame(f, bg=CARD)
        row1.pack(fill="x")
        row1.columnconfigure(0, weight=3)
        row1.columnconfigure(1, weight=1)

        # Name
        nc = tk.Frame(row1, bg=CARD)
        nc.grid(row=0, column=0, sticky="ew", padx=(0,16))
        tk.Label(nc, text="Full Name", bg=CARD, fg=TEXT2,
                 font=("Segoe UI",9)).pack(anchor="w", pady=(0,5))
        self._name_var = tk.StringVar(value=self.profile.get("name",""))
        name_e = tk.Entry(nc, textvariable=self._name_var,
                          bg=CARD2, fg=TEXT, insertbackground=ACCENT,
                          font=("Segoe UI",12), relief="flat", bd=0,
                          highlightthickness=1,
                          highlightbackground=BORDER,
                          highlightcolor=ACCENT, width=28)
        name_e.pack(fill="x", ipady=10)
        self._name_err = tk.Label(nc, text="", bg=CARD, fg=ERR,
                                   font=("Segoe UI",8))
        self._name_err.pack(anchor="w")

        # Age
        ac = tk.Frame(row1, bg=CARD)
        ac.grid(row=0, column=1, sticky="ew")
        tk.Label(ac, text="Age", bg=CARD, fg=TEXT2,
                 font=("Segoe UI",9)).pack(anchor="w", pady=(0,5))
        self._age_var = tk.StringVar(value=str(self.profile.get("age","")))
        vcmd = (self.register(lambda v: v=="" or (v.isdigit() and len(v)<=3)),"%P")
        tk.Entry(ac, textvariable=self._age_var,
                 bg=CARD2, fg=TEXT, insertbackground=ACCENT,
                 font=("Segoe UI",12), relief="flat", bd=0,
                 highlightthickness=1, highlightbackground=BORDER,
                 highlightcolor=ACCENT, width=7,
                 validate="key", validatecommand=vcmd).pack(fill="x", ipady=10)
        self._age_err = tk.Label(ac, text="", bg=CARD, fg=ERR,
                                  font=("Segoe UI",8))
        self._age_err.pack(anchor="w")

        # ── Occupation ────────────────────────────────────────────────────────
        tk.Frame(f, bg=BORDER, height=1).pack(fill="x", pady=20)
        tk.Label(f, text="Occupation", bg=CARD, fg=TEXT2,
                 font=("Segoe UI",9)).pack(anchor="w", pady=(0,5))

        self._occ_var = tk.StringVar(value=self.profile.get("occupation",""))
        self._sel_occ = self.profile.get("occupation", None)

        occ_e = tk.Entry(f, textvariable=self._occ_var,
                         bg=CARD2, fg=TEXT, insertbackground=ACCENT,
                         font=("Segoe UI",12), relief="flat", bd=0,
                         highlightthickness=1, highlightbackground=BORDER,
                         highlightcolor=ACCENT, width=44)
        occ_e.pack(fill="x", ipady=10)

        # Placeholder behaviour
        if not self._occ_var.get():
            occ_e.insert(0, "Type to search careers…")
            occ_e.config(fg=TEXT3)
        def _occ_in(e):
            if occ_e.get() == "Type to search careers…":
                occ_e.delete(0, tk.END); occ_e.config(fg=TEXT)
        def _occ_out(e):
            if not occ_e.get():
                occ_e.insert(0, "Type to search careers…"); occ_e.config(fg=TEXT3)
            t = self._occ_var.get().strip()
            if t in CAREER_NAMES: self._sel_occ = t
        occ_e.bind("<FocusIn>",  _occ_in)
        occ_e.bind("<FocusOut>", _occ_out)

        self._occ_err = tk.Label(f, text="", bg=CARD, fg=ERR,
                                  font=("Segoe UI",8))
        self._occ_err.pack(anchor="w")

        # Dropdown list
        lb_wrap = tk.Frame(f, bg=CARD2,
                           highlightthickness=1, highlightbackground=BORDER)
        lb_wrap.pack(fill="x")
        lb = tk.Listbox(lb_wrap, bg=CARD2, fg=TEXT,
                        selectbackground=ACCENT, selectforeground="#fff",
                        font=("Segoe UI",11), relief="flat", bd=0,
                        highlightthickness=0, height=0, activestyle="none")
        lb.pack(fill="x")

        def _filter(*_):
            q = self._occ_var.get().lower()
            if q in ("", "type to search careers…"):
                lb.delete(0,tk.END); lb.config(height=0); return
            m = [n for n in CAREER_NAMES if q in n.lower()][:8]
            lb.delete(0,tk.END)
            for x in m: lb.insert(tk.END, f"  {x}")
            lb.config(height=len(m))

        def _pick(e):
            s = lb.curselection()
            if s:
                v = lb.get(s[0]).strip()
                self._sel_occ = v
                self._occ_var.set(v)
                occ_e.config(fg=TEXT)
                lb.delete(0,tk.END); lb.config(height=0)
                self._occ_err.config(text="")

        self._occ_var.trace_add("write", _filter)
        lb.bind("<<ListboxSelect>>", _pick)
        lb.bind("<Return>", _pick)
        occ_e.bind("<Down>",
            lambda e: (lb.focus_set(), lb.selection_set(0))
            if lb.size()>0 else None)

        # ── Quick-select career buttons ────────────────────────────────────────
        tk.Frame(f, bg=BORDER, height=1).pack(fill="x", pady=(16,12))
        tk.Label(f, text="Quick select:", bg=CARD, fg=TEXT3,
                 font=("Segoe UI",8)).pack(anchor="w", pady=(0,8))

        chips_outer = tk.Frame(f, bg=CARD)
        chips_outer.pack(fill="x")

        QUICK = [
            "Student", "Nurse (RN / NP)", "Physician / Doctor",
            "Programmer / Dev", "Teacher (K-12)", "Police Officer",
            "Surgeon", "Attorney / Lawyer", "Data Scientist",
            "Firefighter", "Chef / Cook", "Graphic / UX Designer",
        ]
        row_f = None
        for i, c in enumerate(QUICK):
            if i % 4 == 0:
                row_f = tk.Frame(chips_outer, bg=CARD)
                row_f.pack(fill="x", pady=3)
            def _pick_quick(career=c):
                self._sel_occ = career
                self._occ_var.set(career)
                occ_e.config(fg=TEXT)
                lb.delete(0,tk.END); lb.config(height=0)
                self._occ_err.config(text="")
            tk.Button(row_f, text=c, command=_pick_quick,
                      bg=CARD2, fg=TEXT2, font=("Segoe UI",9),
                      relief="flat", padx=10, pady=5, cursor="hand2",
                      bd=0, highlightthickness=1,
                      highlightbackground=BORDER,
                      activebackground=BORDER2,
                      activeforeground=TEXT).pack(side="left", padx=(0,6))

        # ── Sleep & lifestyle questions ───────────────────────────────────────
        tk.Frame(f, bg=BORDER, height=1).pack(fill="x", pady=(16,14))
        tk.Label(f, text="Sleep & Lifestyle", bg=CARD, fg=TEXT2,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0,10))

        def _slider_row(parent, label, var, lo, hi, lo_txt, hi_txt, resolution=1):
            row = tk.Frame(parent, bg=CARD)
            row.pack(fill="x", pady=5)
            lbl_f = tk.Frame(row, bg=CARD)
            lbl_f.pack(fill="x")
            tk.Label(lbl_f, text=label, bg=CARD, fg=TEXT2,
                     font=("Segoe UI", 9)).pack(side="left")
            val_lbl = tk.Label(lbl_f, text=str(var.get()), bg=CARD,
                                fg=ACCENT, font=("Segoe UI", 9, "bold"))
            val_lbl.pack(side="right")
            sc_row = tk.Frame(row, bg=CARD)
            sc_row.pack(fill="x")
            tk.Label(sc_row, text=lo_txt, bg=CARD, fg=TEXT3,
                     font=("Segoe UI", 8), width=6, anchor="e").pack(side="left")
            def _upd(v, vl=val_lbl): vl.config(text=v)
            tk.Scale(sc_row, variable=var, from_=lo, to=hi, orient="horizontal",
                     command=_upd, resolution=resolution,
                     bg=CARD, fg=TEXT2, troughcolor=CARD2,
                     activebackground=ACCENT, highlightthickness=0,
                     font=("Segoe UI", 8), length=430, sliderlength=16,
                     showvalue=False, relief="flat", sliderrelief="flat"
                     ).pack(side="left", padx=6)
            tk.Label(sc_row, text=hi_txt, bg=CARD, fg=TEXT3,
                     font=("Segoe UI", 8)).pack(side="left")

        self._sleep_hrs_var  = tk.DoubleVar(value=self.profile.get("sleep_hours",  7.0))
        self._sleep_qual_var = tk.IntVar(   value=self.profile.get("sleep_quality", 3))
        self._exercise_var   = tk.IntVar(   value=self.profile.get("exercise_days", 3))
        self._caffeine_var   = tk.IntVar(   value=self.profile.get("caffeine_cups", 2))
        self._screen_var     = tk.IntVar(   value=self.profile.get("screen_mins",  30))

        _slider_row(f, "Average nightly sleep (hours)",
                    self._sleep_hrs_var,  3, 12, "3h", "12h", resolution=0.5)
        _slider_row(f, "Sleep quality  (1 = terrible  →  5 = excellent)",
                    self._sleep_qual_var, 1, 5,  "poor", "great")
        _slider_row(f, "Exercise days per week",
                    self._exercise_var,   0, 7,  "none", "daily")
        _slider_row(f, "Caffeinated drinks per day",
                    self._caffeine_var,   0, 10, "0", "10+")
        _slider_row(f, "Screen time in last hour before bed (minutes)",
                    self._screen_var,     0, 60, "0 min", "60 min")

        # ── Stress slider ─────────────────────────────────────────────────────
        tk.Frame(f, bg=BORDER, height=1).pack(fill="x", pady=20)
        stress_hdr = tk.Frame(f, bg=CARD)
        stress_hdr.pack(fill="x", pady=(0,8))
        tk.Label(stress_hdr, text="Stress Level", bg=CARD, fg=TEXT2,
                 font=("Segoe UI",9)).pack(side="left")
        self._stress_disp = tk.Label(stress_hdr, text="5 / 10  •  moderate",
                                      bg=CARD, fg=ACCENT,
                                      font=("Segoe UI",9,"bold"))
        self._stress_disp.pack(side="right")

        self._stress_var = tk.IntVar(value=self.profile.get("stress",5))

        def _upd_stress(v):
            n = int(float(v))
            if n <= 3:   label = "low"
            elif n <= 6: label = "moderate"
            elif n <= 8: label = "high"
            else:        label = "very high"
            self._stress_disp.config(text=f"{n} / 10  •  {label}")

        scale_row = tk.Frame(f, bg=CARD)
        scale_row.pack(fill="x")
        tk.Label(scale_row, text="relaxed", bg=CARD, fg=TEXT3,
                 font=("Segoe UI",8), width=7, anchor="e").pack(side="left")
        sc = tk.Scale(scale_row,
                      variable=self._stress_var, from_=1, to=10,
                      orient="horizontal", command=_upd_stress,
                      bg=CARD, fg=TEXT2, troughcolor=CARD2,
                      activebackground=ACCENT, highlightthickness=0,
                      font=("Segoe UI",9), length=460, sliderlength=18,
                      showvalue=False, relief="flat", sliderrelief="flat")
        sc.pack(side="left", padx=8)
        tk.Label(scale_row, text="extreme", bg=CARD, fg=TEXT3,
                 font=("Segoe UI",8)).pack(side="left")
        _upd_stress(self._stress_var.get())

        # ── Submit ────────────────────────────────────────────────────────────
        tk.Frame(f, bg=BORDER, height=1).pack(fill="x", pady=20)

        sub_row = tk.Frame(f, bg=CARD)
        sub_row.pack(fill="x")

        self._g_err = tk.Label(sub_row, text="", bg=CARD, fg=ERR,
                                font=("Segoe UI",9))
        self._g_err.pack(side="left")

        tk.Button(sub_row, text="Analyse My Sleep Health  →",
                  command=self._submit,
                  bg=ACCENT, fg="#ffffff",
                  font=("Segoe UI",12,"bold"),
                  relief="flat", padx=22, pady=10,
                  cursor="hand2", bd=0,
                  activebackground="#2060cc",
                  activeforeground="#ffffff").pack(side="right")

        tk.Label(body,
                 text="F11 = fullscreen   •   Esc = exit fullscreen",
                 bg=BG2, fg=TEXT3, font=("Segoe UI",8)).pack(pady=(0,32))

    # ── validation + submit ───────────────────────────────────────────────────
    def _submit(self):
        name = self._name_var.get().strip()
        age  = self._age_var.get().strip()

        import re
        ne = None
        if not name:
            ne = "Name is required."
        elif not re.match(r"^[A-Za-z\s'\-\.]+$", name):
            ne = "Letters only — no numbers."
        elif len(name) < 2:
            ne = "At least 2 characters."

        ae = None
        if not age:
            ae = "Age is required."
        elif not age.isdigit() or not (13 <= int(age) <= 110):
            ae = "Must be a number between 13 and 110."

        if not self._sel_occ:
            t = self._occ_var.get().strip()
            if t in CAREER_NAMES: self._sel_occ = t
        oe = None if self._sel_occ else "Please select an occupation."

        self._name_err.config(text=ne or "")
        self._age_err.config(text=ae or "")
        self._occ_err.config(text=oe or "")

        if ne or ae or oe:
            self._g_err.config(text="Please fix the errors above.")
            return
        self._g_err.config(text="Analysing…")
        self.update_idletasks()

        self.profile = {
            "name":          name.title(),
            "age":           int(age),
            "occupation":    self._sel_occ,
            "stress":        self._stress_var.get(),
            "sleep_hours":   float(self._sleep_hrs_var.get()),
            "sleep_quality": int(self._sleep_qual_var.get()),
            "exercise_days": int(self._exercise_var.get()),
            "caffeine_cups": int(self._caffeine_var.get()),
            "screen_mins":   int(self._screen_var.get()),
        }
        save_profile(self.profile)

        def _call():
            result = mock_backend(self.profile)
            self.after(0, lambda: self._show_results(result))

        threading.Thread(target=_call, daemon=True).start()

    # ══════════════════════════════════════════════════════════════════════════
    #  SCREEN 2 — RESULTS
    # ══════════════════════════════════════════════════════════════════════════
    def _show_results(self, data):
        self._clear()
        body = self._page()
        self._nav(body, show_back=True)

        name         = data["name"]
        score        = data["sleep_score"]
        health_score = data.get("health_score", 50)
        risks        = data["results"]
        occ          = data["occupation"]
        age          = data["age"]
        stress       = data["stress"]

        # ── Hero row ─────────────────────────────────────────────────────────
        hero = tk.Frame(body, bg=BG2)
        hero.pack(fill="x", padx=60, pady=(40,0))

        # Moon
        moon_col = tk.Frame(hero, bg=BG2)
        moon_col.pack(side="left", anchor="n")
        moon = Moon(moon_col, size=120, bg=BG2)
        moon.pack()
        self._moons.append(moon)
        tk.Label(moon_col, text="Sleep Nerd", bg=BG2, fg=TEXT,
                 font=("Segoe UI",17,"bold")).pack(pady=(10,0))
        tk.Label(moon_col, text=f"Report for {name}",
                 bg=BG2, fg=TEXT2, font=("Segoe UI",9)).pack()

        # Score rings — sleep + health side by side
        rings_col = tk.Frame(hero, bg=BG2)
        rings_col.pack(side="left", padx=(40,0), anchor="n")

        ring_sl = tk.Frame(rings_col, bg=BG2)
        ring_sl.pack(side="left", padx=(0,18))
        ring = ScoreRing(ring_sl, score, size=155)
        ring.pack()
        tk.Label(ring_sl, text="sleep score",
                 bg=BG2, fg=TEXT3, font=("Segoe UI",8)).pack(pady=(4,0))

        ring_hl = tk.Frame(rings_col, bg=BG2)
        ring_hl.pack(side="left")
        ring_h = ScoreRing(ring_hl, health_score, size=155)
        ring_h.pack()
        tk.Label(ring_hl, text="overall health score",
                 bg=BG2, fg=TEXT3, font=("Segoe UI",8)).pack(pady=(4,0))

        # Profile summary + tips
        sum_col = tk.Frame(hero, bg=BG2)
        sum_col.pack(side="left", padx=(44,0), anchor="n", fill="x", expand=True)

        # Profile recap card
        pcard = tk.Frame(sum_col, bg=CARD,
                         highlightthickness=1, highlightbackground=BORDER)
        pcard.pack(fill="x", pady=(0,16))
        pi = tk.Frame(pcard, bg=CARD)
        pi.pack(fill="x", padx=20, pady=16)
        tk.Label(pi, text="Profile Summary", bg=CARD, fg=TEXT,
                 font=("Segoe UI",11,"bold")).pack(anchor="w", pady=(0,10))

        for label, val in [
            ("Name",        name),
            ("Age",         str(age)),
            ("Occupation",  occ),
            ("Stress",      f"{stress} / 10"),
        ]:
            row = tk.Frame(pi, bg=CARD)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=label, bg=CARD, fg=TEXT3,
                     font=("Segoe UI",9), width=11, anchor="w").pack(side="left")
            tk.Label(row, text=val, bg=CARD, fg=TEXT,
                     font=("Segoe UI",9,"bold"), anchor="w").pack(side="left")

        # Key insight
        # Key insight — use real backend summary if available
        backend_summary = data.get("summary", "")
        dataset_n       = data.get("dataset_n", 0)
        model_type      = data.get("model_type", "")

        if backend_summary:
            insight = backend_summary
        elif score >= 70:
            insight = "Your sleep profile is relatively healthy. Maintain your current habits and watch stress levels."
        elif score >= 45:
            insight = "Moderate risk indicators detected. Small consistent improvements to sleep and stress will reduce long-term risk significantly."
        else:
            insight = "Elevated risk profile detected. We recommend speaking with a healthcare professional about your sleep health."

        if score >= 70:   ic = "#22c55e"
        elif score >= 45: ic = "#f59e0b"
        else:             ic = "#ef4444"

        icard = tk.Frame(sum_col, bg=CARD,
                         highlightthickness=1, highlightbackground=ic)
        icard.pack(fill="x")
        tk.Frame(icard, bg=ic, width=4).pack(side="left", fill="y")
        tk.Label(icard, text=insight, bg=CARD, fg=TEXT2,
                 font=("Segoe UI", 10), wraplength=360,
                 justify="left", padx=16, pady=14).pack(side="left")

        # Personalised advice from backend
        advice = data.get("advice", [])
        if advice:
            tk.Label(sum_col, text="Recommendations",
                     bg=BG2, fg=TEXT, font=("Segoe UI", 11, "bold")).pack(
                         anchor="w", pady=(16, 6))
            for tip in advice:
                tip_row = tk.Frame(sum_col, bg=BG2)
                tip_row.pack(fill="x", pady=2)
                tk.Label(tip_row, text="→", bg=BG2, fg=ACCENT,
                         font=("Segoe UI", 10)).pack(side="left", padx=(0, 8))
                tk.Label(tip_row, text=tip, bg=BG2, fg=TEXT2,
                         font=("Segoe UI", 9), wraplength=380,
                         justify="left", anchor="w").pack(side="left", fill="x")

        # ── Divider ───────────────────────────────────────────────────────────
        tk.Frame(body, bg=BORDER, height=1).pack(
            fill="x", padx=52, pady=(36,0))

        # ── Risk section header ───────────────────────────────────────────────
        rh = tk.Frame(body, bg=BG2)
        rh.pack(fill="x", padx=60, pady=(24,4))
        tk.Label(rh, text="Predicted Health Risks",
                 bg=BG2, fg=TEXT, font=("Segoe UI",16,"bold")).pack(side="left")
        ds_n = data.get("dataset_n", 0)
        ds_label = f"Trained on {ds_n:,} real records  •  {data.get('model_type','')}" if ds_n else "Real data model"
        tk.Label(rh, text=ds_label,
                 bg=BG2, fg=TEXT3, font=("Segoe UI",8)).pack(
                     side="left", padx=12, anchor="s")

        # Filter tabs — store risks as instance var so lambdas close over it correctly
        self._all_risks = risks
        self._filt = tk.StringVar(value="All")
        tab_row = tk.Frame(body, bg=BG2)
        tab_row.pack(fill="x", padx=60, pady=(8,4))
        for opt in ("All", "Physical", "Mental"):
            tk.Radiobutton(tab_row, text=opt,
                           variable=self._filt, value=opt,
                           bg=BG2, fg=TEXT2, selectcolor=CARD2,
                           activebackground=BG2, font=("Segoe UI",10),
                           relief="flat",
                           command=lambda: self._render(self._all_risks)
                           ).pack(side="left", padx=(0,14))

        # Legend: border colours
        legend = tk.Frame(tab_row, bg=BG2)
        legend.pack(side="right")
        for lcol, ltxt in ((ACCENT, "Mental"), ("#20aa66", "Physical")):
            lf = tk.Frame(legend, bg=BG2)
            lf.pack(side="left", padx=(0,12))
            tk.Frame(lf, bg=lcol, width=12, height=12).pack(side="left", padx=(0,5))
            tk.Label(lf, text=ltxt, bg=BG2, fg=TEXT3,
                     font=("Segoe UI",8)).pack(side="left")

        self._grid_container = tk.Frame(body, bg=BG2)
        self._grid_container.pack(fill="x", padx=60, pady=(8,0))
        self._render(risks)

        # ── Disclaimer ────────────────────────────────────────────────────────
        tk.Label(body,
                 text=("ℹ  Risk predictions are derived from published epidemiological "
                       "datasets (CDC, NIH, AASM, Whitehall II). "
                       "This is informational only — not a clinical diagnosis. "
                       "Consult a healthcare professional for personalised advice."),
                 bg=BG2, fg=TEXT3, font=("Segoe UI",8,"italic"),
                 wraplength=860, justify="center").pack(pady=(28,6))

        ts = datetime.now().strftime("%B %d, %Y  %H:%M")
        tk.Label(body, text=f"Generated  {ts}  •  powered by real sleep health dataset",
                 bg=BG2, fg=TEXT3,
                 font=("Segoe UI",8)).pack(pady=(0,36))

    # ── render risk grid ──────────────────────────────────────────────────────
    def _render(self, risks):
        for w in self._grid_container.winfo_children():
            w.destroy()
        for b in self._bars:
            try: b.stop()
            except: pass
        self._bars = []

        filt = self._filt.get()
        shown = [r for r in risks
                 if filt == "All" or r["category"] == filt]

        SEV_ICON = {"critical":"★","high":"▲","moderate":"◆","low":"●"}

        grid = tk.Frame(self._grid_container, bg=BG2)
        grid.pack(fill="x")
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)

        for idx, r in enumerate(shown):
            row, col = divmod(idx, 2)
            sev  = r["severity"]
            cc   = SEV.get(sev, SEV["moderate"])
            cat  = r["category"]
            # Distinct border colour per category: blue=Mental, green=Physical
            cat_col   = ACCENT if cat == "Mental" else "#20aa66"
            border_col = cat_col   # card border reflects category, not just severity
            padl = (0, 10) if col == 0 else (10, 0)

            rc = tk.Frame(grid, bg=CARD,
                          highlightthickness=2, highlightbackground=border_col)
            rc.grid(row=row, column=col, sticky="nsew",
                    padx=padl, pady=7)

            # Left colour stripe — severity colour
            tk.Frame(rc, bg=cc, width=5).pack(side="left", fill="y")

            bdy = tk.Frame(rc, bg=CARD)
            bdy.pack(side="left", fill="both", expand=True, padx=18, pady=14)

            # Title row
            tr = tk.Frame(bdy, bg=CARD)
            tr.pack(fill="x")
            tk.Label(tr,
                     text=f"{SEV_ICON.get(sev,'●')}  {r['condition']}",
                     bg=CARD, fg=cc,
                     font=("Segoe UI",12,"bold"),
                     anchor="w").pack(side="left")

            # Severity badge
            b2 = tk.Frame(tr, bg=cc)
            b2.pack(side="right")
            tk.Label(b2, text=sev.upper(), bg=cc, fg="#fff",
                     font=("Segoe UI",7,"bold"),
                     padx=7, pady=3).pack()

            # Category chip — always blue for Mental, green for Physical
            ch = tk.Frame(bdy, bg=cat_col)
            ch.pack(anchor="w", pady=(7,0))
            tk.Label(ch, text=cat.upper(), bg=cat_col,
                     fg="#fff", font=("Segoe UI",7,"bold"),
                     padx=6, pady=2).pack()

            # Bar + pct
            likelihood = r.get("likelihood", r.get("risk", 0.0))
            pct = round(likelihood * 100)
            tk.Label(bdy, text=f"+{pct}% relative risk increase",
                     bg=CARD, fg=TEXT2, font=("Segoe UI",9)
                     ).pack(anchor="w", pady=(10,3))
            bar = RiskBar(bdy, likelihood, sev, w=300)
            bar.pack(anchor="w")
            self._bars.append(bar)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = SleepNerd()
    app.mainloop()
