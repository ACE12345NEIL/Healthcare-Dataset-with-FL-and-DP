"""Desktop interface (tkinter, included with Python). Run: python gui.py"""
import os
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

from src.config import ExperimentConfig
from src.pipeline import run_round1, run_round2, run_round3, run_round4, run_round5


class App(tk.Tk):
    FIELDS = [("Seed", "seed"), ("Test size", "test_size"), ("Clients", "n_clients"),
              ("Rounds", "rounds"), ("Local epochs", "local_epochs"), ("Learning rate", "lr"),
              ("Batch size", "batch_size"), ("Clip norm", "clip_norm"), ("Repeats", "n_repeats")]

    def __init__(self):
        super().__init__()
        self.title("Federated Learning + DP-inspired noise: Diabetes risk")
        self.geometry("900x720")
        d = ExperimentConfig().to_dict()
        if Path("config.json").exists():
            try:
                d = ExperimentConfig.from_json("config.json").to_dict()
            except Exception:
                pass
        self.v = {k: tk.StringVar(value=str(d[k])) for _, k in self.FIELDS}
        self.v["data_path"] = tk.StringVar(value=d["data_path"])
        self.v["noise_levels"] = tk.StringVar(value=",".join(str(x) for x in d["noise_levels"]))
        self.v["compare_noise"] = tk.StringVar(value=str(d["compare_noise"]))

        frm = ttk.Frame(self, padding=10)
        frm.pack(fill="x")
        ttk.Label(frm, text="Dataset CSV:").grid(row=0, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.v["data_path"], width=60).grid(row=0, column=1, columnspan=5, sticky="w", padx=5)
        ttk.Button(frm, text="Browse...", command=self.browse).grid(row=0, column=6)
        for i, (label, key) in enumerate(self.FIELDS):
            r, c = 1 + i // 3, (i % 3) * 2
            ttk.Label(frm, text=label + ":").grid(row=r, column=c, sticky="w", pady=3)
            ttk.Entry(frm, textvariable=self.v[key], width=10).grid(row=r, column=c + 1, sticky="w", padx=5)
        ttk.Label(frm, text="Noise levels:").grid(row=4, column=0, sticky="w", pady=3)
        ttk.Entry(frm, textvariable=self.v["noise_levels"], width=24).grid(row=4, column=1, sticky="w", padx=5)
        ttk.Label(frm, text="Compare at:").grid(row=4, column=2, sticky="w")
        ttk.Entry(frm, textvariable=self.v["compare_noise"], width=10).grid(row=4, column=3, sticky="w", padx=5)

        btns = ttk.Frame(self, padding=(10, 0))
        btns.pack(fill="x")
        self.buttons = []
        for text, fn in [("1 Data", run_round1), ("2 Centralized RF", run_round2),
                         ("3 Federated", run_round3), ("4 Privacy", run_round4),
                         ("5 Full experiments", run_round5)]:
            b = ttk.Button(btns, text=text, command=lambda f=fn: self.do(f))
            b.pack(side="left", padx=(0, 6))
            self.buttons.append(b)
        btns2 = ttk.Frame(self, padding=(10, 6))
        btns2.pack(fill="x")
        for text, cmd in [("Run tests", self.do_tests), ("Show plots", self.show_plots),
                          ("Open results folder", self.open_results)]:
            b = ttk.Button(btns2, text=text, command=cmd)
            b.pack(side="left", padx=(0, 6))
            if text == "Run tests":
                self.buttons.append(b)

        self.out = scrolledtext.ScrolledText(self, font=("Courier", 9))
        self.out.pack(fill="both", expand=True, padx=10, pady=10)
        self.show("Pick a stage. Round 5 runs everything and writes tables + plots to results/.\n"
                  "Simulated results; no formal differential-privacy guarantee.")

    def browse(self):
        f = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if f:
            self.v["data_path"].set(f)

    def show(self, text):
        self.out.delete("1.0", "end")
        self.out.insert("end", text)

    def cfg(self):
        try:
            d = ExperimentConfig().to_dict()
            for _, k in self.FIELDS:
                d[k] = type(d[k])(self.v[k].get())
            d["data_path"] = self.v["data_path"].get()
            d["noise_levels"] = [float(x) for x in self.v["noise_levels"].get().split(",") if x.strip()]
            d["compare_noise"] = float(self.v["compare_noise"].get())
            return ExperimentConfig.from_dict(d)
        except ValueError as e:
            messagebox.showerror("Invalid input", str(e))
            return None

    def _busy(self, busy):
        for b in self.buttons:
            b.config(state="disabled" if busy else "normal")

    def do(self, fn):
        cfg = self.cfg()
        if cfg:
            self._run(lambda: fn(cfg))

    def _run(self, fn):
        self.show("Running...")
        self._busy(True)

        def work():
            try:
                text = fn()
            except Exception as e:
                text = f"ERROR: {type(e).__name__}: {e}"
            self.after(0, lambda: (self.show(text), self._busy(False)))

        threading.Thread(target=work, daemon=True).start()

    def do_tests(self):
        def fn():
            r = subprocess.run([sys.executable, "-m", "pytest", "-q"], capture_output=True, text=True)
            return r.stdout + r.stderr
        self._run(fn)

    def show_plots(self):
        folder = Path("results/plots")
        files = sorted(folder.glob("*.png")) if folder.exists() else []
        if not files:
            messagebox.showinfo("No plots", "Run stage 5 (Full experiments) first.")
            return
        win = tk.Toplevel(self)
        win.title("Plots")
        nb = ttk.Notebook(win)
        nb.pack(fill="both", expand=True)
        win.images = []
        for f in files:
            try:
                img = tk.PhotoImage(file=str(f))
            except tk.TclError:
                continue
            win.images.append(img)
            frame = ttk.Frame(nb)
            ttk.Label(frame, image=img).pack()
            nb.add(frame, text=f.stem)

    def open_results(self):
        path = Path("results").resolve()
        path.mkdir(exist_ok=True)
        if sys.platform.startswith("win"):
            os.startfile(path)
        else:
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path)])


if __name__ == "__main__":
    App().mainloop()
