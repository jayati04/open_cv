# gui_simulation.py
import tkinter as tk
from tkinter import Label

class SmartHomeGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Smart Home Appliance Controller")
        self.root.geometry("400x300")
        self.root.config(bg="#e6f2ff")

        self.appliances = {
            "Light": False,
            "Fan": False,
            "TV": False
        }

        self.labels = {}
        self.setup_gui()

    def setup_gui(self):
        Label(self.root, text="Home Appliance Simulation", font=("Arial", 18, "bold"), bg="#e6f2ff").pack(pady=10)

        for appliance in self.appliances:
            frame = tk.Frame(self.root, bg="#e6f2ff")
            frame.pack(pady=5)

            tk.Button(frame, text=f"Toggle {appliance}", command=lambda a=appliance: self.toggle_appliance(a),
                      bg="#007acc", fg="white", width=15).pack(side="left", padx=10)

            self.labels[appliance] = Label(frame, text=f"{appliance}: OFF", fg="red",
                                           font=("Arial", 12, "bold"), bg="#e6f2ff")
            self.labels[appliance].pack(side="left")

    def toggle_appliance(self, name):
        self.appliances[name] = not self.appliances[name]
        self.update_status(name)

    def update_status(self, name):
        status = "ON" if self.appliances[name] else "OFF"
        color = "green" if self.appliances[name] else "red"
        self.labels[name].config(text=f"{name}: {status}", fg=color)
        print(f"[INFO] {name} turned {status}")

    def run(self):
        self.root.mainloop()
