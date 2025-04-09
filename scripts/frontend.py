
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, scrolledtext
import subprocess
import threading
import os

# --- Configuration ---
scripts = [
    ("1. charuco_intrinsics", "charuco_intrinsics.py"),
    ("2. format_for_calibration", "format_for_calibration.py"),
    ("3. compute_relative_poses", "compute_relative_poses.py"),
    ("4. concatenate_relative_poses", "concatenate_relative_poses.py"),
    ("5. bundle_adjustment", "bundle_adjustment.py"),
    ("6. global_registration", "global_registration.py"),
    ("7. visualize", "17cams.py"),
]

#default_config_dir = os.path.join(os.path.dirname(__file__), "configs")
default_config_dir = "/nfs/exports/ratlv/calibration"

# --- GUI Functions ---
def log(message):
    log_widget.insert(tk.END, message + "\n")
    log_widget.see(tk.END)

def browse_config():
    path = filedialog.askopenfilename(
        title="Select Config File",
        initialdir=default_config_dir
    )
    if path:
        config_var.set(path)

def check_success(script_name):
    # Example: look for a .success file after script runs
    # Replace this with your real validation logic
    success_marker = f"{script_name}.success"
    return os.path.exists(success_marker)

def run_selected_scripts(config_path, selections):
    script_items = list(scripts)
    for i, (label, script) in enumerate(script_items):
        if not selections[script].get():
            continue

        log(f"Running {label} ({script}) with: -c {config_path}")
        try:
            process = subprocess.Popen(
                ["python", "-u", script, "-c", config_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )

            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    log(line.strip())

            exit_code = process.poll()
            if exit_code != 0:
                log(f"{label} failed with exit code {exit_code}. Stopping.")
                return

            # if not check_success(script):
            #     log(f"⚠️ {label} did not produce expected output. Stopping.")
            #     return

            log(f"{label} completed successfully.\n")

            # Move selection to the next script
            selections[script].set(False)
            if i + 1 < len(script_items):
                next_script = script_items[i + 1][1]
                selections[next_script].set(True)

            return  # Stop after one script to allow step-by-step flow

        except Exception as e:
            log(f"Error running {label}: {e}")
            return

    log("No more scripts selected.")


def start_run():
    config_path = config_var.get()
    if not config_path:
        log("Please select a config file first.")
        return
    log("Starting script sequence...\n")
    threading.Thread(target=run_selected_scripts, args=(config_path, selections), daemon=True).start()

# --- GUI Layout ---
root = tk.Tk()
root.title("Multiview calib")

# Fonts
ui_font = tkfont.Font(family="Segoe UI", size=11)
log_font = tkfont.Font(family="Courier New", size=10)
root.option_add("*Font", ui_font)

config_var = tk.StringVar()
selections = {}

tk.Label(root, text="Config File:").pack(pady=(10, 0))
tk.Entry(root, textvariable=config_var, width=60).pack(padx=10)
tk.Button(root, text="Browse", command=browse_config).pack(pady=5)

tk.Label(root, text="Select Scripts to Run:").pack(pady=(10, 0))
for i, (label, script) in enumerate(scripts):
    default_checked = (i == 0)  # Only Script 1 is checked
    var = tk.BooleanVar(value=default_checked)
    selections[script] = var
    tk.Checkbutton(root, text=label, variable=var).pack(anchor='w', padx=20)

tk.Button(root, text="Run Selected Scripts", command=start_run).pack(pady=10)

log_widget = scrolledtext.ScrolledText(root, width=100, height=25)
log_widget.pack(padx=10, pady=10)

root.mainloop()
