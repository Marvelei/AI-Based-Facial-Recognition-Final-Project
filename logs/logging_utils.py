# logging_utils.py

import os
import pandas as pd
from datetime import datetime

LOG_PATH = os.path.join("logs", "full_system_log.csv")
MODEL_VERSION_FILE = os.path.join("logs", "model_version.txt")


def get_current_model_version():
    """Read current model version from file, if not exists return 'v1'."""
    if not os.path.exists(MODEL_VERSION_FILE):
        # default model
        with open(MODEL_VERSION_FILE, "w") as f:
            f.write("v1")
        return "v1"
    with open(MODEL_VERSION_FILE, "r") as f:
        return f.read().strip()


def bump_model_version():
    """Increment model version and save to file."""
    current = get_current_model_version()
    if current.startswith("v") and current[1:].isdigit():
        num = int(current[1:]) + 1
        new_version = f"v{num}"
    else:
        new_version = "v1"

    with open(MODEL_VERSION_FILE, "w") as f:
        f.write(new_version)

    return new_version


def log_event(
    phase,
    event,
    subject=None,
    psi=None,
    accuracy=None,
    model_version=None,
    extra_info=None,
):
    """
    write a log entry to the CSV log file.
    """
    if model_version is None:
        model_version = get_current_model_version()

    new_row = pd.DataFrame(
        [
            {
                "timestamp": datetime.now().isoformat(),
                "phase": phase,
                "event": event,
                "subject": subject,
                "psi": psi,
                "accuracy": accuracy,
                "model_version": model_version,
                "extra_info": extra_info,
            }
        ]
    )

    if os.path.exists(LOG_PATH):
        df = pd.read_csv(LOG_PATH)
        df = pd.concat([df, new_row], ignore_index=True)
    else:
        df = new_row

    df.to_csv(LOG_PATH, index=False)
