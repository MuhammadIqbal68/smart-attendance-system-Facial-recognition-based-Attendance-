
import os
from datetime import datetime
import pandas as pd

ATTENDANCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attendance")
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

COLUMNS = ["Name", "Date", "Time"]


def _today_str():
    return datetime.now().strftime("%Y-%m-%d")


def get_csv_path(date_str=None):
    """Path to the attendance CSV for a given date (defaults to today)."""
    date_str = date_str or _today_str()
    return os.path.join(ATTENDANCE_DIR, f"Attendance_{date_str}.csv")


def _load_or_create(date_str=None):
    path = get_csv_path(date_str)
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            # Guard against a corrupted/empty file
            for col in COLUMNS:
                if col not in df.columns:
                    df[col] = pd.Series(dtype="object")
            return df[COLUMNS]
        except pd.errors.EmptyDataError:
            return pd.DataFrame(columns=COLUMNS)
    return pd.DataFrame(columns=COLUMNS)


def mark_attendance(name):
    """
    Mark `name` present right now.
    Returns (marked: bool, message: str).
    marked=False means the person was already marked present today (duplicate avoided).
    """
    if not name or name.strip() == "":
        return False, "Empty name - ignored."

    name = name.strip()
    date_str = _today_str()
    df = _load_or_create(date_str)

    if name in df["Name"].values:
        return False, f"{name} is already marked present today - duplicate skipped."

    now = datetime.now()
    new_row = pd.DataFrame([{
        "Name": name,
        "Date": date_str,
        "Time": now.strftime("%H:%M:%S"),
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(get_csv_path(date_str), index=False)
    return True, f"Marked present: {name} at {now.strftime('%H:%M:%S')}"


def view_attendance(date_str=None):
    """Return today's (or a given date's) attendance as a DataFrame."""
    return _load_or_create(date_str)


def list_available_dates():
    """List dates (YYYY-MM-DD) for which an attendance CSV exists."""
    dates = []
    for fname in sorted(os.listdir(ATTENDANCE_DIR)):
        if fname.startswith("Attendance_") and fname.endswith(".csv"):
            dates.append(fname[len("Attendance_"):-len(".csv")])
    return dates


if __name__ == "__main__":
    # Small self-test so this module can be verified without a webcam.
    print("Self-test: marking attendance for 'Alice' and 'Bob', then a duplicate 'Alice'.")
    print(mark_attendance("Alice"))
    print(mark_attendance("Bob"))
    print(mark_attendance("Alice"))  # should be skipped as duplicate
    print("\nToday's attendance sheet:")
    print(view_attendance())
