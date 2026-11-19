# Smart Attendance System (WEEK 4 - Computer Vision / OpenCV)

A webcam-based attendance system: detect a face with a Haar Cascade
classifier, recognize it against previously registered people, and log
attendance to a CSV with pandas - avoiding duplicate entries and adding a
timestamp automatically.

## How this maps to the task hints

| Hint | Where it's done |
|---|---|
| Webcam capture using OpenCV | `register_faces.py`, `take_attendance.py` (`cv2.VideoCapture`) |
| Face Detection - Haar Cascade classifier | `face_engine.py` -> `detect_faces()` (`cv2.CascadeClassifier`) |
| Recognition - store face encodings | `face_engine.py` -> `train_recognizer()` / `load_recognizer()` (LBPH model saved to `trained_model/face_trained.yml`) |
| Avoid duplicate entries | `attendance_utils.py` -> `mark_attendance()` checks if the name is already in today's CSV |
| Add timestamp for each entry | `attendance_utils.py` -> `mark_attendance()` writes `Date` + `Time` columns |
| Save CSV properly using pandas | `attendance_utils.py` (`pandas.read_csv` / `to_csv`) |
| UI (Bonus) - Streamlit buttons | `streamlit_app.py` |
| Test in different lighting conditions | See **Testing** section below; histogram equalization is applied to every face crop to help with this |

## Why LBPH instead of the `face_recognition` library?

Many tutorials for this project use the `face_recognition` package, which
gives you real 128-dimension face "encodings". It's slightly more accurate,
but it depends on `dlib`, which needs to be **compiled from source** on most
systems (CMake + a C++ compiler) and frequently fails to install for
beginners, especially on Windows.

This project instead uses **Haar Cascade (detection) + LBPH Face Recognizer
(recognition)**, both part of `opencv-contrib-python` - a single `pip
install` with no compiler needed. The trained LBPH model saved to
`trained_model/face_trained.yml` is functionally the same idea as "stored
face encodings": a model of each registered person's face that new frames
are compared against.

If you want to upgrade to `face_recognition` later for better accuracy, keep
`face_engine.detect_faces()` for the Haar Cascade step, but replace
`train_recognizer`/`recognize_face` with calls to
`face_recognition.face_encodings()` and `face_recognition.compare_faces()`.

## Setup

1. **Create a virtual environment (recommended)**
   ```bash
   python3 -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   > IMPORTANT: install `opencv-contrib-python`, **not** plain `opencv-python`.
   > Only the `-contrib` build includes `cv2.face` (the recognizer). If you
   > see `AttributeError: module 'cv2' has no attribute 'face'`, you have
   > the wrong package installed - run:
   > ```bash
   > pip uninstall opencv-python opencv-python-headless opencv-contrib-python -y
   > pip install opencv-contrib-python
   > ```

3. If `pip install` says "module not found" after installing, make sure you're
   running the script with the **same** Python/pip pair, e.g.:
   ```bash
   python3 -m pip install -r requirements.txt
   python3 register_faces.py
   ```

## Usage

**Step 1 - Register at least one person:**
```bash
python3 register_faces.py
```
Enter a name, then look at the webcam. It automatically captures ~40 face
samples and trains the recognizer. Run this once per person you want the
system to recognize.

**Step 2 - Take attendance:**
```bash
python3 take_attendance.py
```
Opens the webcam, draws a green box + name around recognized faces (red box
+ "Unknown" otherwise), and logs attendance to
`attendance/Attendance_<today's date>.csv`. Press `q` to quit.

**Bonus - Streamlit UI:**
```bash
streamlit run streamlit_app.py
```
Buttons/pages for Register New Face, Take Attendance, and View Attendance
(uses snapshot-based `st.camera_input` instead of a live video loop, which
is the simplest reliable way to use a webcam inside Streamlit).

## Testing in different lighting conditions

The task explicitly calls this out, so test it deliberately:

1. Register a person in **normal indoor lighting**.
2. Try `take_attendance.py` in:
   - A **brighter** room / near a window
   - A **dim** room
   - With a **light source behind** the person (backlighting)
3. Note when detection (the green/red box) fails vs. when detection works
   but recognition gets the wrong name / says "Unknown".
4. `face_engine.preprocess_face()` applies `cv2.equalizeHist()` to every
   face crop specifically to make recognition more robust to lighting
   changes - you can comment it out temporarily to see the difference it
   makes.
5. For a more robust real system, register each person under 2-3 different
   lighting conditions (just run `register_faces.py` again for the same
   name - new samples get added to their folder before the next retrain).

## Project structure

```
smart_attendance_system/
├── register_faces.py     # Step 1: capture + register a new person
├── take_attendance.py     # Step 2: live recognition + attendance logging
├── streamlit_app.py       # Bonus: simple web UI
├── face_engine.py         # Shared: Haar Cascade detection + LBPH recognition
├── attendance_utils.py    # Shared: pandas CSV logging, dedup, timestamps
├── requirements.txt
├── dataset/                # face samples per person (created at runtime)
│   └── <name>/img_0.jpg ...
├── trained_model/          # LBPH model + label map (created at runtime)
│   ├── face_trained.yml
│   └── labels.pickle
└── attendance/              # one CSV per day (created at runtime)
    └── Attendance_2026-09-09.csv
```

## Known limitations / next steps

- LBPH recognition accuracy is lower than deep-learning-based approaches -
  fine for a demo/internship project, not production security.
- Only registers/recognizes one face per frame during registration (by
  design, to avoid mixing samples); `take_attendance.py` handles multiple
  faces per frame fine.
- No liveness/anti-spoofing check (a printed photo could fool it) - out of
  scope for this project but worth mentioning in your report as a
  limitation.
