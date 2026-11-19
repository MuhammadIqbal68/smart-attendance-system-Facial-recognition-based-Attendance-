

import os
import pickle
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "trained_model")
MODEL_PATH = os.path.join(MODEL_DIR, "face_trained.yml")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.pickle")

IMG_SIZE = (200, 200)          # fixed size every face is resized to
CONFIDENCE_THRESHOLD = 75.0    # LBPH distance: LOWER = more confident match

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


_local_cascade_path = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
_fallback_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

_cascade_path = _local_cascade_path if os.path.exists(_local_cascade_path) else _fallback_cascade_path
face_cascade = cv2.CascadeClassifier(_cascade_path)
if face_cascade.empty():
    raise RuntimeError(
        f"Could not load Haar Cascade classifier. Tried:\n"
        f"  - {_local_cascade_path}\n"
        f"  - {_fallback_cascade_path}\n"
        f"Fix: re-download haarcascade_frontalface_default.xml from "
        f"https://github.com/opencv/opencv/tree/master/data/haarcascades "
        f"and place it in the project folder, or reinstall opencv-contrib-python."
    )


def detect_faces(gray_frame):
    """Return list of (x, y, w, h) boxes for every face found in a grayscale frame."""
    return face_cascade.detectMultiScale(
        gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )


def preprocess_face(gray_frame, box):
    """
    Crop a detected face, resize to a fixed size, and apply histogram
    equalization so the model is more robust to different lighting
    conditions (bright rooms, dim rooms, backlighting).
    """
    x, y, w, h = box
    face = gray_frame[y:y + h, x:x + w]
    face = cv2.resize(face, IMG_SIZE)
    face = cv2.equalizeHist(face)
    return face



def train_recognizer():
    """
    Rebuild the recognizer from every image saved under dataset/<name>/*.jpg.
    Called after each new person is registered.
    Returns (num_people, num_images) trained on.
    """
    faces, labels = [], []
    label_to_name = {}
    name_to_label = {}
    next_label = 0

    people = sorted(
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    )
    for name in people:
        person_dir = os.path.join(DATASET_DIR, name)
        images = [f for f in os.listdir(person_dir) if f.lower().endswith((".jpg", ".png"))]
        if not images:
            continue
        name_to_label[name] = next_label
        label_to_name[next_label] = name
        for img_name in images:
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, IMG_SIZE)
            faces.append(img)
            labels.append(next_label)
        next_label += 1

    if not faces:
        raise ValueError("No training images found. Register at least one person first.")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    recognizer.save(MODEL_PATH)

    with open(LABELS_PATH, "wb") as f:
        pickle.dump(label_to_name, f)

    return len(people), len(faces)


def load_recognizer():
    """Load the trained model + label map. Returns (recognizer, label_to_name) or (None, {})."""
    if not (os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH)):
        return None, {}
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)
    with open(LABELS_PATH, "rb") as f:
        label_to_name = pickle.load(f)
    return recognizer, label_to_name


def recognize_face(recognizer, label_to_name, face_img):
    """
    Predict identity of a preprocessed face crop.
    Returns (name_or_None, confidence). Lower confidence = better match.
    name is None if no model is trained yet or confidence is above threshold (unknown face).
    """
    if recognizer is None:
        return None, None
    label, confidence = recognizer.predict(face_img)
    if confidence <= CONFIDENCE_THRESHOLD:
        return label_to_name.get(label, "Unknown"), confidence
    return None, confidence


def save_face_sample(name, face_img, index):
    """Save one captured face sample to dataset/<name>/img_<index>.jpg."""
    person_dir = os.path.join(DATASET_DIR, name)
    os.makedirs(person_dir, exist_ok=True)
    path = os.path.join(person_dir, f"img_{index}.jpg")
    cv2.imwrite(path, face_img)
    return path


def registered_people():
    """List of names currently registered (have a dataset folder with images)."""
    if not os.path.isdir(DATASET_DIR):
        return []
    return sorted(
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
        and any(f.lower().endswith((".jpg", ".png")) for f in os.listdir(os.path.join(DATASET_DIR, d)))
    )
