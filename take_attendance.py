

import sys
import cv2
import face_engine as fe
from attendance_utils import mark_attendance, get_csv_path

# Track who we've already greeted with an on-screen "Marked!" message this
# session, so we don't spam mark_attendance() every single frame for the
# same person while they're standing in front of the camera.
already_greeted = set()


def main():
    recognizer, label_to_name = fe.load_recognizer()
    if recognizer is None:
        print("No trained model found. Run register_faces.py first to register at "
              "least one person.")
        sys.exit(1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not access the webcam.")
        sys.exit(1)

    print("Starting attendance. Press 'q' to quit.")
    print(f"Attendance will be saved to: {get_csv_path()}\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = fe.detect_faces(gray)

        for (x, y, w, h) in faces:
            face_img = fe.preprocess_face(gray, (x, y, w, h))
            name, confidence = fe.recognize_face(recognizer, label_to_name, face_img)

            if name:
                color = (0, 255, 0)
                label = f"{name} ({confidence:.0f})"
                if name not in already_greeted:
                    marked, message = mark_attendance(name)
                    print(message)
                    already_greeted.add(name)
            else:
                color = (0, 0, 255)
                label = "Unknown"

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2)

        cv2.imshow("Smart Attendance - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nSession ended.")


if __name__ == "__main__":
    main()
