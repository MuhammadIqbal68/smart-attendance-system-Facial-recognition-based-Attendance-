
import sys
import cv2
import face_engine as fe

SAMPLES_TO_CAPTURE = 40


def main():
    name = input("Enter the person's name (no spaces, e.g. John_Doe): ").strip()
    if not name:
        print("Name cannot be empty.")
        sys.exit(1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not access the webcam. Check it is connected and not used "
              "by another application.")
        sys.exit(1)

    print(f"\nLook at the camera. Capturing {SAMPLES_TO_CAPTURE} face samples for '{name}'.")
    print("Tip: slowly turn your head slightly left/right and try a couple of expressions "
          "so recognition works from different angles.")
    print("Tip: for best results, also register under a couple of different lighting "
          "conditions (see README -> Testing).")
    print("Press 'q' at any time to cancel.\n")

    count = 0
    while count < SAMPLES_TO_CAPTURE:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame from webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = fe.detect_faces(gray)

        for (x, y, w, h) in faces:
            face_img = fe.preprocess_face(gray, (x, y, w, h))
            fe.save_face_sample(name, face_img, count)
            count += 1

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Captured {count}/{SAMPLES_TO_CAPTURE}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            break  # only take one face per frame to avoid mixing samples

        cv2.imshow("Register Face - press q to cancel", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Cancelled by user.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if count == 0:
        print("No face samples captured - nothing was registered.")
        return

    print(f"\nCaptured {count} samples for '{name}'. Training recognizer on all "
          f"registered people...")
    n_people, n_imgs = fe.train_recognizer()
    print(f"Done. Model trained on {n_people} people / {n_imgs} total images.")
    print(f"Registered people so far: {fe.registered_people()}")


if __name__ == "__main__":
    main()
