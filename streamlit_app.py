

import numpy as np
import cv2
import streamlit as st

import face_engine as fe
from attendance_utils import mark_attendance, view_attendance, list_available_dates

st.set_page_config(page_title="Smart Attendance System", page_icon="🧑‍💻")
st.title("🧑‍💻 Smart Attendance System")
st.caption("Haar Cascade detection + LBPH recognition, with pandas-based attendance logging.")

mode = st.sidebar.radio("Choose an action", ["Register New Face", "Take Attendance", "View Attendance"])


def snapshot_to_gray(camera_image):
    file_bytes = np.asarray(bytearray(camera_image.getvalue()), dtype=np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)



if mode == "Register New Face":
    st.header("Register a new person")
    st.write("Enter a name, then take a few snapshots (different angles/expressions "
             "help accuracy). Click **Save & Train** when you're done.")

    name = st.text_input("Person's name (no spaces)")

    if "captured_count" not in st.session_state:
        st.session_state.captured_count = 0

    photo = st.camera_input("Take a snapshot")

    if photo is not None and name.strip():
        gray = snapshot_to_gray(photo)
        faces = fe.detect_faces(gray)
        if len(faces) == 0:
            st.warning("No face detected in that snapshot - try again with better lighting/framing.")
        else:
            x, y, w, h = faces[0]
            face_img = fe.preprocess_face(gray, (x, y, w, h))
            fe.save_face_sample(name.strip(), face_img, st.session_state.captured_count)
            st.session_state.captured_count += 1
            st.success(f"Sample #{st.session_state.captured_count} saved for '{name.strip()}'. "
                       "Take another snapshot from a slightly different angle, or click Save & Train.")

    if st.button("Save & Train"):
        try:
            n_people, n_imgs = fe.train_recognizer()
            st.success(f"Model trained on {n_people} people / {n_imgs} images.")
            st.session_state.captured_count = 0
        except ValueError as e:
            st.error(str(e))

    st.divider()
    st.write("**Registered people:**", fe.registered_people() or "None yet")


elif mode == "Take Attendance":
    st.header("Take attendance")
    recognizer, label_to_name = fe.load_recognizer()

    if recognizer is None:
        st.warning("No trained model yet. Register at least one person first.")
    else:
        photo = st.camera_input("Look at the camera and take a snapshot")
        if photo is not None:
            gray = snapshot_to_gray(photo)
            faces = fe.detect_faces(gray)
            if len(faces) == 0:
                st.warning("No face detected - try again with better lighting/framing.")
            else:
                x, y, w, h = faces[0]
                face_img = fe.preprocess_face(gray, (x, y, w, h))
                name, confidence = fe.recognize_face(recognizer, label_to_name, face_img)
                if name:
                    marked, message = mark_attendance(name)
                    if marked:
                        st.success(message)
                    else:
                        st.info(message)
                else:
                    st.error("Face not recognized (unknown person).")


elif mode == "View Attendance":
    st.header("Attendance records")
    dates = list_available_dates()
    if not dates:
        st.info("No attendance recorded yet.")
    else:
        chosen_date = st.selectbox("Select date", options=sorted(dates, reverse=True))
        df = view_attendance(chosen_date)
        st.dataframe(df, use_container_width=True)
        st.download_button("Download CSV", df.to_csv(index=False), file_name=f"Attendance_{chosen_date}.csv")
