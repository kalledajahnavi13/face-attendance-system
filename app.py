import streamlit as st
import cv2
import os
import pandas as pd
from datetime import datetime

# ==========================================
# PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="AI Face Attendance",
    page_icon="📸",
    layout="centered"
)

st.title("📸 AI-Based Face Recognition Attendance System")

st.write(
    "Register students, train the face recognition model, "
    "and automatically mark attendance."
)

# ==========================================
# FILE PATHS
# ==========================================

MODEL_PATH = "model/face_model.yml"
ATTENDANCE_FILE = "attendance.csv"

# ==========================================
# STUDENT NAMES
# ==========================================

student_names = {
    1: "Jahnavi",
    2: "Geetha"
}

# ==========================================
# TRAIN MODEL SECTION
# ==========================================

st.header("🧠 Train Face Recognition Model")

if st.button("Train Model"):

    if not os.path.exists("dataset"):
        st.error("❌ Dataset folder not found.")
        st.stop()

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces = []
    ids = []

    for folder in os.listdir("dataset"):

        folder_path = os.path.join(
            "dataset",
            folder
        )

        if not os.path.isdir(folder_path):
            continue

        try:
            student_id = int(
                folder.split("_")[0]
            )
        except ValueError:
            continue

        for image_name in os.listdir(folder_path):

            image_path = os.path.join(
                folder_path,
                image_name
            )

            image = cv2.imread(
                image_path,
                cv2.IMREAD_GRAYSCALE
            )

            if image is not None:

                faces.append(image)
                ids.append(student_id)

    if len(faces) == 0:

        st.error("❌ No face data available.")

    else:

        recognizer.train(
            faces,
            __import__("numpy").array(ids)
        )

        os.makedirs("model", exist_ok=True)

        recognizer.save(
            MODEL_PATH
        )

        st.success(
            f"✅ Model trained successfully using "
            f"{len(faces)} face images."
        )

# ==========================================
# ATTENDANCE SECTION
# ==========================================

st.header("📸 Mark Attendance")

if st.button("Start Camera"):

    if not os.path.exists(MODEL_PATH):

        st.error(
            "❌ Face recognition model not found. "
            "Please train the model first."
        )

        st.stop()

    # Load trained model
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    recognizer.read(
        MODEL_PATH
    )

    # Face detector
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )

    # Open camera
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        st.error(
            "❌ Camera could not be opened."
        )

        st.stop()

    frame_placeholder = st.empty()
    result_placeholder = st.empty()

    recognized = False

    # Camera loop
    for _ in range(150):

        ret, frame = camera.read()

        if not ret:
            break

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(100, 100)
        )

        for (x, y, w, h) in faces:

            face = gray[
                y:y+h,
                x:x+w
            ]

            student_id, confidence = recognizer.predict(
                face
            )

            # ==================================
            # FACE RECOGNIZED
            # ==================================

            if confidence < 70:

                student_name = student_names.get(
                    student_id,
                    f"Student {student_id}"
                )

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x+w, y+h),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"{student_name}",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                result_placeholder.success(
                    f"✅ Face Recognized: {student_name}"
                )

                # Current date/time
                now = datetime.now()

                date_today = now.strftime(
                    "%Y-%m-%d"
                )

                current_time = now.strftime(
                    "%H:%M:%S"
                )

                # ==================================
                # LOAD ATTENDANCE
                # ==================================

                if os.path.exists(
                    ATTENDANCE_FILE
                ):

                    attendance = pd.read_csv(
                        ATTENDANCE_FILE
                    )

                else:

                    attendance = pd.DataFrame(
                        columns=[
                            "Student ID",
                            "Name",
                            "Date",
                            "Time"
                        ]
                    )

                # ==================================
                # CHECK DUPLICATE
                # ==================================

                already_marked = False

                if not attendance.empty:

                    already_marked = (
                        (
                            attendance["Student ID"]
                            == student_id
                        )
                        &
                        (
                            attendance["Date"]
                            == date_today
                        )
                    ).any()

                # ==================================
                # SAVE ATTENDANCE
                # ==================================

                if not already_marked:

                    new_record = pd.DataFrame([
                        {
                            "Student ID": student_id,
                            "Name": student_name,
                            "Date": date_today,
                            "Time": current_time
                        }
                    ])

                    attendance = pd.concat(
                        [
                            attendance,
                            new_record
                        ],
                        ignore_index=True
                    )

                    attendance.to_csv(
                        ATTENDANCE_FILE,
                        index=False
                    )

                    result_placeholder.success(
                        f"✅ Attendance marked for "
                        f"{student_name}"
                    )

                else:

                    result_placeholder.info(
                        f"ℹ️ {student_name} "
                        f"is already marked present today."
                    )

                recognized = True

            # ==================================
            # UNKNOWN FACE
            # ==================================

            else:

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x+w, y+h),
                    (0, 0, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "Unknown Face",
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

        # Show camera frame
        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        frame_placeholder.image(
            frame_rgb,
            channels="RGB"
        )

        if recognized:
            break

    camera.release()

    if not recognized:

        result_placeholder.warning(
            "⚠️ Face not recognized. "
            "Please look directly at the camera."
        )

# ==========================================
# ATTENDANCE TABLE
# ==========================================

st.header("📋 Attendance Records")

if os.path.exists(ATTENDANCE_FILE):

    attendance = pd.read_csv(
        ATTENDANCE_FILE
    )

    if not attendance.empty:

        st.dataframe(
            attendance,
            use_container_width=True
        )

    else:

        st.info(
            "No attendance records yet."
        )

else:

    st.info(
        "No attendance records yet."
    )