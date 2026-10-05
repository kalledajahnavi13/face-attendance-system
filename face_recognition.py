import cv2
import os
import numpy as np

DATASET_DIR = "dataset"
TRAINER_FILE = "trainer.yml"

os.makedirs(DATASET_DIR, exist_ok=True)

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def capture_faces(student_id, num_samples=30):
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        return False, "Camera could not be opened."

    count = 0

    while True:
        ret, frame = camera.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = FACE_CASCADE.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )

        for (x, y, w, h) in faces:

            count += 1

            face = gray[y:y+h, x:x+w]

            filename = os.path.join(
                DATASET_DIR,
                f"User.{student_id}.{count}.jpg"
            )

            cv2.imwrite(filename, face)

            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                (255, 0, 0),
                2
            )

            cv2.putText(
                frame,
                f"Samples: {count}/{num_samples}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

        cv2.imshow("Face Registration", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27 or count >= num_samples:
            break

    camera.release()
    cv2.destroyAllWindows()

    if count == 0:
        return False, "No face detected."

    return True, f"{count} face samples captured successfully."


def train_model():

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces = []
    ids = []

    image_files = os.listdir(DATASET_DIR)

    for file in image_files:

        if not file.endswith(".jpg"):
            continue

        path = os.path.join(DATASET_DIR, file)

        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        try:
            student_id = int(file.split(".")[1])
        except:
            continue

        faces.append(image)
        ids.append(student_id)

    if len(faces) == 0:
        return False, "No face data available."

    recognizer.train(faces, np.array(ids))
    recognizer.save(TRAINER_FILE)

    return True, "Face model trained successfully."


def recognize_faces():

    if not os.path.exists(TRAINER_FILE):
        return None

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_FILE)

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        return None

    recognized_id = None

    while True:

        ret, frame = camera.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = FACE_CASCADE.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )

        for (x, y, w, h) in faces:

            student_id, confidence = recognizer.predict(
                gray[y:y+h, x:x+w]
            )

            if confidence < 70:

                recognized_id = student_id

                text = f"ID: {student_id}"

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x+w, y+h),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    text,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

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
                    "Unknown",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

        cv2.imshow("Face Recognition - Press Q to Stop", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or recognized_id is not None:
            break

    camera.release()
    cv2.destroyAllWindows()

    return recognized_id