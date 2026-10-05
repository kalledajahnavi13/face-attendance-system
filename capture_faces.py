import cv2
import os

student_id = input("Enter Student ID: ")
student_name = input("Enter Student Name: ")

folder_name = f"{student_id}_{student_name}"
dataset_path = os.path.join("dataset", folder_name)

os.makedirs(dataset_path, exist_ok=True)

cam = cv2.VideoCapture(0)

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

count = 0

print("Camera started.")
print("Look at the camera...")
print("Press Q to stop.")

while True:
    ret, frame = cam.read()

    if not ret:
        print("Could not read camera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:
        count += 1

        face = gray[y:y+h, x:x+w]

        file_path = os.path.join(
            dataset_path,
            f"face_{count}.jpg"
        )

        cv2.imwrite(file_path, face)

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            f"Images: {count}/30",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("Face Registration", frame)

    key = cv2.waitKey(100) & 0xFF

    if key == ord("q") or count >= 30:
        break

cam.release()
cv2.destroyAllWindows()

print()
print(f"Registration completed!")
print(f"Saved {count} face images.")
print(f"Location: {dataset_path}")