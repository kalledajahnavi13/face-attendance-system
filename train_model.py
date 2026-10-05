import cv2
import os
import numpy as np

dataset_path = "dataset"
model_path = "model"

os.makedirs(model_path, exist_ok=True)

recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
ids = []

for folder in os.listdir(dataset_path):

    folder_path = os.path.join(dataset_path, folder)

    if not os.path.isdir(folder_path):
        continue

    try:
        student_id = int(folder.split("_")[0])
    except:
        continue

    for image_name in os.listdir(folder_path):

        image_path = os.path.join(folder_path, image_name)

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            continue

        faces.append(image)
        ids.append(student_id)

print(f"Found {len(faces)} face images.")

if len(faces) == 0:
    print("No face data available.")
    exit()

recognizer.train(faces, np.array(ids))

recognizer.save(os.path.join(model_path, "face_model.yml"))

print("✅ Face recognition model trained successfully!")
print("Model saved to:")
print(os.path.join(model_path, "face_model.yml"))