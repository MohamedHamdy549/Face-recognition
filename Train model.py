import numpy as np
from PIL import Image
import os
import cv2

def train_classifier(data_dir, output_dir):
    # Create the "runs" directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get the list of existing run directories
    existing_runs = os.listdir(output_dir)

    # Determine the name for the new run directory
    run_number = len(existing_runs) + 1
    new_run_dir = os.path.join(output_dir, f"run_{run_number}")

    # Create the new run directory
    os.makedirs(new_run_dir)

    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    faces = []
    ids = []
    label_id = 0
    id_map = {}

    for img_path in path:
        # print("Image Path:", img_path)
        # Load the image using PIL
        img = Image.open(img_path)
        # Convert the image to grayscale
        img = img.convert('L')
        # Convert the grayscale image data to a NumPy array with uint8 data type
        imgNP = np.array(img, dtype='uint8')
        # Extract the ID from the file name
        id = os.path.split(img_path)[1].split('.')[1]
        # print("Extracted ID:", id)

        if id not in id_map:
            id_map[id] = label_id
            label_id += 1
        face_id = id_map[id]

        faces.append(imgNP)
        ids.append(face_id)

    faces = np.array(faces)
    ids = np.array(ids)

    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)

    # Save the classifier XML file in the new run directory
    classifier_file = os.path.join(new_run_dir, "classifier.xml")
    clf.write(classifier_file)

    return new_run_dir

output_dir = "runs"
data_dir = 'dataset/face'
new_run_dir = train_classifier(data_dir, output_dir)
print(f"Output saved in: {new_run_dir}")
