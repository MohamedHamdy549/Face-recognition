# Face Detection and Recognition Project

This project includes code for collecting a dataset of faces, training a classifier for face recognition, and performing face recognition using OpenCV.

## Requirements

- Python (3.9)
- OpenCV 
- NumPy
- PIL (Python Imaging Library)

## Setup

1. Clone the repository:

git clone https://github.com/yourusername/your-repo.git

2. Install the required dependencies:

pip install -r requirements.txt

3. Download the pre-trained Haar cascade classifier XML file from [OpenCV GitHub](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml) and place it in the `utils` directory.

## Usage

### 1. Data Collection

Run the `generate_DS.py` script to collect face samples and create a dataset.

```bash
python generate_DS.py
```

### 2. Training

Run the `train_classifier.py` script to train a face recognition classifier.

```bash
python train_classifier.py
```

### 3. Face Recognition

Run the `face_recognition.py` script to perform face recognition using the trained classifier.

```bash
python face_recognition.py
```



## Directory Structure

- `dataset/`: Contains the collected face images.
- `utils/`: Contains utility files, such as the Haar cascade classifier XML file.
- `runs/`: Contains trained classifier models.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.# Face-recognition

# Face-recognition

