import cv2
import numpy as np
from PIL import Image
import os


def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = classifier.detectMultiScale(gray_img, scaleFactor, minNeighbors)  # detect face?

    coords = []
    for (x, y, w, h) in face:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        id, pred = clf.predict(gray_img[y:y + h, x:x + w])
        conf = int(100 * (1 - pred / 300))
        print("Confidence:", conf, "ID:", id)

        if conf > 77:
            if id == 0:
                cv2.putText(img, "Mohamed Hamdy", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
            elif id == 1:
                cv2.putText(img, "mahmmoud mohy", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
            elif id == 2:
                cv2.putText(img, "faris oun", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
            else:
                cv2.putText(img, "unknown", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
        else:
            cv2.putText(img, "unknown", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 1, cv2.LINE_AA)

        coords = [x, y, w, h]
    return coords


def recognize(img, clf, faceCascade):
    coords = draw_boundary(img, faceCascade, 1.1, 10, (255, 255, 255), "Face", clf)
    return img


faceCascade = cv2.CascadeClassifier("utils/haarcascade_frontalface_default.xml")
clf = cv2.face.LBPHFaceRecognizer_create()
clf.read('runs/run_2/classifier.xml')

cap = cv2.VideoCapture(0)
while True:
    ret, img = cap.read()
    img = recognize(img, clf, faceCascade)
    cv2.imshow("face detect", img)

    if cv2.waitKey(1) == 48:
        break

cap.release()
cv2.destroyAllWindows()
print('completed....')
