import cv2


def generate_DS():
    face_classifier = cv2.CascadeClassifier("utils/haarcascade_frontalface_default.xml")

    def face_cropped(img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face = face_classifier.detectMultiScale(gray_img, 1.3, 5)  # detect face?
        # scaling factor = 1.3
        # Minimum neighbor = 5

        if face == ():
            return None
        for (x, y, w, h) in face:
            cropped_face = img[y:y + h, x:x + w]
        return cropped_face

    cap = cv2.VideoCapture(0)
    id = 3
    img_id = 0

    while True:
        ret, frame = cap.read()
        if face_cropped(frame) is not None:
            img_id += 1
            face = cv2.resize(face_cropped(frame), (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            path = 'dataset/face/user.' + str(id) + '.' + str(img_id) + '.jpg'
            cv2.imwrite(path, face)

            cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # (50, 50) is the point where txt be written
            # font scale = 1
            # thickness = 2
            # (0, 255, 0) Green color

            cv2.imshow('Cropped Face', face)
            if cv2.waitKey(1) == 48 or int(img_id) == 200:
                break

    cap.release()
    cv2.destroyAllWindows()
    print('Collecting sample is completed....')


generate_DS()
