import numpy as np
import pickle
import cv2
# sadece frontal lob'a gore detect ediyor ayni anda birden fazla haarcasscade
# kullabilriiz.
face_cascade = cv2.CascadeClassifier(
    'cascades/data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(
    'cascades\data\haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier("cascades\data\haarcascade_smile.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels = {}
with open("labels.pickle", "rb") as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}
cap = cv2.VideoCapture(0)
while(True):
    # Capture frame-by-frame

    ret, frame = cap.read()
    # Display the resulting frame

    # Gray olarak algiliyor ama goruntuyu gosterirken renkli gosteriyor.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.5, minNeighbors=5)
    for(x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        id_, conf = recognizer.predict(roi_gray)
        if conf >= 45:  # and conf <= 85:
            print(conf)
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x, y), font, 1,
                        color, stroke, cv2.LINE_AA)
        # print(x, y, w, h)

        # recognize ?
        # OpenCV has the ability to train a recognizer
        # deep learned model predict keras , tenserflow , pytorch
        # different methods on how we could identify who this person is

        color = (255, 0, 0)  # BGR BLUE GREEN RED
        stroke = 2
        en_cord_x = x+w
        en_cord_y = y+h
        # starting and ending coordinate
        cv2.rectangle(frame, (x, y), (en_cord_x, en_cord_y), color, stroke)
        subitems = smile_cascade.detectMultiScale(roi_gray)
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for(ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        img_item = 'my-image.png'
        cv2.imwrite(
            img_item, roi_color)
    cv2.imshow('frame', frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture

cap.release()
cv2.destroyAllWindows()
