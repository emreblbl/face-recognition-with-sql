import numpy as np
import cv2
import sqlite3
import os
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
detector = cv2.CascadeClassifier(
    "cascades\data\haarcascade_frontalface_default.xml")


def insertOrUpdate(Id, Name):
    conn = sqlite3.connect("FaceBase.db")
    cmd = "SELECT * FROM People WHERE ID="+str(Id)
    cursor = conn.execute(cmd)
    isRecordExist = 0
    for row in cursor:
        isRecordExist = 1
    if(isRecordExist == 1):
        cmd = "UPDATE people SET Name=' "+str(name)+" ' WHERE ID="+str(Id)

    else:
        cmd = "INSERT INTO people(ID,Name) Values(" + \
            str(Id)+",' "+str(name)+" ' )"

    conn.execute(cmd)
    conn.commit()
    conn.close()


id = input('enter user id')
name = input('enter your name')
insertOrUpdate(id, name)
dirName = "D:\\programlama-dilleri\\FaceRecognition\\OpenCV-Python-Series-master\\src\\images\\" + name


if not os.path.exists(dirName):
    # exit
    os.makedirs(dirName, exist_ok=True)
sampleNum = 0

while(True):
    ret, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector.detectMultiScale(gray, 1.5, 5)
    for (x, y, w, h) in faces:
        roi_color = img[y:y+h, x:x+w]
        sampleNum = sampleNum+1
        # cv2.imwrite("D:\\programlama-dilleri\\Face-recognization-with-SQLite-Database-master\\Face-recognization-with-SQLite-Database-master\\dataSet\\"+str(id)+"." +
        #             str(sampleNum)+".jpg", gray[y:y+h, x:x+w])
        cv2.imwrite("D:\\programlama-dilleri\\FaceRecognition\\OpenCV-Python-Series-master\\src\\images\\"+str(name)+"\\"+str(id)+"." +
                    str(sampleNum)+".jpg", roi_color)
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.waitKey(100)

    cv2.imshow('Face', img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    # cv2.waitKey(1)
    if(sampleNum > 20):
        break

cap.release()
cv2.destroyAllWindows()
