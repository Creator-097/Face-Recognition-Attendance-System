import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


path = 'images'  
images = []
classNames = []

myList = os.listdir(path)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print("Known faces loaded:", classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(images)
print("Face encoding complete.")


if not os.path.exists("Attendance.csv"):
    with open("Attendance.csv", "w") as f:
        f.write("Name,Time\n")


marked_names = set()

def markAttendance(name):
    global marked_names
    if name in marked_names:
        return  

    now = datetime.now()
    dtString = now.strftime('%Y-%m-%d %H:%M:%S')
    with open('Attendance.csv', 'a') as f:
        f.write(f'{name},{dtString}\n')
    marked_names.add(name)
    print(f"Attendance marked: {name} at {dtString}")


if not os.path.exists("unknown_faces"):
    os.makedirs("unknown_faces")


cap = cv2.VideoCapture(0)

print("Press 'q' to quit.")

while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame")
        break


    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

   
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            markAttendance(name)
         
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(img, name, (x1, y2+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        else:
         
            name = "UNKNOWN"
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)
            cv2.putText(img, name, (x1, y2+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
           
            unknown_face_path = f"unknown_faces/unknown_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(unknown_face_path, img[y1:y2, x1:x2])

    cv2.imshow('Face Recognition Attendance', img)

  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
print("Webcam closed successfully!")