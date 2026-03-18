import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk


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

window = tk.Tk()
window.title("Face Recognition Attendance System")
window.geometry("900x650")
window.configure(bg="#2E2E2E")
window.resizable(False, False)

video_frame = tk.Frame(window, bg="#1C1C1C", bd=2, relief=tk.SUNKEN)
video_frame.pack(pady=20)
video_label = tk.Label(video_frame, bg="#1C1C1C")
video_label.pack()


cap = None
running = False

def start_recognition():
    global running, cap
    if not running:
        running = True
        cap = cv2.VideoCapture(0)
        recognize_faces()
        btn_start.config(state="disabled")
        btn_stop.config(state="normal")

def stop_recognition():
    global running, cap
    running = False
    if cap is not None:
        cap.release()
        cap = None
    video_label.config(image='')
    btn_start.config(state="normal")
    btn_stop.config(state="disabled")
    print("Recognition stopped and webcam released.")

def recognize_faces():
    global running, cap
    if not running or cap is None:
        return

    success, img = cap.read()
    if not success:
        messagebox.showerror("Error", "Cannot access webcam")
        return

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

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgPIL = Image.fromarray(imgRGB)
    imgtk = ImageTk.PhotoImage(image=imgPIL)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    video_label.after(10, recognize_faces)


btn_frame = tk.Frame(window, bg="#2E2E2E")
btn_frame.pack(pady=10)

btn_start = tk.Button(btn_frame, text="Mark Attendance", command=start_recognition,
                      bg="#4CAF50", fg="white", font=("Arial", 14), width=20)
btn_start.grid(row=0, column=0, padx=10, pady=5)

btn_stop = tk.Button(btn_frame, text="Marked", command=stop_recognition,
                     bg="#F44336", fg="white", font=("Arial", 14), width=20, state="disabled")
btn_stop.grid(row=0, column=1, padx=10, pady=5)

def on_closing():
    global running, cap
    running = False
    if cap is not None:
        cap.release()
    window.destroy()

window.protocol("WM_DELETE_WINDOW", on_closing)

window.mainloop()