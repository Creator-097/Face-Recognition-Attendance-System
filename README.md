# Face-Recognition-Attendance-System
This project is a real-time face recognition attendance system built using Python 3.10, OpenCV, and the face_recognition library. It features a dark-themed GUI built with Tkinter, enabling multi-face detection from a webcam. Attendance is automatically logged into a CSV file with timestamps, and unknown faces are flagged and saved.

# AI-Based Face Recognition Attendance System with Dark-Themed GUI

A real-time **face recognition-based attendance system** using Python, OpenCV, and face_recognition library. Features a **dark-themed GUI** with Tkinter, multi-face detection, CSV logging, and unknown face alerts.

---

## Features

- Real-time multi-face recognition from webcam  
- Automated attendance logging in `Attendance.csv`  
- Unknown face detection and saving in `unknown_faces/`  
- Dark-themed Tkinter GUI with start/stop buttons  
- Pop-up “Thank You” message when attendance is marked  

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/<your-username>/Face-Recognition-Attendance-GUI.git
cd Face-Recognition-Attendance-GUI

2.Install required Python packages:
pip install -r requirements.txt
Requirements include: OpenCV, NumPy, face_recognition, Pillow

##Usage

1.Add known face images to the images/ folder.
2.Name the images as the person’s name (e.g., alice.jpg, bob.jpg).
3.Run the GUI:
4.Click “Mark Attendance” to start recognition.
5.Click “Marked” to stop the webcam safely.
6.Attendance will be saved automatically in Attendance.csv.


##Folder Structure
Face-Recognition-Attendance-GUI/
│
├── images/                # Known face images
├── unknown_faces/         # Saved unknown faces
├── main_gui_dark_attendance.py
├── Attendance.csv
├── requirements.txt


