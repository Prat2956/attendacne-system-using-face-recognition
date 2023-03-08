import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)
Modi_img = face_recognition.load_image_file("Images_Attendance/Modi.png")
Modi_encoding = face_recognition.face_encodings(Modi_img)[0]

srk_img = face_recognition.load_image_file("Images_Attendance/srk.jpg")
srk_encoding = face_recognition.face_encodings(srk_img)[0]

prat_img = face_recognition.load_image_file("Images_Attendance/prat.jpg")
prat_encoding = face_recognition.face_encodings(prat_img)[0]

known_face_encoding = [
    Modi_encoding,
    srk_encoding,
    prat_encoding
]
known_faces_names = [" Narendra Modi", "shahrukh Khan", "Pratinav Kumar"]
students = known_faces_names.copy()
face_locations = []
face_encodings = []
face_names = []
s = True

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(current_date+'.csv','w+', newline = '')
inwriter = csv.writer(f)
while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encodings in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding, face_encodings)
            name = ""
            face_distance = face_recognition.face_distance(known_face_encoding, face_encodings)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_faces_names[best_match_index]

            face_names.append(name)
            if name in known_faces_names:
                if name in students:
                    students.remove(name)
                    print(students)
                    current_time = now.strftime("%H-%M-%S")
                    inwriter.writerow([name, current_time])

    cv2.imshow("attendance system", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllwindows()
f.close()
