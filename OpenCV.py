import cv2
import os
from datetime import datetime
import numpy as np

# Auto-create folders if they don't exist
os.makedirs("screenshots", exist_ok=True)
os.makedirs("cropped_faces", exist_ok=True)

# Base path (your project directory)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Haarcascade paths
face_cascade_path = os.path.join(BASE_DIR, 'cascades', 'haarcascade_frontalface_default.xml')
eye_cascade_path = os.path.join(BASE_DIR, 'cascades', 'haarcascade_eye.xml')

# Load cascades
face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

# Check if cascades loaded properly
if face_cascade.empty() or eye_cascade.empty():
    print("[ERROR] Haarcascade files not loaded properly.")
    exit()

# Start webcam
cap = cv2.VideoCapture(0)

# Get initial frame for motion detection
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detect eyes in face region
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20))

        # Draw rectangles around eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        # Eye status based on number of eyes detected
        if len(eyes) >= 2:
            eye_status = "Eyes Open"
            color = (0, 255, 0)
        else:
            eye_status = "Eyes Closed"
            color = (0, 0, 255)

        # Put text above face rectangle
        cv2.putText(frame, eye_status, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Motion Detection
    gray_blur = cv2.GaussianBlur(gray, (21, 21), 0)
    frame_delta = cv2.absdiff(prev_gray, gray_blur)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    motion_detected = np.sum(thresh) > 50000  # threshold value, tune if needed

    if motion_detected:
        cv2.putText(frame, "Motion Detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    prev_gray = gray_blur

    # Show the frame
    cv2.imshow('Webcam - Face/Eye/Motion Detection', frame)

    key = cv2.waitKey(1)

    # Save screenshot and face on pressing 's'
    if key == ord('s'):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        cv2.imwrite(f'screenshots/snap_{timestamp}.jpg', frame)

        # Save cropped faces
        for (x, y, w, h) in faces:
            face_crop = frame[y:y+h, x:x+w]
            cv2.imwrite(f'cropped_faces/face_{timestamp}.jpg', face_crop)

        print(f"[INFO] Screenshot and face(s) saved at {timestamp}")

    # Quit on pressing 'q'
    if key == ord('q'):
        break

# Release everything
cap.release()
cv2.destroyAllWindows()

