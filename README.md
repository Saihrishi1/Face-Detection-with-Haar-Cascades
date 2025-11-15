# Face Detection using Haar Cascades with OpenCV and Matplotlib

## Aim

To write a Python program using OpenCV to perform the following image manipulations:  
i) Extract ROI from an image.  
ii) Perform face detection using Haar Cascades in static images.  
iii) Perform eye detection in images.  
iv) Perform face detection with label in real-time video from webcam.

## Software Required

- Anaconda - Python 3.7 or above  
- OpenCV library (`opencv-python`)  
- Matplotlib library (`matplotlib`)  
- Jupyter Notebook or any Python IDE (e.g., VS Code, PyCharm)

## Algorithm

### I) Load and Display Images

- Step 1: Import necessary packages: `numpy`, `cv2`, `matplotlib.pyplot`  
- Step 2: Load grayscale images using `cv2.imread()` with flag `0`  
- Step 3: Display images using `plt.imshow()` with `cmap='gray'`

### II) Load Haar Cascade Classifiers

- Step 1: Load face and eye cascade XML files 
### III) Perform Face Detection in Images

- Step 1: Define a function `detect_face()` that copies the input image  
- Step 2: Use `face_cascade.detectMultiScale()` to detect faces  
- Step 3: Draw white rectangles around detected faces with thickness 10  
- Step 4: Return the processed image with rectangles  

### IV) Perform Eye Detection in Images

- Step 1: Define a function `detect_eyes()` that copies the input image  
- Step 2: Use `eye_cascade.detectMultiScale()` to detect eyes  
- Step 3: Draw white rectangles around detected eyes with thickness 10  
- Step 4: Return the processed image with rectangles  

### V) Display Detection Results on Images

- Step 1: Call `detect_face()` or `detect_eyes()` on loaded images  
- Step 2: Use `plt.imshow()` with `cmap='gray'` to display images with detected regions highlighted  

### VI) Perform Face Detection on Real-Time Webcam Video

- Step 1: Capture video from webcam using `cv2.VideoCapture(0)`  
- Step 2: Loop to continuously read frames from webcam  
- Step 3: Apply `detect_face()` function on each frame  
- Step 4: Display the video frame with rectangles around detected faces  
- Step 5: Exit loop and close windows when ESC key (key code 27) is pressed  
- Step 6: Release video capture and destroy all OpenCV windows

## Code :

```py

import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread(r"C:\Users\admin\Pictures\Camera Roll\WIN_20251115_09_50_31_Pro.jpg", 0)
plt.imshow(img, cmap='gray')
plt.title("Image")
plt.show()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def detect_face(img):
    image_copy = img.copy()
    faces = face_cascade.detectMultiScale(image_copy, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(image_copy, (x, y), (x+w, y+h), (255, 255, 255), 10)
    return image_copy

def detect_eyes(img):
    image_copy = img.copy()
    eyes = eye_cascade.detectMultiScale(image_copy, 1.3, 5)
    for (x, y, w, h) in eyes:
        cv2.rectangle(image_copy, (x, y), (x+w, y+h), (255, 255, 255), 10)
    return image_copy

face_detected = detect_face(img)
plt.imshow(face_detected, cmap='gray')
plt.title("Face Detection")
plt.show()

eyes_detected = detect_eyes(img)
plt.imshow(eyes_detected, cmap='gray')
plt.title("Eyes Detection")
plt.show()

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
        cv2.putText(frame, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.imshow('Face Detection', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()

```

## Output :

<img width="524" height="332" alt="Screenshot 2025-11-15 102035" src="https://github.com/user-attachments/assets/996c347f-d196-46ca-9e5b-ebf8ecd4d063" /><br>
<img width="535" height="330" alt="Screenshot 2025-11-15 102047" src="https://github.com/user-attachments/assets/025d60f2-5fd3-45a6-9ee0-b97939578e9d" /><br>
<img width="526" height="338" alt="Screenshot 2025-11-15 102058" src="https://github.com/user-attachments/assets/eb69bffb-c357-4460-9a3d-adaccac4f77f" /><br>
<img width="640" height="480" alt="Face Detection_screenshot_15 11 2025" src="https://github.com/user-attachments/assets/9f99ac30-9ea8-4af7-a475-c680fb8fd65d" />

## Result :

Thus we have successfully used Haar Cascade with OpenCV and Matplotlib for Face Detection.
