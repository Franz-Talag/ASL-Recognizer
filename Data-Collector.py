import cv2
import os
import time
import HandTrackingModule as htm
import numpy as np
import math

# --- Parameters ---
DATA_DIR = "asl_data"
NUM_CLASSES = 26  # A-Z
IMAGES_PER_CLASS = 200
SLEEP_TIME = 0.25

# --- Folder Setup ---
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# --- Camera and Hand Detector Setup ---
cap = cv2.VideoCapture(0)
detector = htm.handDetector(maxHands=1)

print("Starting Data Collection...")

for j in range(NUM_CLASSES):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f"Collecting data for Class {chr(j + 65)} (Number {j})")

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"Ready? Press 'S' to start collecting for {chr(j + 65)}", 
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Data Collector", frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

    counter = 0
    while counter < IMAGES_PER_CLASS:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        
        img = detector.findHands(frame.copy(), draw=False)
        lmList, bbox = detector.findPosition(img, draw=False)

        if lmList:
            x, y, w, h = bbox
            
            img_size = 400
            img_canvas = np.ones((img_size, img_size, 3), np.uint8) * 255

            y1 = max(0, y - 20)
            y2 = min(frame.shape[0], y + h + 20)
            x1 = max(0, x - 20)
            x2 = min(frame.shape[1], x + w + 20)
            img_crop = frame[y1:y2, x1:x2]

            if img_crop.shape[0] > 0 and img_crop.shape[1] > 0:
                aspect_ratio = img_crop.shape[1] / img_crop.shape[0]

                if aspect_ratio > 1:
                    k = img_size / img_crop.shape[1]
                    h_cal = math.ceil(k * img_crop.shape[0])
                    img_resize = cv2.resize(img_crop, (img_size, h_cal))
                    h_gap = math.ceil((img_size - h_cal) / 2)
                    img_canvas[h_gap:h_cal + h_gap, :] = img_resize
                else:
                    k = img_size / img_crop.shape[0]
                    w_cal = math.ceil(k * img_crop.shape[1])
                    img_resize = cv2.resize(img_crop, (w_cal, img_size))
                    w_gap = math.ceil((img_size - w_cal) / 2)
                    img_canvas[:, w_gap:w_cal + w_gap] = img_resize

                save_path = os.path.join(class_dir, f'{time.time()}.jpg')
                cv2.imwrite(save_path, img_canvas)

                counter += 1
                cv2.putText(frame, f"Collecting: {counter}/{IMAGES_PER_CLASS}", 
                            (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Data Collector", frame)
        cv2.waitKey(1)
        time.sleep(SLEEP_TIME)

cap.release()
cv2.destroyAllWindows()
