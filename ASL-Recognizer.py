import cv2
import numpy as np
import HandTrackingModule as htm
import time

# --- Camera Setup ---
wCam, hCam = 1280, 720
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()
cap.set(3, wCam)
cap.set(4, hCam)
# --- End Camera Setup ---

# Initialize Hand Detector (only for one hand)
detector = htm.handDetector(maxHands=1, detectionCon=0.75)

# --- Variables for letter display ---
current_letter = ""
display_letter = ""
last_detection_time = 0
display_duration = 1.5 # seconds

while True:
    success, img = cap.read()
    if not success: break
    img = cv2.flip(img, 1) # Flip image horizontally for a mirror view

    # Find hand and its landmarks
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=False)

    # --- Letter Recognition Logic ---
    if len(lmList) != 0:
        fingers = detector.fingersUp()
        current_letter = "Unknown" # Default

        # --- RULE-BASED LETTER RECOGNITION (ORDER IS IMPORTANT) ---

        # Rule for 'A' & 'S': Fist shapes
        if fingers == [0, 0, 0, 0, 0]:
            # 'A' has thumb on the side, 'S' has thumb over the fingers
            thumb_tip = lmList[4]
            index_mcp = lmList[5] # Base of index finger
            if thumb_tip[1] > index_mcp[1]:
                current_letter = "A"
            else:
                current_letter = "S"

        # Rule for 'B': Open hand, thumb tucked in
        elif fingers == [0, 1, 1, 1, 1]:
            current_letter = "B"

        # Rule for 'C' & 'O' & 'E': Curved hand shapes
        elif fingers == [0, 0, 0, 0, 0] and not (lmList[4][1] > lmList[5][1]):
            # Check distance between thumb and index finger for 'O'
            dist_thumb_index, _, _ = detector.findDistance(4, 8)
            if dist_thumb_index < 40:
                current_letter = "O"
            else:
                # Differentiate 'C' and 'E'
                dist_thumb_middle, _, _ = detector.findDistance(4, 12)
                if dist_thumb_middle < 40:
                    current_letter = "E"
                else:
                    current_letter = "C"

        # Rule for 'D': Index finger up
        elif fingers == [0, 1, 0, 0, 0]:
            current_letter = "D"

        # Rule for 'F': "OK" sign
        elif fingers == [0, 0, 1, 1, 1]:
            current_letter = "F"

        # Rule for 'G' & 'H': Pointing sideways
        elif fingers == [0, 1, 0, 0, 0] and detector.is_horizontal(8):
            current_letter = "G"
        elif fingers == [0, 1, 1, 0, 0] and detector.is_horizontal(8) and detector.is_horizontal(12):
            current_letter = "H"

        # Rule for 'I': Pinky up
        elif fingers == [0, 0, 0, 0, 1]:
            current_letter = "I"

        # Rule for 'K', 'P', 'V': Two fingers up
        elif fingers == [0, 1, 1, 0, 0]:
            thumb_tip = lmList[4]
            middle_pip = lmList[10]
            # 'K'/'P' has thumb touching middle finger
            if thumb_tip[2] < middle_pip[2]:
                current_letter = "K" # or P
            else:
                current_letter = "V"

        # Rule for 'L': Thumb and Index up
        elif fingers == [1, 1, 0, 0, 0]:
            current_letter = "L"

        # Rule for 'M' & 'N': Fist with fingers over thumb
        elif fingers == [0, 0, 0, 0, 0] and lmList[4][1] < lmList[5][1]:
            # Check how many knuckles are visible
            if lmList[8][2] > lmList[6][2] and lmList[12][2] > lmList[10][2]:
                current_letter = "M"
            else:
                current_letter = "N"

        # Rule for 'Q': Like 'G' but pointing down
        elif fingers == [0, 1, 0, 0, 0] and lmList[8][2] > lmList[6][2]:
            current_letter = "Q"

        # Rule for 'R': Crossed fingers
        elif fingers == [0, 1, 1, 0, 0] and lmList[8][1] > lmList[12][1]:
            current_letter = "R"

        # Rule for 'T': Fist with thumb tucked between index and middle
        elif fingers == [0, 0, 0, 0, 0] and lmList[4][1] < lmList[6][1] and lmList[4][1] > lmList[10][1]:
            current_letter = "T"

        # Rule for 'U': Two fingers up and together
        elif fingers == [0, 1, 1, 0, 0]:
            dist, _, _ = detector.findDistance(8, 12)
            if dist < 40:
                current_letter = "U"

        # Rule for 'W': Three fingers up
        elif fingers == [0, 1, 1, 1, 0]:
            current_letter = "W"

        # Rule for 'X': Bent index finger
        elif fingers == [0, 1, 0, 0, 0] and lmList[8][2] > lmList[7][2]:
            current_letter = "X"

        # Rule for 'Y': "Call me" sign
        elif fingers == [1, 0, 0, 0, 1]:
            current_letter = "Y"

        # --- Display Logic ---
        if current_letter != "Unknown":
            if current_letter != display_letter:
                last_detection_time = time.time()
                display_letter = current_letter

        if time.time() - last_detection_time < display_duration:
            cv2.putText(img, display_letter, (bbox[0], bbox[1] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    # --- UI Drawing ---
    cv2.rectangle(img, (20, 20), (200, 120), (0, 0, 0), cv2.FILLED)
    cv2.putText(img, display_letter, (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)

    cv2.imshow("ASL Recognizer - Bawb", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
