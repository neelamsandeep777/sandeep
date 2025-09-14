import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300

labels = ["Thumbs up", "Peace", "OK sign", "Stop", "Point", "Rock on", "Fist", "Open hand", "Love you", "Fuck", "Thank you"]

def simple_gesture_recognition(landmarks):
    if not landmarks:
        return 6
    
    # Get key positions
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    index_tip = landmarks[8]
    index_pip = landmarks[6]
    middle_tip = landmarks[12]
    middle_pip = landmarks[10]
    ring_tip = landmarks[16]
    ring_pip = landmarks[14]
    pinky_tip = landmarks[20]
    pinky_pip = landmarks[18]
    
    # Check if fingers are up
    fingers = []
    
    # Thumb
    if thumb_tip[0] > thumb_ip[0]:
        fingers.append(1)
    else:
        fingers.append(0)
    
    # Other fingers
    for tip, pip in [(8,6), (12,10), (16,14), (20,18)]:
        if landmarks[tip][1] < landmarks[pip][1]:
            fingers.append(1)
        else:
            fingers.append(0)
    
    finger_count = sum(fingers)
    
    # Gesture recognition
    if finger_count == 0:
        return 6  # "Fist"
    elif finger_count == 1:
        if fingers[0] == 1:  # Only thumb
            return 0  # "Thumbs up"
        elif fingers[1] == 1:  # Only index
            return 4  # "Point"
        elif fingers[2] == 1:  # Only middle finger
            return 9  # "Fuck"
    elif finger_count == 2:
        if fingers[1] == 1 and fingers[2] == 1:  # Index and middle
            return 1  # "Peace"
        elif fingers[1] == 1 and fingers[4] == 1:  # Index and pinky
            return 5  # "Rock on"
        elif fingers[0] == 1 and fingers[1] == 1:  # Thumb and index (OK sign check)
            # Check if thumb and index are close (forming circle)
            distance = math.sqrt((thumb_tip[0] - index_tip[0])**2 + (thumb_tip[1] - index_tip[1])**2)
            if distance < 40:
                return 2  # "OK sign"
    elif finger_count == 3:
        if fingers[0] == 1 and fingers[1] == 1 and fingers[4] == 1:  # Thumb, index, pinky
            return 8  # "Love you"
        else:
            return 10  # "Thank you"
    elif finger_count == 4:
        return 3  # "Stop"
    elif finger_count == 5:
        return 7  # "Open hand"
    
    return 7  # Default to "Open hand"

while True:
    success, img = cap.read()
    if not success or img is None:
        print("Camera not found or failed to read")
        break
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        landmarks = hand['lmList']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
        imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]
        
        if imgCrop.size > 0:
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                if wCal > 0:
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize-wCal)/2)
                    if wGap >= 0 and wCal + wGap <= imgSize:
                        imgWhite[:, wGap: wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                if hCal > 0:
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    if hGap >= 0 and hCal + hGap <= imgSize:
                        imgWhite[hGap: hCal + hGap, :] = imgResize

            index = simple_gesture_recognition(landmarks)
            
            cv2.rectangle(imgOutput,(x-offset,y-offset-70),(x -offset+300, y - offset+60-50),(0,255,0),cv2.FILLED)  
            cv2.putText(imgOutput,labels[index],(x,y-30),cv2.FONT_HERSHEY_COMPLEX,1.5,(0,0,0),2) 
            cv2.rectangle(imgOutput,(x-offset,y-offset),(x + w + offset, y+h + offset),(0,255,0),4)   

            cv2.imshow('ImageCrop', imgCrop)
            cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', imgOutput)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()