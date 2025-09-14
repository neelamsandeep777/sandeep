import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300

labels = ["Hello","I love you","No","Okay","Please","Thank you","Yes"]

def simple_gesture_recognition(landmarks):
    """Simple gesture recognition based on hand landmarks"""
    if not landmarks:
        return 0
    
    # Get key landmark positions
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    
    # Count extended fingers
    fingers = []
    
    # Thumb
    if thumb_tip[0] > landmarks[3][0]:  # Right hand
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
    
    # Simple gesture mapping
    if finger_count == 0:
        return 2  # "No" - closed fist
    elif finger_count == 1 and fingers[1] == 1:
        return 3  # "Okay" - index finger up
    elif finger_count == 2 and fingers[1] == 1 and fingers[2] == 1:
        return 6  # "Yes" - peace sign
    elif finger_count == 3:
        return 1  # "I love you"
    elif finger_count == 5:
        return 0  # "Hello" - open hand
    else:
        return 4  # "Please" - default

while True:
    success, img = cap.read()
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

            # Use simple gesture recognition
            index = simple_gesture_recognition(landmarks)
            
            cv2.rectangle(imgOutput,(x-offset,y-offset-70),(x -offset+400, y - offset+60-50),(0,255,0),cv2.FILLED)  
            cv2.putText(imgOutput,labels[index],(x,y-30),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,0),2) 
            cv2.rectangle(imgOutput,(x-offset,y-offset),(x + w + offset, y+h + offset),(0,255,0),4)   

            cv2.imshow('ImageCrop', imgCrop)
            cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', imgOutput)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()