import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("No camera found, but continuing...")
    cap = None

detector = HandDetector(maxHands=1, detectionCon=0.5)
offset = 15
imgSize = 200

# Text display variables
text_output = ""
last_detection_time = 0
hold_duration = 1.5
last_detected_index = -1

# ASL Alphabet only
labels = [
    # ASL Alphabet (0-25)
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
]

def asl_alphabet_recognition(landmarks):
    if not landmarks or len(landmarks) < 21:
        return 0  # Default to "A"
    
    try:
    
        # Simple finger detection
        fingers = []
        
        # Thumb
        if landmarks[4][0] > landmarks[3][0]:
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
    
        # ASL recognition based on chart
        if finger_count == 0:
            # Check thumb position for A vs S vs E
            if landmarks[4][0] > landmarks[3][0] + 15:  # thumb out
                return 0   # A - fist with thumb out
            elif landmarks[4][1] < landmarks[2][1]:  # thumb over fingers
                return 18  # S - fist with thumb over
            else:
                return 4   # E - closed fist
        
        elif finger_count == 1:
            if fingers[1] == 1:  # index finger up
                return 3   # D - index pointing up
            elif fingers[4] == 1:  # pinky up
                return 8   # I - pinky up
            elif fingers[0] == 1:  # thumb up
                return 0   # A variation
            else:
                return 6   # G - index pointing sideways
        
        elif finger_count == 2:
            if fingers[1] == 1 and fingers[2] == 1:  # index + middle
                # Check if spread apart for V or together for U
                dist = abs(landmarks[8][0] - landmarks[12][0])
                if dist > 30:
                    return 21  # V - peace sign
                else:
                    return 20  # U - fingers together
            elif fingers[0] == 1 and fingers[1] == 1:  # thumb + index
                return 11  # L - thumb and index at 90 degrees
            elif fingers[0] == 1 and fingers[4] == 1:  # thumb + pinky
                return 24  # Y - hang loose
            elif fingers[1] == 1 and fingers[4] == 1:  # index + pinky
                return 8   # I variation or rock sign
            else:
                return 7   # H - index and middle horizontal
        
        elif finger_count == 3:
            if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1:  # index+middle+ring
                return 22  # W - three fingers up
            elif fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1:  # middle+ring+pinky
                return 5   # F - thumb and index circle, others up
            else:
                return 22  # W default
        
        elif finger_count == 4:
            if fingers[0] == 0:  # thumb folded
                return 1   # B - four fingers up, thumb folded
            else:
                return 4   # E variation
        
        elif finger_count == 5:
            return 2   # C - all fingers extended (open hand)
        
        # Special cases for letters that need thumb-finger contact
        thumb_index_dist = abs(landmarks[4][0] - landmarks[8][0]) + abs(landmarks[4][1] - landmarks[8][1])
        if thumb_index_dist < 25 and finger_count >= 3:
            return 14  # O - thumb touching index, others up
        
        # Additional mappings for remaining letters
        if finger_count == 0 and landmarks[4][1] > landmarks[8][1]:  # thumb below
            return 13  # N - thumb under first two fingers
        elif finger_count == 0 and landmarks[4][0] < landmarks[8][0]:  # thumb left
            return 12  # M - thumb under first three fingers
    
    except Exception as e:
        return 0  # Default to A on error
    


print("Starting ASL recognition... Press 'q' to quit, 'c' to clear text")
print(f"Camera status: {cap.isOpened() if cap else 'No camera'}")

try:
    while True:
    if cap is None or not cap.isOpened():
        # Create dummy frame if no camera
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, 'No Camera - Press Q to quit', (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        success = True
    else:
        success, img = cap.read()
        if not success:
            continue
    imgOutput = img.copy()
    
    # Create text display page
    text_page = np.ones((600, 800, 3), np.uint8) * 255
    cv2.putText(text_page, "ASL Text Output:", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(text_page, text_output, (50, 150), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 3)
    cv2.putText(text_page, "Hold gesture for 2 seconds to add letter", (50, 550), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
    
    if cap is not None and success:
        try:
            hands, img = detector.findHands(img, draw=True)
        except Exception as e:
            print(f"Hand detection error: {e}")
            hands = []
    else:
        hands = []
    
    if hands:
        try:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            landmarks = hand['lmList']
        except (KeyError, IndexError) as e:
            print(f"Hand data error: {e}")
            continue

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

            try:
                index = asl_alphabet_recognition(landmarks)
            except Exception as e:
                print(f"Recognition error: {e}")
                index = 0
            
            # Check if same gesture is held for 2 seconds
            current_time = time.time()
            if index == last_detected_index:
                if current_time - last_detection_time >= hold_duration:
                    text_output += labels[index]
                    last_detection_time = current_time
            else:
                last_detected_index = index
                last_detection_time = current_time
            
            cv2.rectangle(imgOutput,(x-offset,y-offset-70),(x -offset+300, y - offset+60-50),(0,255,0),cv2.FILLED)  
            cv2.putText(imgOutput,labels[index],(x,y-30),cv2.FONT_HERSHEY_COMPLEX,1.5,(0,0,0),2) 
            cv2.rectangle(imgOutput,(x-offset,y-offset),(x + w + offset, y+h + offset),(0,255,0),4)   

            cv2.imshow('ImageCrop', imgCrop)
            cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', imgOutput)
    cv2.imshow('Text Output', text_page)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        text_output = ""

except Exception as e:
    print(f"Error occurred: {e}")
    input("Press Enter to close...")

finally:
    if cap:
        cap.release()
    cv2.destroyAllWindows()