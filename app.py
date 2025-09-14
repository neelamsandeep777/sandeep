from flask import Flask, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import numpy as np
import math
import time
import json
import atexit
import signal
import sys
from cvzone.HandTrackingModule import HandDetector

app = Flask(__name__)

# Initialize hand detector
detector = HandDetector(maxHands=2)

# Global variables
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils
offset = 15
imgSize = 200

# Initialize camera once
camera = None
for i in [0, 1, 2]:
    camera = cv2.VideoCapture(i)
    if camera.isOpened():
        break

def get_camera():
    global camera
    if camera is None or not camera.isOpened():
        for i in [0, 1, 2]:
            camera = cv2.VideoCapture(i)
            if camera.isOpened():
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                break
    return camera

def cleanup_camera():
    global camera
    if camera is not None and camera.isOpened():
        camera.release()
        cv2.destroyAllWindows()
        print("Camera released successfully")

@app.route('/shutdown', methods=['POST'])
def shutdown():
    cleanup_camera()
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    return 'Server shutting down...'

def signal_handler(sig, frame):
    cleanup_camera()
    sys.exit(0)

# ASL Alphabet variables
asl_text_output = ""
asl_last_detection_time = 0
asl_last_detected_index = -1

# Simple Gestures variables
gesture_text_output = ""
gesture_last_detection_time = 0
gesture_last_detected_index = -1
gesture_added_flag = False

hold_duration = 2.0
current_mode = "asl"  # "asl" or "gestures"

asl_labels = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", 
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
]

gesture_labels = ["Thumbs up", "Peace", "OK sign", "Stop", "Point", "Rock on", "Fist", "Open hand", "Love you", "Fuck", "Thank you"]

def asl_alphabet_recognition(landmarks):
    if not landmarks or len(landmarks) < 21:
        return 0
    
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
    
    # Simple ASL mapping
    if finger_count == 0:
        return 0   # A
    elif finger_count == 1:
        if fingers[1] == 1:  # index only
            return 3   # D
        elif fingers[4] == 1:  # pinky only
            return 8   # I
        else:
            return 6   # G
    elif finger_count == 2:
        if fingers[1] == 1 and fingers[2] == 1:  # index + middle
            return 21  # V
        elif fingers[0] == 1 and fingers[1] == 1:  # thumb + index
            return 11  # L
        elif fingers[0] == 1 and fingers[4] == 1:  # thumb + pinky
            return 24  # Y
        else:
            return 20  # U
    elif finger_count == 3:
        if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1:
            return 22  # W
        else:
            return 5   # F
    elif finger_count == 4:
        return 1   # B
    else:
        return 7   # H

def simple_gesture_recognition(landmarks):
    if not landmarks or len(landmarks) < 21:
        return 6
    
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
    
    if finger_count == 0:
        return 6  # Fist
    elif finger_count == 1:
        if fingers[0] == 1:
            return 0  # Thumbs up
        elif fingers[1] == 1:
            return 4  # Point
        elif fingers[2] == 1:
            return 9  # Middle finger
        else:
            return 4  # Point
    elif finger_count == 2:
        if fingers[1] == 1 and fingers[2] == 1:
            return 1  # Peace
        elif fingers[1] == 1 and fingers[4] == 1:
            return 5  # Rock on
        elif fingers[0] == 1 and fingers[1] == 1:
            return 2  # OK sign
        else:
            return 1  # Peace
    elif finger_count == 3:
        if fingers[0] == 1 and fingers[1] == 1 and fingers[4] == 1:
            return 8  # Love you
        else:
            return 10 # Thank you
    elif finger_count == 4:
        return 3  # Stop
    else:
        return 7  # Open hand

def check_both_hands_fist(hands):
    """Check if both hands are making fist gestures"""
    if len(hands) < 2:
        return False
    
    both_fists = True
    for hand in hands:
        landmarks = hand['lmList']
        if landmarks:
            # Check if all fingers are closed (fist)
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
            
            # If any finger is extended, it's not a fist
            if sum(fingers) > 0:
                both_fists = False
                break
    
    return both_fists

def generate_frames():
    global asl_text_output, asl_last_detection_time, asl_last_detected_index
    global gesture_text_output, gesture_last_detection_time, gesture_last_detected_index
    global current_mode
    
    cap = get_camera()
    frame_count = 0
    
    while True:
        success, frame = cap.read()
        if not success:
            cap.release()
            cap = get_camera()
            continue
        
        # Process every frame for smoother video
        frame = cv2.flip(frame, 1)  # Mirror the image
        
        try:
            hands, frame = detector.findHands(frame)
        except Exception as e:
            hands = []
        current_label = ""
        
        if hands:
            if current_mode == "asl":
                # For ASL, use only the first hand
                hand = hands[0]
                x, y, w, h = hand['bbox']
                landmarks = hand['lmList']
                
                index = asl_alphabet_recognition(landmarks)
                current_label = asl_labels[index]
                
                current_time = time.time()
                if index == asl_last_detected_index:
                    if current_time - asl_last_detection_time >= hold_duration:
                        asl_text_output += asl_labels[index]
                        asl_last_detection_time = current_time
                else:
                    asl_last_detected_index = index
                    asl_last_detection_time = current_time
                
                cv2.rectangle(frame, (x-offset, y-offset-70), (x-offset+300, y-offset+10), (0,255,0), cv2.FILLED)
                cv2.putText(frame, current_label, (x, y-30), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0,0,0), 2)
                cv2.rectangle(frame, (x-offset, y-offset), (x+w+offset, y+h+offset), (0,255,0), 4)
            else:
                # For gestures, check for both hands fist (clear gesture)
                if len(hands) >= 2 and check_both_hands_fist(hands):
                    current_label = "Clear Text"
                    current_time = time.time()
                    if gesture_last_detected_index == -999:  # Special index for clear
                        if current_time - gesture_last_detection_time >= hold_duration:
                            gesture_text_output = ""  # Clear the text
                            gesture_last_detection_time = current_time
                            global gesture_added_flag
                            gesture_added_flag = True
                    else:
                        gesture_last_detected_index = -999
                        gesture_last_detection_time = current_time
                    
                    # Draw rectangles for both hands with clear text label
                    for i, hand in enumerate(hands[:2]):  # Only first two hands
                        x, y, w, h = hand['bbox']
                        cv2.rectangle(frame, (x-offset, y-offset-70), (x-offset+300, y-offset+10), (255,0,0), cv2.FILLED)
                        if i == 0:  # Only show label on first hand
                            cv2.putText(frame, current_label, (x, y-30), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255,255,255), 2)
                        cv2.rectangle(frame, (x-offset, y-offset), (x+w+offset, y+h+offset), (255,0,0), 4)
                else:
                    # Regular single hand gesture recognition
                    hand = hands[0]
                    x, y, w, h = hand['bbox']
                    landmarks = hand['lmList']
                    
                    index = simple_gesture_recognition(landmarks)
                    current_label = gesture_labels[index]
                    
                    current_time = time.time()
                    if index == gesture_last_detected_index:
                        if current_time - gesture_last_detection_time >= hold_duration:
                            gesture_text_output += gesture_labels[index] + " "
                            gesture_last_detection_time = current_time
                            gesture_added_flag = True  # Set flag when gesture is added
                    else:
                        gesture_last_detected_index = index
                        gesture_last_detection_time = current_time
                    
                    # Draw rectangle for single hand
                    cv2.rectangle(frame, (x-offset, y-offset-70), (x-offset+300, y-offset+10), (0,255,0), cv2.FILLED)
                    cv2.putText(frame, current_label, (x, y-30), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0,0,0), 2)
                    cv2.rectangle(frame, (x-offset, y-offset), (x+w+offset, y+h+offset), (0,255,0), 4)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    global current_mode
    current_mode = "asl"  # Automatically switch to ASL mode
    return render_template('index.html')

@app.route('/gestures')
def gestures():
    global current_mode
    current_mode = "gestures"  # Automatically switch to gestures mode
    return render_template('gestures.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_text')
def get_text():
    global gesture_added_flag
    if current_mode == "asl":
        return jsonify({'text': asl_text_output})
    else:
        result = {'text': gesture_text_output, 'new_gesture': gesture_added_flag}
        gesture_added_flag = False  # Reset flag after sending
        return result

@app.route('/clear_text')
def clear_text():
    global asl_text_output, gesture_text_output
    if current_mode == "asl":
        asl_text_output = ""
    else:
        gesture_text_output = ""
    return jsonify({'status': 'cleared'})

@app.route('/switch_mode/<mode>')
def switch_mode(mode):
    global current_mode
    if mode in ["asl", "gestures"]:
        current_mode = mode
        return jsonify({'status': 'switched', 'mode': mode})
    return jsonify({'status': 'error'})

@app.route('/cleanup')
def cleanup():
    cleanup_camera()
    return jsonify({'status': 'camera released'})

if __name__ == '__main__':
    # Register cleanup functions
    atexit.register(cleanup_camera)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("\n=== Sign Language Detection App ===")
    print("Server starting at http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("Or visit http://localhost:5000/cleanup to release camera")
    print("=====================================\n")
    
    try:
        app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
        cleanup_camera()
    finally:
        cleanup_camera()