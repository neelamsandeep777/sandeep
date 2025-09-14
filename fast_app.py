from flask import Flask, render_template, Response, jsonify
import cv2
from cvzone.HandTrackingModule import HandDetector
import time

app = Flask(__name__)

detector = HandDetector(maxHands=1, detectionCon=0.4)
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

text_output = ""
last_time = 0
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

def detect_gesture(landmarks):
    if not landmarks:
        return 0
    fingers = []
    if landmarks[4][0] > landmarks[3][0]:
        fingers.append(1)
    else:
        fingers.append(0)
    for tip, pip in [(8,6), (12,10), (16,14), (20,18)]:
        if landmarks[tip][1] < landmarks[pip][1]:
            fingers.append(1)
        else:
            fingers.append(0)
    return sum(fingers)

def generate_frames():
    global text_output, last_time
    frame_count = 0
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        frame_count += 1
        if frame_count % 3 != 0:  # Process every 3rd frame
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            continue
            
        hands, frame = detector.findHands(frame)
        
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            landmarks = hand['lmList']
            
            index = detect_gesture(landmarks)
            current_time = time.time()
            
            if current_time - last_time > 1.5:
                text_output += labels[index % len(labels)]
                last_time = current_time
            
            cv2.rectangle(frame, (x-10, y-40), (x+100, y-10), (0,255,0), -1)
            cv2.putText(frame, labels[index % len(labels)], (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
            cv2.rectangle(frame, (x-10, y-10), (x+w+10, y+h+10), (0,255,0), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return '''<!DOCTYPE html>
<html><head><title>Fast Sign Detection</title></head>
<body style="text-align:center;font-family:Arial;">
<h2>Sign Language Detection</h2>
<img src="/video_feed" style="border:2px solid #333;">
<div id="text" style="font-size:24px;margin:20px;"></div>
<button onclick="clearText()">Clear</button>
<script>
setInterval(()=>{
fetch('/get_text').then(r=>r.json()).then(d=>document.getElementById('text').innerText=d.text)
},500);
function clearText(){fetch('/clear_text')}
</script>
</body></html>'''

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_text')
def get_text():
    return jsonify({'text': text_output})

@app.route('/clear_text')
def clear_text():
    global text_output
    text_output = ""
    return jsonify({'status': 'cleared'})

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)