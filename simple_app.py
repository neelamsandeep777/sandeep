from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)

# Initialize camera
camera = None

def get_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return camera

def generate_frames():
    cap = get_camera()
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Add some text to show it's working
        cv2.putText(frame, 'Sign Language Detection - Camera Working!', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, 'Install compatible mediapipe version to enable hand detection', 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sign Language Detection</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; margin: 50px; }
            .container { max-width: 800px; margin: 0 auto; }
            img { border: 2px solid #333; border-radius: 10px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Sign Language Detection System</h1>
            <p>Camera feed is working. To enable hand detection, resolve the dependency conflicts.</p>
            <img src="{{ url_for('video_feed') }}" width="640" height="480">
            <p>Press Ctrl+C in terminal to stop the server</p>
        </div>
    </body>
    </html>
    '''

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("Starting Flask app...")
    print("Open your browser and go to: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)