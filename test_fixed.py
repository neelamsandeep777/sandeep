import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import os

# Disable TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import tensorflow as tf
    # Try to load the model with legacy format
    model = tf.keras.models.load_model("Model/keras_model.h5", compile=False)
    
    # Load labels
    with open("Model/labels.txt", "r") as f:
        labels = [line.strip() for line in f.readlines()]
    
    print("Model loaded successfully!")
    print(f"Labels: {labels}")
    
except Exception as e:
    print(f"Error loading model: {e}")
    # Fallback labels if model fails to load
    labels = ["Hello","I love you","No","Okay","Please","Thank you","Yes"]
    model = None

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read from camera")
        break
        
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]
        
        if imgCrop.size > 0:  # Check if crop is valid
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize-wCal)/2)
                imgWhite[:, wGap: wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap: hCal + hGap, :] = imgResize

            # Try to make prediction if model is loaded
            if model is not None:
                try:
                    # Normalize the image
                    imgArray = np.asarray(imgWhite, dtype=np.float32).reshape(1, imgSize, imgSize, 3)
                    imgArray = (imgArray / 127.5) - 1
                    
                    prediction = model.predict(imgArray, verbose=0)
                    index = np.argmax(prediction)
                    confidence = prediction[0][index]
                    
                    print(f"Prediction: {labels[index]}, Confidence: {confidence:.2f}")
                except Exception as e:
                    print(f"Prediction error: {e}")
                    index = 0
            else:
                index = 0  # Default to first label if no model

            # Draw bounding box and label
            cv2.rectangle(imgOutput,(x-offset,y-offset-70),(x -offset+400, y - offset+60-50),(0,255,0),cv2.FILLED)  
            cv2.putText(imgOutput,labels[index],(x,y-30),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,0),2) 
            cv2.rectangle(imgOutput,(x-offset,y-offset),(x + w + offset, y+h + offset),(0,255,0),4)   

            cv2.imshow('ImageCrop', imgCrop)
            cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', imgOutput)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()