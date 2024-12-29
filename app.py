from flask import Flask, render_template, Response, redirect, url_for, jsonify, request
import cv2
import numpy as np
import mediapipe as mp
import os
from io import BytesIO

# Flask app setup
app = Flask(__name__)

# Mediapipe and OpenCV Setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Global variables for pose detection
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Variables for bicep curl counter
counter = 0
stage = None

def calculate_angles(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360.0 - angle

    return angle

@app.route("/process_frame", methods=['POST'])
def process_frame():
    global counter, stage
    
    if 'frame' not in request.files:
        return jsonify({'error': 'No frame provided'}), 400

    try:
        # Read frame from request
        frame_file = request.files['frame']
        frame_bytes = frame_file.read()
        
        # Convert to numpy array
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Process frame
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Calculate angle
            angle = calculate_angles(shoulder, elbow, wrist)

            # Curl counter logic
            if angle > 160:
                stage = "down"
            if angle < 30 and stage == "down":
                stage = "up"
                counter += 1

            return jsonify({
                'reps': counter,
                'stage': stage,
                'angle': angle
            })

        return jsonify({
            'reps': counter,
            'stage': stage
        })

    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/bicep_curl")
def bicep_curl():
    global counter
    counter = 0
    return render_template("bicep_curl.html")

@app.route('/reset_counter')
def reset_counter():
    global counter
    counter = 0
    return jsonify({'reps': counter})

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
