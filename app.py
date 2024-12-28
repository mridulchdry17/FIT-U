from flask import Flask, render_template, Response, redirect, url_for
import cv2
import mediapipe as mp
import numpy as np
from collections import deque

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Global variables for counter and stage
counter = 0
stage = None

# Mediapipe setup
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Deque for smoothing angles
angle_deque = deque(maxlen=10)  # Store the last 10 angles for smoothing

# Angle calculation function
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Midpoint
    c = np.array(c)  # Endpoint

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# Video feed generator
def generate_frames():
    global counter, stage

    cap = cv2.VideoCapture(0)  # Use the default camera

    # Set up Mediapipe Pose model
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Camera feed not available.")
                break

            frame_height, frame_width, _ = frame.shape

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Process the image and get pose landmarks
            results = pose.process(image)

            # Recolor back to BGR for rendering
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                # Extract landmarks
                landmarks = results.pose_landmarks.landmark
                if not landmarks:
                    print("Landmarks not detected!")
                    continue

                # Get coordinates
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                # Debug: Print coordinates
                print(f"Shoulder: {shoulder}, Elbow: {elbow}, Wrist: {wrist}")

                # Calculate angle
                angle = calculate_angle(shoulder, elbow, wrist)
                angle_deque.append(angle)

                # Smooth angle
                smoothed_angle = sum(angle_deque) / len(angle_deque)
                print(f"Angle: {angle}, Smoothed Angle: {smoothed_angle}")

                # Visualize angle
                elbow_coord = tuple(np.multiply(elbow, [frame_width, frame_height]).astype(int))
                cv2.putText(image, str(int(smoothed_angle)), elbow_coord,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                # Curl counter logic
                if smoothed_angle > 160:
                    stage = "down"
                if smoothed_angle < 30 and stage == "down":
                    stage = "up"
                    counter += 1
                    print(f"Reps: {counter}")

            except Exception as e:
                print(f"Error in detection: {e}")

            # Render the counter and stage on the video feed
            cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)
            cv2.putText(image, 'Reps:', (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter),
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, 'Stage:', (65, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, stage if stage else "None",
                        (65, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/bicep_curl')
def bicep_curl():
    return render_template('bicep_curl.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/reset_counter')
def reset_counter():
    global counter, stage
    counter = 0
    stage = None
    return redirect(url_for('bicep_curl'))

if __name__ == "__main__":
    app.run(debug=True)
