from flask import Flask, render_template, Response, redirect, url_for
import cv2
import numpy as np
import mediapipe as mp

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
cap = None

def calculate_angles(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360.0 - angle

    return angle


# Generate frames for bicep curl detection
def generate_frame_bicep():
    global counter, stage

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Unable to access the camera.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Recolor images
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor Back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks and perform angle calculations
            try:
                landmarks = results.pose_landmarks.landmark
                shoulder = [
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
                ]
                elbow = [
                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
                ]
                wrist = [
                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
                ]

                angle = calculate_angles(shoulder, elbow, wrist)

                # Curl counter logic
                if angle > 160:
                    stage = "down"
                if angle < 30 and stage == "down":
                    stage = "up"
                    counter += 1
            except:
                pass

            # Render counter and stage on the left side of the frame
            cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)
            
            # Rep data 
            cv2.putText(image, 'REPS', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE', (65, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Render pose landmarks
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

            # Encode the frame
            ret, buffer = cv2.imencode(".jpg", image)
            frame = buffer.tobytes()
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
    finally:
        if cap:
            cap.release()
        cv2.destroyAllWindows()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/bicep_curl")
def bicep_curl():
    global counter
    counter = 0
    return render_template("bicep_curl.html")


@app.route("/video_feed_bicep")
def video_feed_bicep():
    return Response(generate_frame_bicep(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route('/reset_counter')
def reset_counter():
    global counter
    counter = 0
    return redirect(url_for('bicep_curl'))


if __name__ == "__main__":
    app.run(debug=False)  # debug = False since it's in production now
