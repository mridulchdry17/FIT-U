from flask import Flask, render_template, Response, redirect, url_for, jsonify
import cv2
import numpy as np
import mediapipe as mp
import os

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


# Generate frames for bicep curl detection
def generate_frame_bicep():
    global counter, stage
    
    # Try different camera indices and device paths
    camera_indexes = [0, 1, '/dev/video0', '/dev/video1']
    cap = None
    
    for index in camera_indexes:
        try:
            cap = cv2.VideoCapture(index)
            if cap and cap.isOpened():
                print(f"Successfully opened camera with index {index}")
                break
        except Exception as e:
            print(f"Failed to open camera with index {index}: {str(e)}")
            continue
    
    if not cap or not cap.isOpened():
        print("Failed to open any camera")
        return jsonify({'error': 'Camera access failed'}), 500

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            try:
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

                # Render counter and stage on the frame
                cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)
                cv2.putText(image, f"Reps: {counter}", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(image, f"Stage: {stage}", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Render pose landmarks
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

            except Exception as e:
                print(f"Error processing frame: {str(e)}")
                continue

            try:
                ret, buffer = cv2.imencode(".jpg", image)
                if not ret:
                    print("Failed to encode frame")
                    continue
                frame = buffer.tobytes()
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            except Exception as e:
                print(f"Error encoding frame: {str(e)}")
                continue

    except Exception as e:
        print(f"Camera stream error: {str(e)}")
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
    return render_template("bicep_curl.html", counter=counter)


@app.route("/video_feed_bicep")
def video_feed_bicep():
    return Response(generate_frame_bicep(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route('/reset_counter')
def reset_counter():
    global counter
    counter = 0
    return redirect(url_for('bicep_curl'))


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
