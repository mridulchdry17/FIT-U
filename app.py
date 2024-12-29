from flask import Flask, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import numpy as np
import base64

app = Flask(__name__)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Global variables
counter = 0
stage = None

def calculate_angles(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360.0 - angle
    
    return angle

@app.route('/process_frame', methods=['POST'])
def process_frame():
    global counter, stage
    
    try:
        # Get the frame data from the request
        frame_data = request.json['frame'].split(',')[1]
        frame_bytes = base64.b64decode(frame_data)
        np_arr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # Process at lower resolution
        frame = cv2.resize(frame, (320, 240))
        
        with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=0,  # Use simplest model
            smooth_landmarks=True  # Enable landmark smoothing
        ) as pose:
            # Process frame
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                angle = calculate_angles(shoulder, elbow, wrist)
                
                # Curl counter logic
                if angle > 160:
                    stage = "down"
                if angle < 30 and stage == 'down':
                    stage = "up"
                    counter += 1
            
            # Optimize landmark drawing
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image, 
                    results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=1, circle_radius=1)
                )
            
            # Highly compress output image
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]  # 30% quality
            _, buffer = cv2.imencode('.jpg', image, encode_param)
            processed_frame = base64.b64encode(buffer).decode('utf-8')
            
            return jsonify({
                'counter': counter,
                'stage': stage,
                'processed_frame': f'data:image/jpeg;base64,{processed_frame}'
            })
            
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/bicep_curl')
def bicep_curl():
    global counter
    counter = 0  # Reset counter when page loads
    return render_template('bicep_curl.html')

@app.route('/reset_counter')
def reset_counter():
    global counter, stage
    counter = 0
    stage = None
    return jsonify({'success': True})

if __name__ == "__main__":
    app.run(debug=True) 