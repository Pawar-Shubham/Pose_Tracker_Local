from waitress import serve
from flask import Flask, render_template, Response, request, redirect, url_for, session
import cv2
import os
from mvp import ExerciseTracker  # Import your MVP code
import random

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # For session management
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

video_capture = None
tracker = None  # Exercise tracker instance
filename = ["push-up_3.mp4","plank_5.mp4","pull up_1.mp4","hammer curl_8.mp4","tricep dips_11.mp4","tricep pushdown_40.mp4"]
exercises = None

def generate_frames():
    global video_capture, tracker
    while video_capture and video_capture.isOpened():
        success, frame = video_capture.read()
        if not success:
            break
        frame = cv2.resize(frame, (640, 500))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = tracker.pose.process(rgb_frame)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            angles,keypoints = tracker.get_angles_from_landmarks(landmarks)  # Extract angles
            tracker.count_reps(frame, angles, result, landmarks, keypoints)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = cv2.resize(frame, (640, 500))
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    global exercises
    exercises = ["Push-up", "Plank", "Pull-up", "Hammer Curl", "Tricep Dip", "Tricep Pull-down"]
    return render_template('index.html', exercises=exercises)

@app.route('/start_exercise', methods=['GET'])
def start_exercise():
    global video_capture, tracker, exercise_id
    exercise_id = int(request.args.get('exercise', 0))  # Get from URL params
    session['exercise_id'] = exercise_id
    tracker = ExerciseTracker(exercise_id=exercise_id)
    # video_capture = cv2.VideoCapture(filename[exercise_id])  # Using local videos
    return render_template('exercise.html', exercise_type = exercises[exercise_id])

@app.route('/start_webcam', methods=['POST'])
def start_webcam():
    global video_capture
    if video_capture:
        video_capture.release()
    video_capture = cv2.VideoCapture(0)  # Start webcam
    return render_template('exercise.html',exercise_type = exercises[exercise_id])

@app.route('/upload_video', methods=['POST'])
def upload_video():
    global video_capture
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    if video_capture:
        video_capture.release()
    video_capture = cv2.VideoCapture(filepath)  # Load uploaded video

    return render_template('exercise.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_exercise', methods=['POST'])
def stop_exercise():
    global video_capture, tracker
    if video_capture:
        video_capture.release()
    reps = tracker.rep_count if tracker else 0
    calories = round(tracker.calories_burned if tracker else 0, 2)
    duration = round(tracker.exercise_duration if tracker else 0,2)
    return render_template('results.html', reps=reps, calories=calories, duration=duration, exercise_type=tracker.exercise_type, sets_completed=random.randint(0,6), heart_rate=98)

if __name__ == '__main__':
    app.run(debug=True) #uncomment to use flask development server
    # serve(app, port=5000) #uncomment if you want to use waitress server