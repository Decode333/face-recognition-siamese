# siamese_flask_app.py

import os
import cv2
import uuid
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, render_template, Response, request, redirect, url_for
from siamese_model import train, collect_images, realtime,DATA_DIR  # Import functions from siamese_model.py

# L1 Distance class (needed for model load)
class L1Dist(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

# Flask App
app = Flask(__name__)
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
model = load_model(r"E:\face recognition data\MODEL WEIGHTS\siamese_model.h5", custom_objects={"L1Dist": L1Dist})
# Load class embeddings and class names
class_embeddings = np.load(r"E:\face recognition data\class_embeddings.npy", allow_pickle=True).item()
class_names = np.load(r"E:\face recognition data\class_names.npy")

# Define current_frame globally
current_frame = None

# Generate video feed for real-time detection
import time  # Add this import at the top of the file

"""
def generate(save=False, class_name=None, num_images=0):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow backend
    if not cap.isOpened():
        print("[ERROR] Unable to access the camera.")
        return

    count = 0
    failure_count = 0  # Counter for consecutive frame grab failures
    max_failures = 10  # Maximum allowed consecutive failures
    class_path = None

    # Create the class directory if saving images
    if save and class_name:
        class_path = os.path.join(r"E:\face recognition data\Class", class_name)
        os.makedirs(class_path, exist_ok=True)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[ERROR] Failed to grab frame.")
                failure_count += 1
                if failure_count >= max_failures:
                    print("[ERROR] Maximum frame grab failures reached. Exiting...")
                    break
                continue  # Skip this iteration if the frame is invalid

            # Reset failure count on successful frame grab
            failure_count = 0

            # Save the frame if 'save' is True and the count is less than num_images
            if save and class_name and count < num_images:
                if frame is not None:  # Ensure the frame is valid
                    imgname = os.path.join(class_path, '{}.jpg'.format(uuid.uuid1()))
                    cv2.imwrite(imgname, frame)  # Save the original frame
                    print(f"Saved {imgname}")
                    count += 1

            # Stop the loop if the required number of images is collected
            if save and count >= num_images:
                break

            # Encode the frame as JPEG for streaming
            _, encoded_frame = cv2.imencode('.jpg', frame)
            encoded_frame = encoded_frame.tobytes()

            # Yield the encoded frame to the browser
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + encoded_frame + b'\r\n')

            # Add a small delay to prevent overloading the camera
            time.sleep(0.1)  # 100ms delay
    finally:
        cap.release()

"""
# Function to generate video feed for real-time detection
def generate(save=False, class_name=None, num_images=0):
    
    global current_frame  # Declare current_frame as global
    
    face_cascade = cv2.CascadeClassifier(r"E:\face recognition data\haarcascade_frontalface_default.xml")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow backend
    if not cap.isOpened():
        print("[ERROR] Unable to access the camera.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[ERROR] Failed to grab frame.")
                continue

            # Update the global current_frame
            current_frame = frame.copy()
            print("[DEBUG] Current frame shape:", current_frame.shape if current_frame is not None else None)
            print("[DEBUG] Frame updated in generate function.")  # Debug statement

            # Encode the frame as JPEG for streaming
            _, encoded_frame = cv2.imencode('.jpg', frame)
            encoded_frame = encoded_frame.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + encoded_frame + b'\r\n')

            time.sleep(0.1)
    finally:
        cap.release()

# Route: Home Page
@app.route('/')
def index():
    return render_template('index.html')

# Route: Video Feed for Real-Time Detection
@app.route('/video_feed')
def video_feed():
    return Response(generate(save=False), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route: Data Collection
@app.route('/collect', methods=['GET', 'POST'])
def collect():
    if request.method == 'POST':
        class_name = request.form['class_name']
        num_images = int(request.form['num_images'])
        return render_template('collect_feed.html', class_name=class_name, num_images=num_images)
    return render_template('collect.html')

@app.route('/collect_feed/<class_name>/<int:num_images>')
def collect_feed(class_name, num_images):
    from siamese_model import collect_images  # Import the updated collect_images function
    return Response(collect_images(class_name=class_name, num_images=num_images),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/save_image/<class_name>')
def save_image(class_name):
    global current_frame
    target_count = request.args.get('target', type=int)

    class_path = os.path.join(DATA_DIR, class_name)
    os.makedirs(class_path, exist_ok=True)

    # Count existing images
    existing_images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png'))]
    if target_count is not None and len(existing_images) >= target_count:
        print("[INFO] Target image count reached.")
        return "Target count reached", 400

    if current_frame is None:
        print("[DEBUG] current_frame is None. Attempting to update.")
        for _ in generate():
            break

    if current_frame is not None:
        imgname = os.path.join(class_path, '{}.jpg'.format(uuid.uuid1()))
        cv2.imwrite(imgname, current_frame)
        print(f"[DEBUG] Saved {imgname}")
        return "Image saved successfully!", 200

    print("[DEBUG] current_frame is still None.")
    return "No frame available to save.", 400

@app.route('/image_count/<class_name>')
def image_count(class_name):
    class_path = os.path.join(DATA_DIR, class_name)
    if os.path.exists(class_path):
        count = len([f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png'))])
    else:
        count = 0
    return {"count": count}

# Route: Train the Model
@app.route('/train', methods=['POST'])
def train_model():
    train()  # Call the train function from siamese_model.py
    return render_template('message.html', message="Training completed successfully!")

# Route: Real-Time Detection
@app.route('/realtime_feed')
def realtime_feed():
    from siamese_model import realtime  # Import the realtime function
    return Response(realtime(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/realtime', methods=['GET','POST'])
def realtime_detection():
    return render_template('realtimefeed.html')

if __name__ == "__main__":
    app.run(debug=True)