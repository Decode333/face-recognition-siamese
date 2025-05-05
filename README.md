# Siamese Neural Network for Face Recognition

This project implements a Siamese Neural Network for face recognition using TensorFlow and Flask. The application allows for data collection, training, and real-time face recognition.

---

## Features
- **Data Collection**: Collect images for training by capturing frames from a webcam.
- **Model Training**: Train a Siamese Neural Network on collected data.
- **Real-Time Detection**: Perform real-time face recognition using the trained model.
- **Web Interface**: Interact with the application through a Flask-based web interface.

---

## Project Structure

project/
├── app/
│   ├── siamese_flask_app.py
│   ├── siamese_model.py
│   ├── templates/
│   └── static/
├── data/
│   ├── Class/
│   ├── class_embeddings.npy
│   └── class_names.npy
├── models/
│   ├── siamese_model.h5
│   └── embedding.weights.h5
├── requirements.txt
└── README.md


---

## Installation

### Prerequisites
- Python 3.8 or higher
- Virtual environment (recommended)

### Steps
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd project

2. Create a virtual environment:
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies:
    pip install -r requirements.txt

4. Ensure your webcam is functional and accessible.

## Usage

1. Start the Flask Application
    Run the Flask app: python [siamese_flask_app.py](http://_vscodecontentref_/2)
    Access the application in your browser at http://127.0.0.1:5000.

2. Data Collection
    Navigate to the Data Collection page.
    Enter a class name and the number of images to collect.
    Start the webcam feed and collect images.

3. Train the Model
    Navigate to the Train Model page.
    Click the "Train" button to train the Siamese Neural Network on the collected data.

4. Real-Time Detection
    Navigate to the Real-Time Detection page.
    Start the webcam feed to perform real-time face recognition.

## Environment Variables
For deployment, you can configure paths using environment variables:

    DATA_DIR: Path to the directory containing class images.
    MODEL_WEIGHTS_PATH: Path to the trained model weights.
    Example: 
        export DATA_DIR=/path/to/data/Class
        export MODEL_WEIGHTS_PATH=/path/to/models/siamese_model.h5

### Dependencies
    Flask
    TensorFlow
    NumPy
    OpenCV
    Matplotlib
Install all dependencies using:
pip install -r requirements.txt


## Real-Time Detection
The realtime() function in siamese_model.py handles real-time detection:
    def realtime():
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess the frame
            roi = preprocess_frame(frame)

            # Perform inference
            embedding = model.predict(roi)
            class_name = identify_class(embedding)

            # Display the result
            cv2.putText(frame, class_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            yield frame

## Preprocessing
The preprocess_frame() function resizes and normalizes the input frame:
    def preprocess_frame(frame):
    roi = frame[120:120+330, 150:150+330, :]
    roi = cv2.resize(roi, IMAGE_SIZE)
    roi = roi / 255.0
    return np.expand_dims(roi, axis=0)

## Class Identification
The identify_class() function compares the embedding with stored class embeddings:
    def identify_class(embedding):
    min_distance = float('inf')
    identified_class = None
    for class_name, class_embedding in class_embeddings.items():
        distance = np.linalg.norm(embedding - class_embedding)
        if distance < min_distance:
            min_distance = distance
            identified_class = class_name
    return identified_class

## Troubleshooting
    Issue: Webcam feed not working.
        Solution: Check camera permissions and ensure no other application is using the webcam.
    Issue: Incorrect predictions.
        Solution: Collect more training data and retrain the model.
    Issue: Application crashes on the server.
        Solution: Check logs for errors and ensure all dependencies are installed.