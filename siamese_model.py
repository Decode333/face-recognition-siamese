import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
import cv2
import uuid
import numpy as np
from matplotlib import pyplot as plt
import random
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Layer
from sklearn.model_selection import train_test_split

# Set these before running
DATA_DIR = r"E:\face recognition data\Class"
EMBEDDING_PATH = r"E:\face recognition data\class_embeddings.npy"
CLASS_NAMES_PATH = r"E:\face recognition data\class_names.npy"
MODEL_WEIGHTS_PATH = r"E:\face recognition data\siamese_model.h5"
MODEL_DIR = r"E:\face recognition data"

MODE = "realtime"  # change to "realtime" to run OpenCV-based identification or "data_collection" to collect images or "train" to train the model
IMAGE_SIZE = (100, 100)

"""
def collect_images(class_name, num_images):
    print("\nStarting image collection... Press 's' to save, 'q' to quit.")
    class_path = os.path.join(DATA_DIR, class_name)
    os.makedirs(class_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            break
        roi = frame[120:120+330, 150:150+330, :]
        cv2.imshow("Capturing", roi)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            imgname = os.path.join(class_path, '{}.jpg'.format(uuid.uuid1()))
            cv2.imwrite(imgname, roi)
            print(f"Saved {imgname}")
            count += 1
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
"""
def collect_images(class_name, num_images):
    print("\nStarting image collection...")
    class_path = os.path.join(DATA_DIR, class_name)
    os.makedirs(class_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0

    while cap.isOpened() and count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        # Extract the region of interest (ROI)
        roi = frame[120:120+330, 150:150+330, :]

        # Store the current frame in a global variable for saving
        global current_frame
        current_frame = roi

        # Encode the frame as JPEG for streaming
        _, encoded_frame = cv2.imencode('.jpg', frame)
        frame_bytes = encoded_frame.tobytes()

        # Yield the frame for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()
    
def preprocess_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, IMAGE_SIZE)
    img = img / 255.0
    return img

def create_pairs():
    pairs = []
    labels = []
    class_dirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    class_to_imgs = {
        c: [os.path.join(DATA_DIR, c, f) for f in os.listdir(os.path.join(DATA_DIR, c))]
        for c in class_dirs
    }
    for c in class_dirs:
        imgs = class_to_imgs[c]
        for i in range(len(imgs)):
            for j in range(i + 1, len(imgs)):
                pairs.append((imgs[i], imgs[j]))
                labels.append(1)
            other_classes = [k for k in class_dirs if k != c]
            for neg_class in other_classes:
                neg_img = np.random.choice(class_to_imgs[neg_class])
                pairs.append((imgs[i], neg_img))
                labels.append(0)
    return pairs, labels, class_dirs

def make_embedding_model():
    inp = Input(shape=(100, 100, 3))
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inp)
    x = MaxPooling2D()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    out = Dense(128, activation='sigmoid')(x)
    return Model(inp, out)

class L1Dist(Layer):
    def call(self, a, b):
        return tf.math.abs(a - b)
    
def make_siamese_model(embedding):
    input_a = Input(shape=(100, 100, 3))
    input_b = Input(shape=(100, 100, 3))
    emb_a = embedding(input_a)
    emb_b = embedding(input_b)
    distance = L1Dist()(emb_a, emb_b)
    output = Dense(1, activation='sigmoid')(distance)
    return Model([input_a, input_b], output)

def train():
    print("[INFO] Starting training...")

    # Load image paths
    class_to_imgs = {}
    for class_dir in os.listdir(DATA_DIR):
        class_path = os.path.join(DATA_DIR, class_dir)
        if not os.path.isdir(class_path): continue
        image_files = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.lower().endswith((".jpg", ".png"))]
        class_to_imgs[class_dir] = image_files

    class_names = list(class_to_imgs.keys())

    # Save class names to a .npy file
    np.save(os.path.join(MODEL_DIR, "class_names.npy"), class_names)

    # Generate training pairs (positive and negative)
    pairs = []
    labels = []

    for class_dir in class_names:
        images = class_to_imgs[class_dir]
        for i in range(len(images) - 1):
            img1, img2 = preprocess_image(images[i]), preprocess_image(images[i + 1])
            pairs.append((img1, img2))
            labels.append(1)

            # Sample a negative from another class
            other_class = random.choice([c for c in class_names if c != class_dir])
            neg_img = preprocess_image(random.choice(class_to_imgs[other_class]))
            pairs.append((img1, neg_img))
            labels.append(0)

    # Convert to numpy arrays
    pair_left = np.array([p[0] for p in pairs])
    pair_right = np.array([p[1] for p in pairs])
    labels = np.array(labels)

    # Train the model
    embedding = make_embedding_model()
    siamese_model = make_siamese_model(embedding)

    siamese_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    siamese_model.fit([pair_left, pair_right], labels, batch_size=16, epochs=10)

    # Save models
    siamese_model.save(os.path.join(MODEL_DIR, "siamese_model.h5"))
    embedding.save_weights(os.path.join(MODEL_DIR, "embedding.weights.h5"))  # Save embedding weights

    # Create and save embeddings for all classes
    embedding_model = make_embedding_model()
    embedding_model.load_weights(os.path.join(MODEL_DIR, "embedding.weights.h5"))  # Correct weights

    class_embeddings = {}
    for class_dir in class_names:
        imgs = class_to_imgs[class_dir][:5]  # use first 5
        embs = [embedding_model.predict(np.expand_dims(preprocess_image(p), axis=0))[0] for p in imgs]
        mean_emb = np.mean(embs, axis=0)
        class_embeddings[class_dir] = mean_emb

    np.save(os.path.join(MODEL_DIR, "class_embeddings.npy"), class_embeddings)
    print("[INFO] Training and embedding creation complete.")
"""
def realtime():
    # Load the Siamese model
    model = tf.keras.models.load_model(MODEL_WEIGHTS_PATH, custom_objects={'L1Dist': L1Dist})

    # embedding = model.get_layer(index=2).input[0]._keras_history[0]  # not elegant, works
    # class_embeddings = np.load(EMBEDDING_PATH, allow_pickle=True).item()
    # class_names = np.load(CLASS_NAMES_PATH)

    # Recreate the embedding model and load its weights
    embedding_model = make_embedding_model()
    embedding_model.load_weights(os.path.join(MODEL_DIR, "embedding.weights.h5"))

    # Load class embeddings and class names
    class_embeddings = np.load(EMBEDDING_PATH, allow_pickle=True).item()
    class_names = np.load(CLASS_NAMES_PATH)


    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Unable to access the camera.")
    else:
        print("[INFO] Camera is working. Press 'q' to exit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Extract the region of interest (ROI) and preprocess it
        roi = frame[120:120+330, 150:150+330, :]
        img = cv2.resize(roi, IMAGE_SIZE)
        img = img / 255.0

        # Generate the embedding for the input image
        emb = embedding_model.predict(np.expand_dims(img, axis=0))[0]

        # Compute similarities with class embeddings
        sims = [np.linalg.norm(emb - ce) for ce in class_embeddings.values()]
        min_idx = np.argmin(sims)
        min_dist = sims[min_idx]

        # Determine the class name
        name = class_names[min_idx] if min_dist < 0.6 else "Unknown"

        # Display the result
        cv2.rectangle(frame, (150, 120), (480, 450), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} ({min_dist:.2f})", (150, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Real-time Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
"""

def realtime():
    # Load the Siamese model
    model = tf.keras.models.load_model(MODEL_WEIGHTS_PATH, custom_objects={'L1Dist': L1Dist})

    # Recreate the embedding model and load its weights
    embedding_model = make_embedding_model()
    embedding_model.load_weights(os.path.join(MODEL_DIR, "embedding.weights.h5"))

    # Load class embeddings and class names
    class_embeddings = np.load(EMBEDDING_PATH, allow_pickle=True).item()
    class_names = np.load(CLASS_NAMES_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Unable to access the camera.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Extract the region of interest (ROI) and preprocess it
        roi = frame[120:120+330, 150:150+330, :]
        img = cv2.resize(roi, IMAGE_SIZE)
        img = img / 255.0

        # Generate the embedding for the input image
        emb = embedding_model.predict(np.expand_dims(img, axis=0))[0]

        # Compute similarities with class embeddings
        sims = [np.linalg.norm(emb - ce) for ce in class_embeddings.values()]
        min_idx = np.argmin(sims)
        min_dist = sims[min_idx]

        # Determine the class name
        name = class_names[min_idx] if min_dist < 0.6 else "Unknown"

        # Draw bounding box and label on the frame
        cv2.rectangle(frame, (150, 120), (480, 450), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} ({min_dist:.2f})", (150, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Display the frame locally
        cv2.imshow("Real-time Recognition", frame)

        # Encode the frame as JPEG for streaming
        _, encoded_frame = cv2.imencode('.jpg', frame)
        frame_bytes = encoded_frame.tobytes()

        # Yield the frame for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Only execute this block when running siamese_model.py directly
    MODE = "realtime"  # or "train" or "data_collection"
    if MODE == "realtime":
        realtime()
    elif MODE == "train":
        train()
    elif MODE == "data_collection":
        class_name = input("Enter the class name for data collection: ")
        num_images = int(input("Enter the number of images to collect: "))
        collect_images()