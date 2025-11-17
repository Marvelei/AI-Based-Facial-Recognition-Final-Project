import os
import cv2
import numpy as np
import pandas as pd
import sys
import joblib
from deepface import DeepFace
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

# Correct Evidently imports for v0.7.16
from evidently import Report
from evidently.presets import DataDriftPreset

import tensorflow as tf

# --- GPU / CPU CONFIGURATION ---
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✅ Using {len(gpus)} GPU(s)")
else:
    print("⚠️ No GPU detected. Using CPU")

# --- CONFIGURATION ---
ASSETS_DIR = "initial_assets"
KNN_MODEL_PATH = os.path.join(ASSETS_DIR, 'initial_knn_model.pkl')
LABEL_ENCODER_PATH = os.path.join(ASSETS_DIR, 'label_encoder.pkl')
BASELINE_EMBEDDINGS_PATH = os.path.join(ASSETS_DIR, 'baseline_embeddings.csv')

DETECTOR = "mtcnn"
EMBEDDING_MODEL = "ArcFace"
CAMERA_INDEX = 0

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_SKIP = 2
WINDOW_SIZE = 50
DRIFT_THRESHOLD_PSI = 0.1

# --- LOAD ASSETS ---
try:
    KNN_MODEL = joblib.load(KNN_MODEL_PATH)
    LABEL_ENCODER = joblib.load(LABEL_ENCODER_PATH)
    DEEPFACE_MODEL = DeepFace.build_model(EMBEDDING_MODEL)
    BASELINE_EMBEDDINGS = pd.read_csv(BASELINE_EMBEDDINGS_PATH).drop(columns=['label'], errors='ignore')
    print("✅ Assets Loaded: k-NN, LabelEncoder, ArcFace, Baseline Embeddings")
except Exception as e:
    print(f"FATAL: Missing or invalid assets: {e}")
    sys.exit(1)

# --- HELPER FUNCTIONS ---
def predict_identity(embedding):
    embedding_vector = np.array(embedding).reshape(1, -1)
    distances, _ = KNN_MODEL.kneighbors(embedding_vector)
    min_distance = float(distances[0][0])
    threshold = 0.60
    if min_distance < threshold:
        idx = KNN_MODEL.predict(embedding_vector)[0]
        name = LABEL_ENCODER.inverse_transform([idx])[0]
        return f"{name} ({min_distance:.2f})"
    else:
        return f"Unknown ({min_distance:.2f})"

def compute_psi(baseline_df, current_df):
    """Compute PSI using Evidently 0.7.16 with DataDriftPreset"""
    report = Report(metrics=[DataDriftPreset(method="psi")])
    result = report.run(reference_data=baseline_df, current_data=current_df)
    result_dict = result.as_dict()
    try:
        psi_value = result_dict["metrics"][0]["result"]["score"]
        return psi_value
    except Exception as e:
        print("Error reading PSI value from report result:", e)
        return None

# --- REAL-TIME STREAM ---
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"FATAL: Cannot open camera {CAMERA_INDEX}")
    sys.exit(1)

print("\n--- STARTING REAL-TIME MONITORING ---")
frame_count = 0
window_embeddings = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        continue

    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
    display_frame = frame.copy()

    try:
        faces = DeepFace.extract_faces(frame, detector_backend=DETECTOR, enforce_detection=False)
        for face_data in faces:
            emb_objs = DeepFace.represent(face_data['face'], model=DEEPFACE_MODEL,
                                          detector_backend='skip', enforce_detection=False)
            if emb_objs:
                embedding = emb_objs[0]['embedding']
                window_embeddings.append(embedding)

                identity = predict_identity(embedding)

                area = face_data['facial_area']
                x, y, w, h = area.get('x', 0), area.get('y', 0), area.get('w', 0), area.get('h', 0)
                color = (0, 255, 0) if "Unknown" not in identity else (0, 0, 255)
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(display_frame, identity, (x, max(y - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    except Exception:
        pass

    if len(window_embeddings) >= WINDOW_SIZE:
        current_df = pd.DataFrame(window_embeddings)
        psi_value = compute_psi(BASELINE_EMBEDDINGS, current_df)
        if psi_value is not None:
            if psi_value > DRIFT_THRESHOLD_PSI:
                print(f"⚠️ Drift detected! PSI = {psi_value:.4f}")
            else:
                print(f"PSI = {psi_value:.4f} (no drift)")
        else:
            print("⚠️ PSI computation failed.")
        window_embeddings = []

    cv2.imshow("Drift + Recognition", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Stopped monitoring.")
