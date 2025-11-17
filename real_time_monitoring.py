# real_time_monitoring.py
# Complete real-time face recognition + PSI drift monitoring
# Includes GPU utilization, bounding box colors, k-NN confidence, and logging.

import cv2
import torch
import numpy as np
import pandas as pd
import joblib

from deepface import DeepFace
from evidently import Report
from evidently.presets import DataDriftPreset

from logs.logging_utils import log_event, get_current_model_version


# =========================================================
# CONFIGURATION
# =========================================================

BASELINE_EMBEDDINGS_PATH = "initial_assets/baseline_embeddings.csv"
MODEL_PATH = "initial_assets/initial_knn_model.pkl"

WINDOW_SIZE = 30
DRIFT_THRESHOLD_PSI = 0.25   # Adjust depending on drift sensitivity

DETECTION_BACKEND = "mtcnn"
EMBEDDING_MODEL = "ArcFace"


# =========================================================
# GPU SETUP
# =========================================================

def get_device():
    """Return 'cuda' if GPU is available, else 'cpu'."""
    if torch.cuda.is_available():
        print("[INFO] CUDA GPU detected. DeepFace will use GPU.")
        return "cuda"
    print("[INFO] GPU not available. Using CPU.")
    return "cpu"


DEVICE = get_device()


# =========================================================
# LOAD BASELINE + MODEL
# =========================================================

knn_model = joblib.load(MODEL_PATH)
baseline_df = pd.read_csv(BASELINE_EMBEDDINGS_PATH)

last_psi = None
last_confidence = None


# =========================================================
# PSI CALCULATION
# =========================================================

def compute_psi(reference_df, window_embeddings):
    """
    Compute PSI (Population Stability Index) between baseline embeddings
    and current window embeddings using EvidentlyAI.
    """
    current_df = pd.DataFrame(window_embeddings)
    current_df.columns = reference_df.columns

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)

    psi_value = report.as_dict()["metrics"][0]["result"]["dataset_drift"]["psi"]
    return psi_value


# =========================================================
# FACE RECOGNITION (Embedding + k-NN)
# =========================================================

def recognize_face(face_img):
    """
    Extract ArcFace embedding and run recognition using k-NN classifier.
    Returns:
        embedding (list)
        label (string)
        confidence (float)
    """
    embedding = DeepFace.represent(
        img_path=face_img,
        model_name=EMBEDDING_MODEL,
        detector_backend="skip",
        enforce_detection=False,
        device=DEVICE
    )[0]["embedding"]

    embedding_np = np.array(embedding).reshape(1, -1)

    # Predict label
    prediction = knn_model.predict(embedding_np)[0]

    # Predict confidence (probability)
    proba = knn_model.predict_proba(embedding_np)[0]
    confidence = float(np.max(proba))

    return embedding, prediction, confidence


# =========================================================
# DRAW BOUNDING BOX WITH STATUS COLORS
# =========================================================

def get_box_color(psi_value):
    """
    Return the RGB color for bounding box based on PSI drift severity.
    GREEN  = safe
    YELLOW = mild drift
    RED    = drift alert
    """
    if psi_value is None:
        return (0, 255, 0)   # Default green when no PSI yet

    if psi_value < DRIFT_THRESHOLD_PSI * 0.5:
        return (0, 255, 0)   # SAFE: green
    elif psi_value < DRIFT_THRESHOLD_PSI:
        return (0, 255, 255) # WARNING: yellow
    else:
        return (0, 0, 255)   # ALERT: red


# =========================================================
# MAIN REAL-TIME LOOP
# =========================================================

def main():
    global last_psi, last_confidence

    print("[INFO] Starting real-time monitoring...")
    print(f"[INFO] Model Version: {get_current_model_version()}")

    cap = cv2.VideoCapture(0)
    embedding_window = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read webcam frame.")
            break

        # Detect faces using MTCNN
        try:
            detections = DeepFace.extract_faces(
                img_path=frame,
                detector_backend=DETECTION_BACKEND,
                enforce_detection=False,
                align=True
            )
        except Exception as e:
            print(f"[WARNING] Detection error: {e}")
            continue

        if len(detections) == 0:
            # No face found, just show video
            cv2.imshow("Monitoring", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # Use first detected face
        face = detections[0]
        face_img = face["face"]
        area = face["facial_area"]

        x, y = area["x"], area["y"]
        w, h = area["w"], area["h"]

        # Face recognition
        embedding, label, confidence = recognize_face(face_img)
        last_confidence = confidence

        # Add embedding to window
        embedding_window.append(embedding)

        # Determine bounding box color based on PSI
        box_color = get_box_color(last_psi)

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

        # Build display text
        text = f"{label} | Acc:{confidence:.2f}"
        if last_psi is not None:
            text += f" | PSI:{last_psi:.3f}"

        # Put label above face
        cv2.putText(frame, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

        # Show frame
        cv2.imshow("Monitoring", frame)

        # =====================================================
        # When window full → compute PSI
        # =====================================================
        if len(embedding_window) >= WINDOW_SIZE:
            psi_value = compute_psi(baseline_df, np.array(embedding_window))
            last_psi = psi_value

            # Log PSI event
            log_event(
                phase="phase2",
                event="psi_window",
                subject=label,
                psi=psi_value,
                accuracy=confidence,
                extra_info="PSI computed from monitoring window"
            )

            print(f"[INFO] PSI: {psi_value:.4f}")

            # DRIFT ALERT
            if psi_value >= DRIFT_THRESHOLD_PSI:
                print("[ALERT] DRIFT DETECTED!")

                log_event(
                    phase="phase2",
                    event="DRIFT_ALERT",
                    subject=label,
                    psi=psi_value,
                    accuracy=confidence,
                    extra_info=f"Drift detected. PSI={psi_value}"
                )

            embedding_window = []  # Reset window

        # Quit with 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()
# End of real_time_monitoring.py