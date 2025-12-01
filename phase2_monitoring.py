import os
import sys
import time
from collections import deque

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from insightface.utils.face_align import norm_crop
import onnxruntime as ort
import joblib
from colorama import Fore, Style, init

# ---------------- CONFIG ----------------
STREAM_URL = "http://192.168.0.100:8080/video"

ARC_MODEL_PATH = "models/arcface_w600k_r50.onnx"
YOLO_WEIGHTS = "models/yolov8s-face.pt"

BASELINE_CSV = "initial_assets/baseline_embeddings.csv"
KNN_PATH = "initial_assets/initial_knn_model.pkl"
LE_PATH = "initial_assets/label_encoder.pkl"
CENTROIDS_PATH = "initial_assets/centroids.pkl"

DRIFT_DIR = "drift_assets"
DRIFT_LOG = "monitoring_log.csv"

WINDOW = 20                  # PSI window size per person
PSI_THRESHOLD = 0.10         # drift threshold
MIN_KNN_CONFIDENCE = 0.75    # below this → treat as UNKNOWN (no drift)
MIN_CENTROID_SIMILARITY = 0.35  # centroid consistency threshold

AUTO_RETRAIN = True
PHASE3_SCRIPT = "phase3_retrain_person.py"
# If you ever want to only auto-retrain Marvel, set this to "Marvel".
RETRAIN_ONLY_PERSON = None   # None = allow auto-retrain for all persons

# ------------- INIT UTILITIES -------------
init(autoreset=True)
os.makedirs(DRIFT_DIR, exist_ok=True)


def log(msg, color="white"):
    c = {
        "red": Fore.RED,
        "green": Fore.GREEN,
        "yellow": Fore.YELLOW,
        "cyan": Fore.CYAN,
        "magenta": Fore.MAGENTA,
        "blue": Fore.BLUE,
        "white": Fore.WHITE,
    }.get(color, Fore.WHITE)
    print(c + msg + Style.RESET_ALL)


def expand_box(x1, y1, x2, y2, img_w, img_h, scale=0.60):
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2

    new_w = w * (1 + scale)
    new_h = h * (1 + scale)

    nx1 = int(max(0, cx - new_w / 2))
    ny1 = int(max(0, cy - new_h / 2))
    nx2 = int(min(img_w, cx + new_w / 2))
    ny2 = int(min(img_h, cy + new_h / 2))
    return nx1, ny1, nx2, ny2


def psi(base, curr, buckets=10):
    """Population Stability Index between 2 distributions."""
    base = np.array(base).ravel()
    curr = np.array(curr).ravel()

    edges = np.linspace(min(base.min(), curr.min()),
                        max(base.max(), curr.max()),
                        buckets + 1)

    b_hist, _ = np.histogram(base, bins=edges)
    c_hist, _ = np.histogram(curr, bins=edges)

    b_hist = b_hist / (len(base) + 1e-8)
    c_hist = c_hist / (len(curr) + 1e-8)

    psi_vals = (c_hist - b_hist) * np.log((c_hist + 1e-8) / (b_hist + 1e-8))
    return float(np.sum(psi_vals))


def cos_sim(a, b):
    """Cosine similarity between two vectors."""
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ------------- LOAD MODELS / BASELINES -------------
log("[INIT] Loading YOLOv8-face detector...", "cyan")
det = YOLO(YOLO_WEIGHTS)

log("[INIT] Loading InsightFace landmarks...", "cyan")
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(512, 512))

log("[INIT] Loading ArcFace ONNX...", "cyan")
arc = ort.InferenceSession(
    ARC_MODEL_PATH,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
arc_input_name = arc.get_inputs()[0].name

log("[INIT] Loading baselines, models & centroids...", "cyan")
df = pd.read_csv(BASELINE_CSV)
knn = joblib.load(KNN_PATH)
le = joblib.load(LE_PATH)
centroids = joblib.load(CENTROIDS_PATH)

# baseline embeddings per person
baseline_per_person = {
    person: df[df["label"] == person].iloc[:, :-1].values
    for person in df["label"].unique()
}

# Create drift windows (sliding buffer per person)
buffers = {
    person: deque(maxlen=WINDOW)
    for person in baseline_per_person.keys()
}

# Prepare drift log CSV
if not os.path.exists(DRIFT_LOG):
    pd.DataFrame(columns=["timestamp", "person", "psi", "status"]).to_csv(
        DRIFT_LOG, index=False
    )


# ------------- OPEN CAMERA -------------
log(f"[CAM] Opening stream: {STREAM_URL}", "cyan")
cap = cv2.VideoCapture(STREAM_URL)

if not cap.isOpened():
    raise RuntimeError("Cannot open camera stream.")

log("[RUN] Real-time drift monitoring started...", "green")
log("Press CTRL+C to stop.\n", "green")


def get_embedding_from_frame(frame):
    """Detect, align and embed biggest face from frame."""
    results = det(frame, verbose=False)
    if len(results[0].boxes) == 0:
        return None

    boxes = results[0].boxes.xyxy.cpu().numpy()
    box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    x1, y1, x2, y2 = box.astype(int)

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = expand_box(x1, y1, x2, y2, w, h, scale=0.60)

    face_crop = frame[y1:y2, x1:x2]
    faces = app.get(face_crop)
    if len(faces) == 0:
        return None

    f = faces[0]
    kps = f.kps + np.array([x1, y1])
    aligned = norm_crop(frame, kps)

    resized = cv2.resize(aligned, (112, 112))
    blob = resized[:, :, ::-1].astype(np.float32)
    blob = (blob - 127.5) / 128.0
    blob = np.transpose(blob, (2, 0, 1))[None, :, :, :]

    emb = arc.run(None, {arc_input_name: blob})[0][0]
    return emb


# ------------- MAIN LOOP -------------
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            log("[WARN] No frame received.", "yellow")
            continue

        emb = get_embedding_from_frame(frame)
        if emb is None:
            continue

        # ---------- KNN RECOGNITION ----------
        pred_encoded = knn.predict([emb])[0]
        probs = knn.predict_proba([emb])[0]

        cls_index = int(np.where(knn.classes_ == pred_encoded)[0][0])
        confidence = float(probs[cls_index])
        person = le.inverse_transform([pred_encoded])[0]

        # ---------- CENTROID CONSISTENCY CHECK ----------
        if person not in centroids:
            log(f"[WARN] Person '{person}' not in centroids dict. Skipping drift.", "yellow")
            continue

        sim = cos_sim(emb, centroids[person])

        if confidence < MIN_KNN_CONFIDENCE or sim < MIN_CENTROID_SIMILARITY:
            log(
                f"[UNKNOWN] Rejecting embedding: pred='{person}', conf={confidence:.3f}, sim={sim:.3f}",
                "yellow",
            )
            continue

        # ---------- DRIFT BUFFER ----------
        if person not in buffers:
            buffers[person] = deque(maxlen=WINDOW)

        buffers[person].append(emb)

        if len(buffers[person]) < WINDOW:
            continue

        base = baseline_per_person[person]
        curr = np.array(buffers[person])

        p = psi(base, curr)

        status = "DRIFT" if p > PSI_THRESHOLD else "OK"
        color = "red" if status == "DRIFT" else "green"
        log(
            f"[PSI] {person}: psi={p:.3f} -> {status} (conf={confidence:.3f}, sim={sim:.3f})",
            color,
        )

        # Log to CSV
        ts = time.time()
        row = pd.DataFrame(
            [[ts, person, p, status]],
            columns=["timestamp", "person", "psi", "status"],
        )
        row.to_csv(DRIFT_LOG, mode="a", header=False, index=False)

        # ---------- AUTO RETRAIN HOOK ----------
        if status == "DRIFT" and AUTO_RETRAIN:
            if (RETRAIN_ONLY_PERSON is not None) and (person != RETRAIN_ONLY_PERSON):
                log(
                    f"[INFO] Drift detected for {person}, but auto-retrain limited to {RETRAIN_ONLY_PERSON}.",
                    "magenta",
                )
                continue

            drift_path = os.path.join(DRIFT_DIR, f"{person}_drift_embeddings.npy")
            np.save(drift_path, curr)
            log(f"[SAVE] Drift embeddings saved at {drift_path}", "magenta")

            # Call Phase 3 retrain
            try:
                log(f"[AUTO] Running Phase 3 retraining for {person}...", "cyan")
                import subprocess

                subprocess.run(
                    [sys.executable, PHASE3_SCRIPT, "--person", person],
                    check=True,
                )

                # Reload updated model, baselines & centroids after retrain
                log("[INFO] Reloading updated model, baselines & centroids...", "cyan")
                df = pd.read_csv(BASELINE_CSV)
                knn = joblib.load(KNN_PATH)
                le = joblib.load(LE_PATH)
                centroids = joblib.load(CENTROIDS_PATH)

                baseline_per_person = {
                    p_name: df[df["label"] == p_name].iloc[:, :-1].values
                    for p_name in df["label"].unique()
                }

                # buffers keep their structure; they’ll refill naturally
                log("[INFO] Model & baselines updated after retraining.", "green")
            except Exception as e:
                log(f"[ERROR] Phase 3 retrain failed: {e}", "red")

except KeyboardInterrupt:
    log("\n[EXIT] Stopped by user.", "yellow")
finally:
    cap.release()
