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

# NEW: logging utilities
from logs.logging_utils import log_event
# ----------------------------------------------------------

# ---------------- CONFIG ----------------
STREAM_URL = "http://192.168.0.100:8080/video"

ARC_MODEL_PATH = "models/arcface_w600k_r50.onnx"
BASELINE_CSV = "initial_assets/baseline_embeddings.csv"
KNN_PATH = "initial_assets/initial_knn_model.pkl"
LE_PATH = "initial_assets/label_encoder.pkl"
CENTROIDS_PATH = "initial_assets/centroids.pkl"

DRIFT_DIR = "drift_assets"
DRIFT_LOG = "monitoring_log.csv"

WINDOW = 20                    # PSI window size
PSI_THRESHOLD = 0.10           # drift threshold
MIN_KNN_CONFIDENCE = 0.75      # reject unreliable predictions
MIN_CENTROID_SIM = 0.35        # centroid consistency threshold

AUTO_RETRAIN = True
PHASE3_SCRIPT = "phase3_retrain_person.py"
RETRAIN_ONLY_PERSON = None     # None = retrain all subjects

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
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ------------- LOAD MODELS -------------
log("[INIT] Loading YOLOv8-face detector...", "cyan")
det = YOLO("models/yolov8s-face.pt")

log("[INIT] Loading InsightFace landmarks...", "cyan")
fa = FaceAnalysis(name="buffalo_l")
fa.prepare(ctx_id=0, det_size=(512, 512))

log("[INIT] Loading ArcFace ONNX...", "cyan")
arc = ort.InferenceSession(
    ARC_MODEL_PATH,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
arc_input = arc.get_inputs()[0].name

log("[INIT] Loading baselines, model & centroids...", "cyan")
df = pd.read_csv(BASELINE_CSV)
knn = joblib.load(KNN_PATH)
le = joblib.load(LE_PATH)
centroids = joblib.load(CENTROIDS_PATH)

baseline_per_person = {
    person: df[df["label"] == person].iloc[:, :-1].values
    for person in df["label"].unique()
}

buffers = {p: deque(maxlen=WINDOW) for p in baseline_per_person.keys()}

# Drift CSV
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


def get_embedding(frame):
    """Extract ArcFace embedding from biggest face."""
    results = det(frame, verbose=False)
    if len(results[0].boxes) == 0:
        return None, None

    boxes = results[0].boxes.xyxy.cpu().numpy()
    x1, y1, x2, y2 = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])).astype(int)

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = expand_box(x1, y1, x2, y2, w, h)

    face_crop = frame[y1:y2, x1:x2]
    faces = fa.get(face_crop)
    if len(faces) == 0:
        return None, None

    kps = faces[0].kps + np.array([x1, y1])
    aligned = norm_crop(frame, kps)

    resized = cv2.resize(aligned, (112, 112))
    blob = resized[:, :, ::-1].astype(np.float32)
    blob = (blob - 127.5) / 128.0
    blob = np.transpose(blob, (2, 0, 1))[None]

    emb = arc.run(None, {arc_input: blob})[0][0]
    return emb, aligned


# ------------- MAIN LOOP -------------
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        emb, aligned = get_embedding(frame)
        if emb is None:
            continue

        # ---------- KNN RECOGNITION ----------
        pred_enc = knn.predict([emb])[0]
        probs = knn.predict_proba([emb])[0]
        cls_idx = int(np.where(knn.classes_ == pred_enc)[0][0])

        confidence = float(probs[cls_idx])
        person = le.inverse_transform([pred_enc])[0]

        # ---------- CENTROID CONSISTENCY ----------
        if person not in centroids:
            continue

        sim = cos_sim(emb, centroids[person])

        if confidence < MIN_KNN_CONFIDENCE or sim < MIN_CENTROID_SIM:
            log(
                f"[REJECT] '{person}' conf={confidence:.3f}, sim={sim:.3f}",
                "yellow",
            )
            log_event(
                phase="phase2",
                event="REJECTED_SAMPLE",
                subject=person,
                extra_info=f"conf={confidence:.3f}, sim={sim:.3f}"
            )
            continue

        # Append to drift buffer
        buffers[person].append(emb)

        if len(buffers[person]) < WINDOW:
            continue

        # ---------- PSI DRIFT ----------
        base = baseline_per_person[person]
        curr = np.array(buffers[person])
        p = psi(base, curr)

        status = "DRIFT" if p > PSI_THRESHOLD else "OK"
        color = "red" if status == "DRIFT" else "green"

        log(f"[PSI] {person}: psi={p:.3f} → {status}", color)

        # Log PSI event
        log_event(
            phase="phase2",
            event="PSI_OK" if status == "OK" else "DRIFT_DETECTED",
            subject=person,
            psi=p,
            extra_info=f"conf={confidence:.3f}, sim={sim:.3f}"
        )

        # Save drift to CSV
        ts = time.time()
        row = pd.DataFrame([[ts, person, p, status]],
                           columns=["timestamp", "person", "psi", "status"])
        row.to_csv(DRIFT_LOG, mode="a", header=False, index=False)

        # ---------- RETRAIN ----------
        if status == "DRIFT" and AUTO_RETRAIN:

            if RETRAIN_ONLY_PERSON and person != RETRAIN_ONLY_PERSON:
                continue

            drift_path = os.path.join(DRIFT_DIR, f"{person}_drift_embeddings.npy")
            np.save(drift_path, curr)

            log_event(
                phase="phase2",
                event="RETRAIN_TRIGGERED",
                subject=person,
                psi=p
            )

            log(f"[AUTO] Running Phase 3 retraining for {person}...", "cyan")

            try:
                import subprocess
                subprocess.run(
                    [sys.executable, PHASE3_SCRIPT, "--person", person],
                    check=True,
                )

                # Reload updated models
                df = pd.read_csv(BASELINE_CSV)
                knn = joblib.load(KNN_PATH)
                le = joblib.load(LE_PATH)
                centroids = joblib.load(CENTROIDS_PATH)

                baseline_per_person = {
                    p_name: df[df["label"] == p_name].iloc[:, :-1].values
                    for p_name in df["label"].unique()
                }

                log_event(
                    phase="phase2",
                    event="RETRAIN_COMPLETED_RELOAD",
                    subject=person
                )

                log("[INFO] Model & baselines updated.", "green")

            except Exception as e:
                log(f"[ERROR] Retrain failed: {e}", "red")
                log_event(
                    phase="phase2",
                    event="RETRAIN_FAILED",
                    subject=person,
                    extra_info=str(e)
                )

except KeyboardInterrupt:
    log("\n[EXIT] User stopped.", "yellow")

finally:
    cap.release()
