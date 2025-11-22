# real_time_monitoring.py
# Real-time face recognition + PSI drift monitoring (InsightFace + Evidently)
# GPU-stable, non-freezing version

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import numpy as np
import pandas as pd
import joblib
import time
from concurrent.futures import ThreadPoolExecutor

from insightface.app import FaceAnalysis
from evidently import Report
from evidently.presets import DataDriftPreset

from logs.logging_utils import log_event, get_current_model_version

# =========================================================
# CONFIGURATION
# =========================================================

BASELINE_EMBEDDINGS_PATH = "initial_assets/baseline_embeddings.csv"
MODEL_PATH = "initial_assets/initial_knn_model.pkl"

WINDOW_SIZE = 30
DRIFT_THRESHOLD_PSI = 0.25

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_SKIP = 1

TARGET_FPS = 25
FRAME_DELAY = 1 / TARGET_FPS

INSIGHTFACE_MODEL_NAME = "buffalo_l"

# OpenCV stability tweaks for Windows setups that commonly freeze.
cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)

# =========================================================
# LOAD BASELINE + MODEL
# =========================================================

knn_model = joblib.load(MODEL_PATH)
baseline_raw = pd.read_csv(BASELINE_EMBEDDINGS_PATH)

if "label" in baseline_raw.columns:
    baseline_labels = baseline_raw["label"]
    baseline_embeddings_df = baseline_raw.drop(columns=["label"])
else:
    baseline_labels = None
    baseline_embeddings_df = baseline_raw

baseline_embeddings_df = baseline_embeddings_df.astype(np.float32, copy=False)
baseline_embeddings = baseline_embeddings_df.values

BASELINE_DIM = baseline_embeddings.shape[1]
print(f"[INFO] Baseline embedding dimension: {BASELINE_DIM}")

last_psi = None
last_confidence = None


# =========================================================
# PSI CALCULATION
# =========================================================

def compute_psi(reference_df: pd.DataFrame, window_embeddings: np.ndarray) -> float:
    if window_embeddings.ndim != 2:
        window_embeddings = window_embeddings.reshape(-1, BASELINE_DIM)

    if window_embeddings.shape[1] != reference_df.shape[1]:
        print("[ERROR] PSI dimension mismatch.")
        return None

    current_df = pd.DataFrame(window_embeddings, columns=reference_df.columns)

    try:
        report = Report(metrics=[DataDriftPreset(method="psi", include_tests=False)])
        snapshot = report.run(
            reference_data=reference_df,
            current_data=current_df
        )
    except Exception as err:
        print(f"[ERROR] PSI error: {err}")
        return None

    psi_scores = []
    for metric in snapshot.metric_results.values():
        display_name = getattr(metric, "display_name", "")
        if isinstance(display_name, str) and display_name.lower().startswith("value drift"):
            value = getattr(metric, "value", None)
            if isinstance(value, (int, float)):
                psi_scores.append(float(value))

    if not psi_scores:
        return None

    return float(np.nanmean(psi_scores))


# =========================================================
# FACE RECOGNITION
# =========================================================

def recognize_face(face_obj):
    embedding = face_obj.embedding
    if embedding is None:
        return None, None, None

    embedding = np.asarray(embedding, dtype=np.float32)

    if embedding.shape[0] != BASELINE_DIM:
        return None, None, None

    embedding_np = embedding.reshape(1, -1)
    prediction = knn_model.predict(embedding_np)[0]

    if hasattr(knn_model, "predict_proba"):
        confidence = float(np.max(knn_model.predict_proba(embedding_np)[0]))
    else:
        confidence = 1.0

    return embedding, prediction, confidence


# =========================================================
# STATUS COLORS
# =========================================================

def get_box_color(psi_value):
    if psi_value is None:
        return (0, 255, 0)
    if psi_value < DRIFT_THRESHOLD_PSI * 0.5:
        return (0, 255, 0)
    elif psi_value < DRIFT_THRESHOLD_PSI:
        return (0, 255, 255)
    else:
        return (0, 0, 255)


# =========================================================
# MAIN LOOP
# =========================================================

def main():
    global last_psi, last_confidence

    print("[INFO] Initializing InsightFace model (buffalo_l)...")
    app = FaceAnalysis(
        name=INSIGHTFACE_MODEL_NAME,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    app.prepare(ctx_id=0, det_size=(FRAME_WIDTH, FRAME_HEIGHT))
    print("[INFO] InsightFace model loaded.")

    dummy = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    _ = app.get(dummy)

    print("[INFO] Starting real-time monitoring...")
    print(f"[INFO] Model Version: {get_current_model_version()}")

    # ---- FIX: Prevent cv2/CUDA freeze ----
    cv2.namedWindow("Monitoring", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Monitoring", FRAME_WIDTH, FRAME_HEIGHT)
    # --------------------------------------

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    embedding_window = []
    frame_count = 0
    last_label = "unknown"

    psi_executor = ThreadPoolExecutor(max_workers=1)
    psi_future = None
    pending_psi_context = None

    try:
        while True:
            loop_start = time.time()

            # Handle previously scheduled PSI computation
            if psi_future is not None and psi_future.done():
                try:
                    psi_value = psi_future.result()
                except Exception as future_error:
                    print(f"[ERROR] PSI background task failed: {future_error}")
                    psi_value = None
                subject_ctx, accuracy_ctx = pending_psi_context or (last_label, last_confidence)
                psi_future = None
                pending_psi_context = None

                if psi_value is not None:
                    last_psi = psi_value
                    log_event(
                        phase="phase2",
                        event="psi_window",
                        subject=subject_ctx or "unknown",
                        psi=psi_value,
                        accuracy=accuracy_ctx if accuracy_ctx is not None else last_confidence,
                        extra_info="PSI window"
                    )
                    print(f"[INFO] PSI: {psi_value:.4f}")

                    if psi_value >= DRIFT_THRESHOLD_PSI:
                        print("[ALERT] DRIFT DETECTED!")
                        log_event(
                            phase="phase2",
                            event="DRIFT_ALERT",
                            subject=subject_ctx or "unknown",
                            psi=psi_value,
                            accuracy=accuracy_ctx if accuracy_ctx is not None else last_confidence,
                            extra_info=f"Drift PSI={psi_value}"
                        )

            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if FRAME_SKIP > 1 and frame_count % FRAME_SKIP != 0:
                try:
                    cv2.imshow("Monitoring", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    pass
                time.sleep(0.002)
                continue

            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            try:
                faces = app.get(frame)
                time.sleep(0.002)  # Reduce GPU load
            except Exception:
                faces = []

            if len(faces) == 0:
                try:
                    cv2.imshow("Monitoring", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    pass
                time.sleep(0.002)
                continue

            face = faces[0]
            x1, y1, x2, y2 = face.bbox.astype(int)

            embedding, label, confidence = recognize_face(face)

            if embedding is None:
                box_color = get_box_color(last_psi)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            else:
                last_confidence = confidence
                last_label = label
                embedding_window.append(embedding)

                box_color = get_box_color(last_psi)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                text = f"{label} | Acc:{confidence:.2f}"
                if last_psi is not None:
                    text += f" | PSI:{last_psi:.3f}"

                cv2.putText(
                    frame, text, (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2
                )

            try:
                cv2.imshow("Monitoring", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            except cv2.error:
                pass

            # PSI WINDOW
            if len(embedding_window) >= WINDOW_SIZE:
                if psi_future is None:
                    window_arr = np.vstack(embedding_window)
                    pending_psi_context = (last_label, last_confidence)
                    psi_future = psi_executor.submit(
                        compute_psi,
                        baseline_embeddings_df.copy(deep=False),
                        window_arr
                    )
                    embedding_window = []
                else:
                    # Keep recent embeddings until the previous PSI task completes
                    embedding_window = embedding_window[-WINDOW_SIZE:]

            # FPS limiter
            elapsed = time.time() - loop_start
            if elapsed < FRAME_DELAY:
                time.sleep(FRAME_DELAY - elapsed)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        psi_executor.shutdown(wait=False)


if __name__ == "__main__":
    main()
