from flask import Flask, render_template, Response, jsonify, request
import cv2
import os
import json
import pandas as pd
from datetime import datetime
import logging
import threading
import time
from collections import deque
import numpy as np
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from insightface.utils.face_align import norm_crop
import onnxruntime as ort
import joblib

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============ CONFIGURATION ============
STREAM_URL = "http://192.168.0.151:8080/video"
ARC_MODEL_PATH = "models/arcface_w600k_r50.onnx"
BASELINE_CSV = "initial_assets/baseline_embeddings.csv"
KNN_PATH = "initial_assets/initial_knn_model.pkl"
LE_PATH = "initial_assets/label_encoder.pkl"
CENTROIDS_PATH = "initial_assets/centroids.pkl"
LOG_FILE = "logs/full_system_log.csv"
DRIFT_LOG = "monitoring_log.csv"

WINDOW = 20
PSI_THRESHOLD = 0.10
MIN_KNN_CONFIDENCE = 0.75
MIN_CENTROID_SIM = 0.35

# Global variables
camera = None
monitoring_thread = None
monitoring_active = False
latest_frame = None
current_stats = {
    'total_detections': 0,
    'total_faces': 0,
    'current_person': 'Unknown',
    'confidence': 0.0,
    'psi_status': 'OK',
    'last_detection_time': None
}

# ============ HELPER FUNCTIONS ============
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

# ============ MODELS INITIALIZATION ============
def init_models():
    global det, fa, arc, arc_input, knn, le, centroids, baseline_per_person, buffers
    
    logger.info("[INIT] Loading YOLOv8-face detector...")
    det = YOLO("models/yolov8s-face.pt")
    
    logger.info("[INIT] Loading InsightFace landmarks...")
    fa = FaceAnalysis(name="buffalo_l")
    fa.prepare(ctx_id=0, det_size=(512, 512))
    
    logger.info("[INIT] Loading ArcFace ONNX...")
    arc = ort.InferenceSession(
        ARC_MODEL_PATH,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    arc_input = arc.get_inputs()[0].name
    
    logger.info("[INIT] Loading baselines, model & centroids...")
    df = pd.read_csv(BASELINE_CSV)
    knn = joblib.load(KNN_PATH)
    le = joblib.load(LE_PATH)
    centroids = joblib.load(CENTROIDS_PATH)
    
    baseline_per_person = {
        person: df[df["label"] == person].iloc[:, :-1].values
        for person in df["label"].unique()
    }
    
    buffers = {p: deque(maxlen=WINDOW) for p in baseline_per_person.keys()}
    
    logger.info("[INIT] Models loaded successfully!")

def get_embedding(frame):
    """Extract ArcFace embedding from biggest face."""
    global det, fa, arc, arc_input
    
    results = det(frame, verbose=False)
    if len(results[0].boxes) == 0:
        return None, None, None

    boxes = results[0].boxes.xyxy.cpu().numpy()
    x1, y1, x2, y2 = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])).astype(int)

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = expand_box(x1, y1, x2, y2, w, h)

    face_crop = frame[y1:y2, x1:x2]
    faces = fa.get(face_crop)
    if len(faces) == 0:
        return None, None, None

    kps = faces[0].kps + np.array([x1, y1])
    aligned = norm_crop(frame, kps)

    resized = cv2.resize(aligned, (112, 112))
    blob = resized[:, :, ::-1].astype(np.float32)
    blob = (blob - 127.5) / 128.0
    blob = np.transpose(blob, (2, 0, 1))[None]

    emb = arc.run(None, {arc_input: blob})[0][0]
    return emb, aligned, (x1, y1, x2, y2)

# ============ MONITORING THREAD ============
def monitoring_worker():
    global monitoring_active, latest_frame, current_stats, camera
    global knn, le, centroids, baseline_per_person, buffers
    
    logger.info(f"[CAM] Opening stream: {STREAM_URL}")
    camera = cv2.VideoCapture(STREAM_URL)
    
    if not camera.isOpened():
        logger.error("Cannot open camera stream.")
        return
    
    logger.info("[RUN] Real-time drift monitoring started...")
    
    while monitoring_active:
        ret, frame = camera.read()
        if not ret:
            time.sleep(0.1)
            continue
        
        # Store frame for video feed
        latest_frame = frame.copy()
        
        emb, aligned, bbox = get_embedding(frame)
        if emb is None:
            continue
        
        # Update stats
        current_stats['total_detections'] += 1
        current_stats['total_faces'] += 1
        
        # KNN Recognition
        pred_enc = knn.predict([emb])[0]
        probs = knn.predict_proba([emb])[0]
        cls_idx = int(np.where(knn.classes_ == pred_enc)[0][0])
        
        confidence = float(probs[cls_idx])
        person = le.inverse_transform([pred_enc])[0]
        
        # Centroid consistency
        if person not in centroids:
            continue
        
        sim = cos_sim(emb, centroids[person])
        
        if confidence < MIN_KNN_CONFIDENCE or sim < MIN_CENTROID_SIM:
            logger.warning(f"[REJECT] '{person}' conf={confidence:.3f}, sim={sim:.3f}")
            continue
        
        # Update current stats
        current_stats['current_person'] = person
        current_stats['confidence'] = confidence
        current_stats['last_detection_time'] = datetime.now().isoformat()
        
        # Draw detection on frame
        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(latest_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{person} ({confidence:.2f})"
            cv2.putText(latest_frame, text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # PSI Drift Detection
        buffers[person].append(emb)
        
        if len(buffers[person]) >= WINDOW:
            base = baseline_per_person[person]
            curr = np.array(buffers[person])
            p = psi(base, curr)
            
            status = "DRIFT" if p > PSI_THRESHOLD else "OK"
            current_stats['psi_status'] = status
            current_stats['psi_value'] = p
            
            logger.info(f"[PSI] {person}: psi={p:.3f} → {status}")
            
            # Save to drift log
            ts = time.time()
            row = pd.DataFrame([[ts, person, p, status]],
                             columns=["timestamp", "person", "psi", "status"])
            row.to_csv(DRIFT_LOG, mode="a", header=False, index=False)
        
        time.sleep(0.03)  # ~30 FPS
    
    if camera:
        camera.release()
    logger.info("[EXIT] Monitoring stopped.")

def start_monitoring():
    global monitoring_thread, monitoring_active
    
    if monitoring_thread and monitoring_thread.is_alive():
        logger.warning("Monitoring already running!")
        return
    
    monitoring_active = True
    monitoring_thread = threading.Thread(target=monitoring_worker, daemon=True)
    monitoring_thread.start()
    logger.info("Monitoring thread started.")

def stop_monitoring():
    global monitoring_active
    monitoring_active = False
    logger.info("Stopping monitoring...")

# ============ FLASK ROUTES ============
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/monitor')
def monitor():
    return render_template('monitor.html')

@app.route('/logs')
def logs():
    return render_template('logs.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            if latest_frame is not None:
                ret, buffer = cv2.imencode('.jpg', latest_frame)
                if ret:
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.03)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/logs')
def get_logs():
    try:
        if os.path.exists(LOG_FILE):
            df = pd.read_csv(LOG_FILE)
            recent_logs = df.tail(100).to_dict('records')
            return jsonify({'success': True, 'logs': recent_logs})
        else:
            return jsonify({'success': False, 'message': 'Log file not found'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/drift_logs')
def get_drift_logs():
    try:
        if os.path.exists(DRIFT_LOG):
            df = pd.read_csv(DRIFT_LOG)
            recent_logs = df.tail(50).to_dict('records')
            return jsonify({'success': True, 'logs': recent_logs})
        else:
            return jsonify({'success': False, 'message': 'Drift log not found'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/stats')
def get_stats():
    return jsonify({
        'success': True,
        'stats': current_stats
    })

@app.route('/api/status')
def system_status():
    return jsonify({
        'success': True,
        'monitoring_active': monitoring_active,
        'camera_connected': camera is not None and camera.isOpened() if camera else False,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/control/start', methods=['POST'])
def start_monitoring_api():
    try:
        start_monitoring()
        return jsonify({'success': True, 'message': 'Monitoring started'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/control/stop', methods=['POST'])
def stop_monitoring_api():
    try:
        stop_monitoring()
        return jsonify({'success': True, 'message': 'Monitoring stopped'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

# ============ MAIN ============
if __name__ == '__main__':
    # Initialize models on startup
    init_models()
    
    # Start monitoring automatically
    start_monitoring()
    
    # Run Flask (accessible from WSL and Windows)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)