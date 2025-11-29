import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from insightface.utils.face_align import norm_crop
import onnxruntime as ort
import joblib
from scipy.spatial.distance import cdist

STREAM_URL = "http://192.168.0.100:8080/video"
ARC_MODEL_PATH = "models/arcface_w600k_r50.onnx"
BASELINE_CSV = "initial_assets/baseline_embeddings.csv"
KNN_PATH = "initial_assets/initial_knn_model.pkl"
LE_PATH = "initial_assets/label_encoder.pkl"

WINDOW = 20
PSI_THRESHOLD = 0.25   # good threshold for your case


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
    base = np.array(base)
    curr = np.array(curr)
    edges = np.linspace(min(base.min(), curr.min()), max(base.max(), curr.max()), buckets+1)
    b_hist, _ = np.histogram(base, bins=edges)
    c_hist, _ = np.histogram(curr, bins=edges)
    b_hist = b_hist / (len(base) + 1e-8)
    c_hist = c_hist / (len(curr) + 1e-8)
    psi_vals = (c_hist - b_hist) * np.log((c_hist + 1e-8) / (b_hist + 1e-8))
    return np.sum(psi_vals)


# ---------------- LOAD MODELS ------------------
print("[INIT] Loading YOLOv8-face detector...")
det = YOLO("models/yolov8s-face.pt")

print("[INIT] Loading InsightFace landmark model...")
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(512, 512))

print("[INIT] Loading ArcFace ONNX...")
arc = ort.InferenceSession(
    ARC_MODEL_PATH,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
arc_in = arc.get_inputs()[0].name

print("[INIT] Loading baseline embeddings & models...")
df = pd.read_csv(BASELINE_CSV)
knn = joblib.load(KNN_PATH)
le = joblib.load(LE_PATH)

baseline_per_person = {
    person: df[df["label"] == person].iloc[:, :-1].values
    for person in df["label"].unique()
}


# ---------------- START CAMERA ------------------
print("[CAM] Opening stream:", STREAM_URL)
cap = cv2.VideoCapture(STREAM_URL)

if not cap.isOpened():
    raise RuntimeError("Cannot open camera stream.")

print("[RUN] Headless real-time drift monitoring started.")
print("Press CTRL+C to stop.\n")

buffers = {person: [] for person in baseline_per_person.keys()}

while True:
    ret, frame = cap.read()
    if not ret:
        print("[WARN] No frame received.")
        continue

    # YOLO detect
    res = det(frame, verbose=False)
    if len(res[0].boxes) == 0:
        continue

    box = res[0].boxes.xyxy.cpu().numpy()[0].astype(int)
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = expand_box(*box, w, h, scale=0.60)

    # crop expanded face
    face_crop = frame[y1:y2, x1:x2]

    faces = app.get(face_crop)
    if len(faces) == 0:
        continue

    f = faces[0]
    kps = f.kps + np.array([x1, y1])
    aligned = norm_crop(frame, kps)

    # arcface preprocess
    resized = cv2.resize(aligned, (112, 112))
    blob = resized[:, :, ::-1].astype(np.float32)
    blob = (blob - 127.5) / 128.0
    blob = np.transpose(blob, (2, 0, 1))[None, :, :, :]

    emb = arc.run(None, {arc_in: blob})[0][0]

    # recognize
    pred = knn.predict([emb])[0]
    person = le.inverse_transform([pred])[0]

    buffers[person].append(emb)
    if len(buffers[person]) > WINDOW:
        buffers[person].pop(0)

    # compute PSI
    if len(buffers[person]) == WINDOW:
        base = baseline_per_person[person]
        curr = np.array(buffers[person])
        p = psi(base.flatten(), curr.flatten())

        status = "DRIFT" if p > PSI_THRESHOLD else "OK"
        print(f"[PSI] {person}: {p:.3f}  -> {status}")
