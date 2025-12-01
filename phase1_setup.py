import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from insightface.utils.face_align import norm_crop
import onnxruntime as ort
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# ---------------- CONFIG ----------------
BASELINE_DIR = "baseline_db"       # folder with subfolders per person
SAVE_DIR = "initial_assets"
ARC_MODEL_PATH = "models/arcface_w600k_r50.onnx"
YOLO_WEIGHTS = "models/yolov8s-face.pt"

os.makedirs(SAVE_DIR, exist_ok=True)


def expand_box(x1, y1, x2, y2, img_w, img_h, scale=0.60):
    """Expand a face box by given scale, clamped to image boundaries."""
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


# ---------------- LOAD MODELS ----------------
print("[INIT] Loading YOLOv8-face detector...")
detector = YOLO(YOLO_WEIGHTS)

print("[INIT] Loading InsightFace (landmarks only)...")
landmark_app = FaceAnalysis(name="buffalo_l")
landmark_app.prepare(ctx_id=0, det_size=(512, 512))

print("[INIT] Loading ArcFace ONNX...")
arcface = ort.InferenceSession(
    ARC_MODEL_PATH,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
arc_input = arcface.get_inputs()[0].name


def extract_embedding(img_bgr):
    """Detect biggest face, align using landmarks, get 512-D ArcFace embedding."""
    results = detector(img_bgr, verbose=False)
    if results[0].boxes is None or len(results[0].boxes) == 0:
        return None

    # biggest face
    boxes = results[0].boxes.xyxy.cpu().numpy()
    box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    x1, y1, x2, y2 = box.astype(int)

    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = expand_box(x1, y1, x2, y2, w, h, scale=0.60)

    face_crop = img_bgr[y1:y2, x1:x2]

    faces = landmark_app.get(face_crop)
    if len(faces) == 0:
        return None

    f = faces[0]
    kps = f.kps + np.array([x1, y1])
    aligned = norm_crop(img_bgr, kps)

    # ArcFace preprocessing
    resized = cv2.resize(aligned, (112, 112))
    blob = resized[:, :, ::-1].astype(np.float32)
    blob = (blob - 127.5) / 128.0
    blob = np.transpose(blob, (2, 0, 1))[None, :, :, :]

    emb = arcface.run(None, {arc_input: blob})[0][0]
    return emb


# ---------------- EXTRACT BASELINE EMBEDDINGS ----------------
all_embs = []
all_labels = []

print("\n--- Extracting baseline embeddings (aligned ArcFace) ---\n")

for person in sorted(os.listdir(BASELINE_DIR)):
    person_dir = os.path.join(BASELINE_DIR, person)
    if not os.path.isdir(person_dir):
        continue

    print(f"[SUBJECT] {person}")

    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"[SKIP] Cannot read: {img_name}")
            continue

        emb = extract_embedding(img)
        if emb is None:
            print(f"[SKIP] No valid face/landmarks: {img_name}")
            continue

        all_embs.append(emb)
        all_labels.append(person)

print("\n[INFO] Embedding extraction complete.")
print("Total samples:", len(all_embs))

if len(all_embs) == 0:
    raise RuntimeError("No embeddings extracted. Check baseline_db images & models.")


# ---------------- SAVE BASELINE EMBEDDINGS ----------------
df = pd.DataFrame(all_embs)
df["label"] = all_labels

baseline_csv_path = os.path.join(SAVE_DIR, "baseline_embeddings.csv")
df.to_csv(baseline_csv_path, index=False)
print(f"[SAVE] baseline_embeddings.csv saved at {baseline_csv_path}")


# ---------------- TRAIN KNN + LABEL ENCODER ----------------
print("\n--- Training KNN model ---\n")

X = df.iloc[:, :-1].astype(float).values
y = df["label"].values

le = LabelEncoder()
y_encoded = le.fit_transform(y)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y_encoded)

knn_path = os.path.join(SAVE_DIR, "initial_knn_model.pkl")
le_path = os.path.join(SAVE_DIR, "label_encoder.pkl")

joblib.dump(knn, knn_path)
joblib.dump(le, le_path)

print(f"[SAVE] KNN model saved at {knn_path}")
print(f"[SAVE] LabelEncoder saved at {le_path}")


# ---------------- COMPUTE & SAVE CENTROIDS ----------------
print("\n--- Computing centroids per identity ---\n")

centroids = {}
for person in df["label"].unique():
    embs = df[df["label"] == person].iloc[:, :-1].values
    centroids[person] = np.mean(embs, axis=0)

centroids_path = os.path.join(SAVE_DIR, "centroids.pkl")
joblib.dump(centroids, centroids_path)
print(f"[SAVE] Centroids saved at {centroids_path}")

print("\n[DONE] Phase 1 baseline creation finished.")
