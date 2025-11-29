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

BASELINE_DIR = "baseline_db"
SAVE_DIR = "initial_assets"
ARC_MODEL_PATH = "models/arcface_w600k_r50.onnx"

os.makedirs(SAVE_DIR, exist_ok=True)


# ---------------------------
# EXPANDED FACE BOX
# ---------------------------
def expand_box(x1, y1, x2, y2, img_w, img_h, scale=0.60):
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2

    new_w = w * (1 + scale)
    new_h = h * (1 + scale)

    new_x1 = int(max(0, cx - new_w / 2))
    new_y1 = int(max(0, cy - new_h / 2))
    new_x2 = int(min(img_w, cx + new_w / 2))
    new_y2 = int(min(img_h, cy + new_h / 2))

    return new_x1, new_y1, new_x2, new_y2


# ---------------------------
# LOAD MODELS
# ---------------------------
print("[INIT] Loading YOLOv8-face detector...")
detector = YOLO("models/yolov8s-face.pt")

print("[INIT] Loading InsightFace (landmarks only)...")
landmark_app = FaceAnalysis(name="buffalo_l")
landmark_app.prepare(ctx_id=0, det_size=(512, 512))

print("[INIT] Loading ArcFace ONNX...")
arcface = ort.InferenceSession(
    ARC_MODEL_PATH,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
arc_input = arcface.get_inputs()[0].name


# ---------------------------
# EXTRACT EMBEDDINGS
# ---------------------------
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

        # detect face
        results = detector(img, verbose=False)
        if results[0].boxes is None or len(results[0].boxes) == 0:
            print(f"[SKIP] No YOLO face: {img_name}")
            continue

        # largest face
        box = results[0].boxes.xyxy.cpu().numpy()
        box = max(box, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
        x1, y1, x2, y2 = box.astype(int)

        # expand bounding box
        h, w = img.shape[:2]
        x1, y1, x2, y2 = expand_box(x1, y1, x2, y2, w, h, scale=0.60)

        face_crop = img[y1:y2, x1:x2]

        # detect landmarks INSIDE CROPPED FACE
        faces = landmark_app.get(face_crop)
        if len(faces) == 0:
            print(f"[SKIP] No landmarks: {img_name}")
            continue

        face = faces[0]
        kps = face.kps + np.array([x1, y1])  # shift keypoints
        aligned = norm_crop(img, kps)

        # preprocess for arcface
        resized = cv2.resize(aligned, (112, 112))
        blob = resized[:, :, ::-1].astype(np.float32)
        blob = (blob - 127.5) / 128.0
        blob = np.transpose(blob, (2, 0, 1))[None, :, :, :]

        emb = arcface.run(None, {arc_input: blob})[0][0]

        all_embs.append(emb)
        all_labels.append(person)

print("\n[INFO] Embedding extraction complete.")
print("Total samples:", len(all_embs))


# ---------------------------
# SAVE BASELINE EMBEDDINGS
# ---------------------------
df = pd.DataFrame(all_embs)
df["label"] = all_labels

df.to_csv(f"{SAVE_DIR}/baseline_embeddings.csv", index=False)
print(f"[SAVE] baseline_embeddings.csv saved.")


# ---------------------------
# TRAIN KNN
# ---------------------------
print("\n--- Training KNN model ---\n")

X = df.iloc[:, :-1].astype(float).values
y = df["label"].values

le = LabelEncoder()
y_encoded = le.fit_transform(y)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y_encoded)

joblib.dump(knn, f"{SAVE_DIR}/initial_knn_model.pkl")
joblib.dump(le, f"{SAVE_DIR}/label_encoder.pkl")

print("[SAVE] Model saved.")
print("[DONE] Phase 1 baseline creation finished.")
