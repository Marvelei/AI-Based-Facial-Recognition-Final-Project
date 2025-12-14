import os
import sys
import argparse
import cv2
import numpy as np
import pandas as pd
import joblib
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from insightface.utils.face_align import norm_crop
import onnxruntime as ort
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from colorama import Fore, Style, init
from collections import deque

# Initialize colorama
init(autoreset=True)

# ============ CONFIGURATION ============
ARC_MODEL_PATH = "models/arcface_w600k_r50.onnx"
BASELINE_CSV = "initial_assets/baseline_embeddings.csv"
KNN_PATH = "initial_assets/initial_knn_model.pkl"
LE_PATH = "initial_assets/label_encoder.pkl"
CENTROIDS_PATH = "initial_assets/centroids.pkl"
REGISTRATION_DIR = "registration_photos"  # Where uploaded photos are temporarily stored
WINDOW = 20

os.makedirs(REGISTRATION_DIR, exist_ok=True)


def log(msg, color="white"):
    """Print colored log messages"""
    colors = {
        "red": Fore.RED,
        "green": Fore.GREEN,
        "yellow": Fore.YELLOW,
        "cyan": Fore.CYAN,
        "magenta": Fore.MAGENTA,
        "blue": Fore.BLUE,
        "white": Fore.WHITE,
    }
    c = colors.get(color, Fore.WHITE)
    print(c + msg + Style.RESET_ALL)


def expand_box(x1, y1, x2, y2, img_w, img_h, scale=0.60):
    """Expand bounding box"""
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


def get_embedding(frame, det, fa, arc, arc_input):
    """Extract ArcFace embedding from biggest face in frame"""
    results = det(frame, verbose=False)
    if len(results[0].boxes) == 0:
        return None

    boxes = results[0].boxes.xyxy.cpu().numpy()
    x1, y1, x2, y2 = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])).astype(int)

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = expand_box(x1, y1, x2, y2, w, h)

    face_crop = frame[y1:y2, x1:x2]
    faces = fa.get(face_crop)
    if len(faces) == 0:
        return None

    kps = faces[0].kps + np.array([x1, y1])
    aligned = norm_crop(frame, kps)

    resized = cv2.resize(aligned, (112, 112))
    blob = resized[:, :, ::-1].astype(np.float32)
    blob = (blob - 127.5) / 128.0
    blob = np.transpose(blob, (2, 0, 1))[None]

    emb = arc.run(None, {arc_input: blob})[0][0]
    return emb


def register_person(name, photo_dir):
    """
    Register a new person by processing their photos
    
    Args:
        name: Person's name
        photo_dir: Directory containing their photos
    
    Returns:
        dict: Status of registration with success flag and message
    """
    log(f"\n[REGISTER] Starting registration for: {name}", "cyan")
    
    # Check if photo directory exists
    if not os.path.exists(photo_dir):
        return {"success": False, "message": f"Photo directory not found: {photo_dir}"}
    
    # Get list of image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    photo_files = [
        os.path.join(photo_dir, f) 
        for f in os.listdir(photo_dir) 
        if f.lower().endswith(image_extensions)
    ]
    
    if len(photo_files) < 3:
        return {
            "success": False, 
            "message": f"Not enough photos. Found {len(photo_files)}, need at least 3"
        }
    
    log(f"[INFO] Found {len(photo_files)} photos", "green")
    
    # ============ LOAD MODELS ============
    log("[INIT] Loading models...", "cyan")
    
    try:
        det = YOLO("models/yolov8s-face.pt")
        fa = FaceAnalysis(name="buffalo_l")
        fa.prepare(ctx_id=0, det_size=(512, 512))
        arc = ort.InferenceSession(
            ARC_MODEL_PATH,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        arc_input = arc.get_inputs()[0].name
    except Exception as e:
        return {"success": False, "message": f"Failed to load models: {str(e)}"}
    
    # ============ EXTRACT EMBEDDINGS ============
    log("[PROCESS] Extracting face embeddings...", "cyan")
    
    embeddings = []
    processed = 0
    failed = 0
    
    for i, photo_path in enumerate(photo_files, 1):
        log(f"Processing {i}/{len(photo_files)}: {os.path.basename(photo_path)}", "white")
        
        img = cv2.imread(photo_path)
        if img is None:
            log(f"  ✗ Failed to read image", "red")
            failed += 1
            continue
        
        emb = get_embedding(img, det, fa, arc, arc_input)
        if emb is not None:
            embeddings.append(emb)
            processed += 1
            log(f"  ✓ Extracted embedding", "green")
        else:
            log(f"  ✗ No face detected", "yellow")
            failed += 1
    
    if len(embeddings) < 3:
        return {
            "success": False,
            "message": f"Not enough valid faces. Got {len(embeddings)} from {len(photo_files)} photos"
        }
    
    log(f"\n[SUCCESS] Extracted {len(embeddings)} embeddings", "green")
    log(f"[INFO] Success: {processed}, Failed: {failed}", "white")
    
    # ============ UPDATE BASELINE DATA ============
    log("\n[UPDATE] Updating baseline database...", "cyan")
    
    try:
        # Load existing baseline
        df_existing = pd.read_csv(BASELINE_CSV)
        
        # Create new rows
        new_rows = []
        for emb in embeddings:
            row = list(emb) + [name]
            new_rows.append(row)
        
        # Combine with existing data
        columns = df_existing.columns.tolist()
        df_new = pd.DataFrame(new_rows, columns=columns)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(BASELINE_CSV, index=False)
        
        log(f"[SUCCESS] Added {len(embeddings)} embeddings to baseline", "green")
        
    except Exception as e:
        return {"success": False, "message": f"Failed to update baseline: {str(e)}"}
    
    # ============ UPDATE CENTROIDS ============
    log("[UPDATE] Updating centroids...", "cyan")
    
    try:
        centroids = joblib.load(CENTROIDS_PATH)
        person_embeddings = np.array(embeddings)
        centroids[name] = np.mean(person_embeddings, axis=0)
        joblib.dump(centroids, CENTROIDS_PATH)
        log("[SUCCESS] Centroids updated", "green")
    except Exception as e:
        return {"success": False, "message": f"Failed to update centroids: {str(e)}"}
    
    # ============ UPDATE LABEL ENCODER ============
    log("[UPDATE] Updating label encoder...", "cyan")
    
    try:
        all_labels = df_combined['label'].unique()
        le = LabelEncoder()
        le.fit(all_labels)
        joblib.dump(le, LE_PATH)
        log("[SUCCESS] Label encoder updated", "green")
    except Exception as e:
        return {"success": False, "message": f"Failed to update label encoder: {str(e)}"}
    
    # ============ RETRAIN KNN MODEL ============
    log("[TRAIN] Retraining KNN model...", "cyan")
    
    try:
        X = df_combined.iloc[:, :-1].values
        y = df_combined['label'].values
        y_encoded = le.transform(y)
        
        knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')
        knn.fit(X, y_encoded)
        joblib.dump(knn, KNN_PATH)
        
        log("[SUCCESS] KNN model retrained", "green")
    except Exception as e:
        return {"success": False, "message": f"Failed to retrain model: {str(e)}"}
    
    # ============ FINAL SUCCESS ============
    log(f"\n{'='*60}", "green")
    log(f"✓ REGISTRATION COMPLETE", "green")
    log(f"  Person: {name}", "green")
    log(f"  Embeddings: {len(embeddings)}", "green")
    log(f"  Total people in system: {len(all_labels)}", "green")
    log(f"{'='*60}\n", "green")
    
    return {
        "success": True,
        "message": f"Successfully registered {name}",
        "embeddings_count": len(embeddings),
        "photos_processed": processed,
        "photos_failed": failed
    }


def main():
    parser = argparse.ArgumentParser(description="Register a new person in the face recognition system")
    parser.add_argument("--name", required=True, help="Person's name")
    parser.add_argument("--photo-dir", required=True, help="Directory containing person's photos")
    
    args = parser.parse_args()
    
    result = register_person(args.name, args.photo_dir)
    
    if result["success"]:
        sys.exit(0)
    else:
        log(f"\n[ERROR] {result['message']}", "red")
        sys.exit(1)


if __name__ == "__main__":
    main()