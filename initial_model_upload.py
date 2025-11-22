import os
import pandas as pd
import numpy as np
import joblib
from deepface import DeepFace
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

DB_DIR = "baseline_db"                 # Folder where each person has images
OUTPUT_ASSETS_DIR = "initial_assets"   # Output folder

EMBEDDING_MODEL = "ArcFace"
DETECTOR = "retinaface"
K_VALUE = 1

os.makedirs(OUTPUT_ASSETS_DIR, exist_ok=True)
OUTPUT_EMBEDDINGS_PATH = os.path.join(OUTPUT_ASSETS_DIR, "baseline_embeddings.csv")
OUTPUT_MODEL_PATH = os.path.join(OUTPUT_ASSETS_DIR, "initial_knn_model.pkl")

def get_arcface_embedding(img_path):
    """
    Generates ArcFace embedding with auto-fix to ensure 512 dimensions.
    This avoids the rare ArcFace 513-dim bug.
    """
    try:
        result = DeepFace.represent(
            img_path=img_path,
            model_name="ArcFace",
            detector_backend=DETECTOR,
            enforce_detection=True
        )
        emb = np.array(result[0]["embedding"])

        # FIX: If ArcFace returns 513 dims, slice to 512
        if emb.shape[0] == 513:
            emb = emb[:512]

        # Validate embedding length
        if emb.shape[0] != 512:
            print(f"[WARNING] Invalid embedding dim ({emb.shape[0]}) → skipping")
            return None

        return emb.tolist()

    except Exception:
        return None

print(f"--- 1. Generating {EMBEDDING_MODEL} embeddings from {DB_DIR} ---")

embeddings_list = []
labels_list = []

for subject_folder in os.listdir(DB_DIR):
    subject_path = os.path.join(DB_DIR, subject_folder)

    if not os.path.isdir(subject_path):
        continue

    print(f"-> Processing Subject: {subject_folder}")

    for filename in os.listdir(subject_path):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(subject_path, filename)

            emb = get_arcface_embedding(img_path)

            if emb is not None:
                embeddings_list.append(emb)
                labels_list.append(subject_folder)
            else:
                print(f"  [SKIP] Failed: {filename}")


# Save dataframe
if len(embeddings_list) == 0:
    print("\n[CRITICAL ERROR] No embeddings extracted at all! Check your images.")
    df_embeddings = pd.DataFrame()
else:
    df_embeddings = pd.DataFrame(embeddings_list)
    df_embeddings["label"] = labels_list

df_embeddings.to_csv(OUTPUT_EMBEDDINGS_PATH, index=False)

print(f"\nTotal subjects: {df_embeddings['label'].nunique()}")
print(f"Total embeddings saved: {len(df_embeddings)}")
print(f"Saved baseline to: {OUTPUT_EMBEDDINGS_PATH}")


# ==========================================================
# STEP 2 — Train initial kNN classifier
# ==========================================================
print("\n--- 2. Training Initial k-NN Classifier ---")

if df_embeddings.empty:
    print("[ERROR] Cannot train: No embeddings.")
else:
    X_train = df_embeddings.drop(columns=["label"]).values
    y_train = df_embeddings["label"].values

    knn_model = KNeighborsClassifier(n_neighbors=K_VALUE)
    knn_model.fit(X_train, y_train)

    # Check baseline accuracy
    y_pred = knn_model.predict(X_train)
    acc = accuracy_score(y_train, y_pred) * 100

    print(f"BASELINE ACCURACY: {acc:.2f}%")

    joblib.dump(knn_model, OUTPUT_MODEL_PATH)
    print(f"Initial model saved to: {OUTPUT_MODEL_PATH}")

    # Print final embedding length
    print("Baseline embedding length (should be 512):", X_train.shape[1])

print("\n--- INITIAL MODEL UPLOAD COMPLETE ---")
