import os
import pandas as pd
import numpy as np
import joblib
from deepface import DeepFace
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# --- PROJECT CONFIGURATION (Adjust to match args in recognition_app.py) ---
DB_DIR = "baseline_db" # Database directory
OUTPUT_ASSETS_DIR = "initial_assets" # Output 

# Adjust model and detector used in the main application
EMBEDDING_MODEL = "ArcFace" 
DETECTOR = "retinaface" 
K_VALUE = 1 

# Ensure the output folder exists
os.makedirs(OUTPUT_ASSETS_DIR, exist_ok=True)
OUTPUT_EMBEDDINGS_PATH = os.path.join(OUTPUT_ASSETS_DIR, "baseline_embeddings.csv")
OUTPUT_MODEL_PATH = os.path.join(OUTPUT_ASSETS_DIR, "initial_knn_model.pkl")

# --- Helper Function ---
def get_arcface_embedding(img_path):
    """Generates ArcFace embedding, suppressing errors."""
    try:
        result = DeepFace.represent(
            img_path=img_path, 
            model_name=EMBEDDING_MODEL, 
            detector_backend=DETECTOR, 
            enforce_detection=True 
        )
        # Retrieves the embedding of the first detected face
        return result[0]['embedding']
    except Exception:
        return None

# --- Step 1: Generate Reference Embeddings from facial_db ---
print(f"--- 1. Generating {EMBEDDING_MODEL} Embeddings from {DB_DIR} ---")
embeddings_list = []
labels_list = []

# Iterate through subject folders (Hansen, Jeniffer, Marvel, etc.)
for subject_folder in os.listdir(DB_DIR):
    subject_label = subject_folder 
    subject_path = os.path.join(DB_DIR, subject_folder)

    if os.path.isdir(subject_path):
        print(f"-> Processing Subject: {subject_label}")
        
        # Iterate through images inside the subject folder
        for filename in os.listdir(subject_path):
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                file_path = os.path.join(subject_path, filename)
                
                embedding = get_arcface_embedding(file_path)
                
                if embedding is not None:
                    embeddings_list.append(embedding)
                    labels_list.append(subject_label) # Folder name as label
                else:
                    print(f"  [SKIP] Failed to detect face in {filename} ({subject_label})")

if not embeddings_list:
    print("\n[CRITICAL ERROR] Total embeddings saved: 0. Check your 'baseline_db' images.")
    df_embeddings = pd.DataFrame()
else:
    df_embeddings = pd.DataFrame(embeddings_list)
    df_embeddings['label'] = labels_list
    
# Save as the Reference Distribution (Baseline)
df_embeddings.to_csv(OUTPUT_EMBEDDINGS_PATH, index=False)
print(f"\nTotal unique subjects: {df_embeddings['label'].nunique()}")
print(f"Total embeddings saved: {len(df_embeddings)}")
print(f"**Reference Embeddings saved to: {OUTPUT_EMBEDDINGS_PATH}**")

# --- Step 2: Train Initial Model ---
print("\n--- 2. Training Initial k-NN Classifier ---")

if df_embeddings.empty:
    print("[ERROR] Cannot train model: No embeddings extracted.")
else:
    X_train = df_embeddings.drop(columns=['label']).values 
    y_train = df_embeddings['label'].values 
    
    knn_model = KNeighborsClassifier(n_neighbors=K_VALUE)
    knn_model.fit(X_train, y_train)
    
    # Evaluate accuracy (Optional: use cross-validation for a more valid result)
    y_pred = knn_model.predict(X_train)
    baseline_accuracy = accuracy_score(y_train, y_pred) * 100
    
    print(f"**BASELINE ACCURACY: {baseline_accuracy:.2f}%**")
    
    joblib.dump(knn_model, OUTPUT_MODEL_PATH)
    print(f"**Initial Model saved to: {OUTPUT_MODEL_PATH}**")

print("\n--- INITIAL MODEL UPLOAD COMPLETE ---")