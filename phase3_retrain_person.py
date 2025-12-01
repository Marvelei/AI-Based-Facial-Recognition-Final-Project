# phase3_retrain_person.py

import argparse
import os

import numpy as np
import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

BASELINE_CSV = "initial_assets/baseline_embeddings.csv"
KNN_PATH = "initial_assets/initial_knn_model.pkl"
LE_PATH = "initial_assets/label_encoder.pkl"
CENTROIDS_PATH = "initial_assets/centroids.pkl"
DRIFT_DIR = "drift_assets"


def load_numeric_embeddings(df: pd.DataFrame):
    """Return only numeric embedding columns as a matrix."""
    numeric_df = df.select_dtypes(include=[np.number])
    return numeric_df.values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--person", required=True, help="Person to retrain")
    args = parser.parse_args()
    person = args.person

    print(f"[PHASE 3] Starting retrain for: {person}")

    drift_path = os.path.join(DRIFT_DIR, f"{person}_drift_embeddings.npy")
    if not os.path.exists(drift_path):
        print(f"[ERROR] No drift embeddings found for {person} at {drift_path}")
        return

    drift_embs = np.load(drift_path)
    if drift_embs.ndim == 1:
        drift_embs = drift_embs.reshape(1, -1)

    print(f"[INFO] Loaded {len(drift_embs)} drift embeddings for '{person}'")

    if not os.path.exists(BASELINE_CSV):
        print(f"[ERROR] Baseline CSV not found at {BASELINE_CSV}")
        return

    # ------------ LOAD OLD BASELINE CSV ------------
    df = pd.read_csv(BASELINE_CSV)

    emb_matrix = load_numeric_embeddings(df)
    labels = df["label"].values

    # ------------ APPEND DRIFT EMBEDDINGS ------------
    emb_matrix = np.vstack([emb_matrix, drift_embs])
    labels = np.concatenate([labels, [person] * len(drift_embs)])

    # ------------ SAVE NEW BASELINE CSV ------------
    updated_df = pd.DataFrame(emb_matrix)
    updated_df["label"] = labels
    updated_df.to_csv(BASELINE_CSV, index=False)
    print(f"[SAVE] Updated baseline CSV saved at {BASELINE_CSV}")

    # ------------ RETRAIN KNN + LABEL ENCODER ------------
    le = LabelEncoder()
    y_encoded = le.fit_transform(labels)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(emb_matrix, y_encoded)

    joblib.dump(knn, KNN_PATH)
    joblib.dump(le, LE_PATH)
    print(f"[SAVE] Updated KNN model and LabelEncoder saved.")

    # ------------ RECOMPUTE CENTROIDS ------------
    centroids = {}
    for p_name in updated_df["label"].unique():
        embs_p = updated_df[updated_df["label"] == p_name].iloc[:, :-1].values
        centroids[p_name] = np.mean(embs_p, axis=0)

    joblib.dump(centroids, CENTROIDS_PATH)
    print(f"[SAVE] Updated centroids saved at {CENTROIDS_PATH}")

    print(f"[DONE] Phase 3 retraining finished for '{person}'")


if __name__ == "__main__":
    main()
