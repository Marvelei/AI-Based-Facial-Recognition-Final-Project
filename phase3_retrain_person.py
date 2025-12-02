import argparse
import numpy as np
import pandas as pd
import os
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

from logs.logging_utils import log_event, bump_model_version
# ---------------------------------------------------------------------

BASELINE_CSV = "initial_assets/baseline_embeddings.csv"
KNN_PATH = "initial_assets/initial_knn_model.pkl"
LE_PATH = "initial_assets/label_encoder.pkl"
CENTROIDS_PATH = "initial_assets/centroids.pkl"
DRIFT_DIR = "drift_assets"


def load_numeric(df):
    return df.select_dtypes(include=[np.number]).values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--person", required=True)
    args = parser.parse_args()
    person = args.person

    log_event("phase3", "RETRAIN_STARTED", subject=person)

    drift_path = os.path.join(DRIFT_DIR, f"{person}_drift_embeddings.npy")
    if not os.path.exists(drift_path):
        log_event("phase3", "RETRAIN_FAILED", subject=person,
                  extra_info="no drift file")
        return

    drift_embs = np.load(drift_path)
    if drift_embs.ndim == 1:
        drift_embs = drift_embs.reshape(1, -1)

    df = pd.read_csv(BASELINE_CSV)
    X = load_numeric(df)
    y = df["label"].values

    X = np.vstack([X, drift_embs])
    y = np.concatenate([y, [person] * len(drift_embs)])

    updated = pd.DataFrame(X)
    updated["label"] = y
    updated.to_csv(BASELINE_CSV, index=False)

    # retrain knn
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y_enc)

    joblib.dump(knn, KNN_PATH)
    joblib.dump(le, LE_PATH)

    # recompute centroids
    centroids = {}
    for p in updated["label"].unique():
        embs = updated[updated["label"] == p].iloc[:, :-1].values
        centroids[p] = np.mean(embs, axis=0)

    joblib.dump(centroids, CENTROIDS_PATH)

    new_version = bump_model_version()

    log_event(
        "phase3",
        "RETRAIN_SUCCESS",
        subject=person,
        model_version=new_version,
        extra_info=f"{len(drift_embs)} drift samples"
    )


if __name__ == "__main__":
    main()
