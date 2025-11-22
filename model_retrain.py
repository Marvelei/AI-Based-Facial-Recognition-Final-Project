import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier

from logging_utils import log_event, get_current_model_version, bump_model_version


def retrain_knn_model(
    embeddings_csv_path: str,
    model_save_path: str,
    psi_trigger: float | None = None,
):
    """
    Retrain the k-NN model using the latest embeddings.
    Logs retraining events into logs/full_system_log.csv.
    """

    # 1. Load latest embeddings
    df = pd.read_csv(embeddings_csv_path)

    # Assume last column is the label (subject name), others are embedding values
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # 2. Try to load old model to compute accuracy before retraining
    try:
        old_model = joblib.load(model_save_path)
        acc_before = old_model.score(X, y)
    except FileNotFoundError:
        old_model = None
        acc_before = None

    current_version = get_current_model_version()

    # 3. Log that retraining started
    log_event(
        phase="phase3",
        event="RETRAIN_START",
        subject=None,
        psi=psi_trigger,
        accuracy=acc_before,
        model_version=current_version,
        extra_info="Retraining k-NN started"
    )

    # 4. Train a new k-NN model
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)
    acc_after = model.score(X, y)

    # 5. Save the new model (overwrite the previous one)
    joblib.dump(model, model_save_path)

    # 6. Bump model version (v1 -> v2 -> ...)
    new_version = bump_model_version()

    # 7. Log that retraining finished
    log_event(
        phase="phase3",
        event="RETRAIN_FINISH",
        subject=None,
        psi=psi_trigger,
        accuracy=acc_after,
        model_version=new_version,
        extra_info=f"Retraining finished. acc_before={acc_before}, acc_after={acc_after}"
    )

    print(f"[INFO] Retraining completed. Model version updated to: {new_version}")
    print(f"[INFO] Accuracy before: {acc_before}, Accuracy after: {acc_after}")


if __name__ == "__main__":
    # You can adjust these paths based on your project structure
    embeddings_path = "initial_assets/baseline_embeddings.csv"
    model_path = "initial_assets/initial_knn_model.pkl"

    # You can pass PSI from last drift if you want,
    # for now we leave it as None or fill manually.
    psi_trigger_value = None

    retrain_knn_model(
        embeddings_csv_path=embeddings_path,
        model_save_path=model_path,
        psi_trigger=psi_trigger_value
    )
