"""
train_model.py
--------------
Trains, evaluates and saves the K-Nearest Neighbors (KNN) model.
Run it with:  python train_model.py

How KNN works (simple version):
  To classify a new connection, find the k most SIMILAR connections in the
  training data (smallest distance) and let them vote. If most of them were
  "suspicious", the new connection is predicted "suspicious".
"""
import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from ml_model import DATASET_PATH, FEATURES, METRICS_PATH, MODEL_PATH


def build_pipeline(k):
    """Pre-processing + KNN chained together, so the same steps are always
    applied when training and when predicting."""
    return Pipeline([
        # 1. log1p squashes huge numbers (e.g. 600,000 bytes) so they don't
        #    drown out small ones (e.g. 3 failed connections).
        ("log", FunctionTransformer(np.log1p)),
        # 2. KNN measures distance, so every feature must be on the same scale.
        ("scale", StandardScaler()),
        # 3. The actual classifier.
        ("knn", KNeighborsClassifier(n_neighbors=k)),
    ])


def train_and_save():
    # Load data. Label: normal -> 0, suspicious -> 1
    data = pd.read_csv(DATASET_PATH)
    X = data[FEATURES]
    y = (data["label"] == "suspicious").astype(int)

    # Keep 20% of the data hidden for an honest test of the final model.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Choose k (number of neighbours) using 5-fold cross-validation on training data only.
    best_k, best_score = 5, 0
    for k in [3, 5, 7, 9, 11]:
        score = cross_val_score(build_pipeline(k), X_train, y_train, cv=5).mean()
        print(f"  k={k:<2}  cross-validation accuracy = {score:.3f}")
        if score >= best_score:  # on a tie, prefer the larger k (smoother voting)
            best_k, best_score = k, score

    # Train the final model and evaluate it on the unseen test set.
    model = build_pipeline(best_k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # What does "normal" look like? (95th percentile of each feature) -
    # used later to explain predictions in plain English.
    normal_rows = X[y == 0]
    normal_profile = {name: {"typical_max": round(float(normal_rows[name].quantile(0.95)), 1)}
                      for name in FEATURES}

    metrics = {
        "algorithm": "K-Nearest Neighbors (KNN)",
        "best_k": best_k,
        "cv_accuracy": round(best_score, 4),
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),  # [[TN, FP], [FN, TP]]
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "normal_rows": int((y == 0).sum()),
        "suspicious_rows": int((y == 1).sum()),
        "normal_profile": normal_profile,
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print(f"\nChosen k = {best_k}")
    print(f"Test accuracy : {metrics['accuracy']:.1%}")
    print(f"Precision     : {metrics['precision']:.1%}  (when it says suspicious, how often is it right?)")
    print(f"Recall        : {metrics['recall']:.1%}  (how many suspicious records did it catch?)")
    print(f"Confusion matrix [[TN, FP], [FN, TP]]: {metrics['confusion_matrix']}")
    print(f"Saved model   -> {MODEL_PATH}")
    return metrics


if __name__ == "__main__":
    train_and_save()
