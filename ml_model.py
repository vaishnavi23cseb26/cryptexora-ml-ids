"""
ml_model.py
-----------
Everything about the machine-learning model that the web app needs:
  * file paths and the list of features
  * loading the saved KNN model (re-training it automatically if needed)
  * predict_traffic(): turns 4 numbers into a prediction + explanation
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn

BASE_DIR = Path(__file__).parent
DATASET_PATH = BASE_DIR / "data" / "network_traffic.csv"
MODEL_PATH = BASE_DIR / "model" / "knn_model.joblib"
METRICS_PATH = BASE_DIR / "model" / "metrics.json"

# The model only ever sees these 4 numbers. The IP address is NOT a feature.
FEATURES = ["duration", "packet_count", "bytes_transferred", "failed_connections"]

# Friendly names + valid ranges (used for form validation and explanations)
FEATURE_INFO = {
    "duration": {"label": "Connection duration", "unit": "seconds", "max": 86400},
    "packet_count": {"label": "Packet count", "unit": "packets", "max": 10_000_000},
    "bytes_transferred": {"label": "Bytes transferred", "unit": "bytes", "max": 10_000_000_000},
    "failed_connections": {"label": "Failed connections", "unit": "attempts", "max": 100_000},
}


def load_model():
    """Return (model, metrics). Trains a fresh model if none is saved yet,
    or if it was saved with a different scikit-learn version."""
    needs_training = not (MODEL_PATH.exists() and METRICS_PATH.exists())
    if not needs_training:
        metrics = json.loads(METRICS_PATH.read_text())
        needs_training = metrics.get("sklearn_version") != sklearn.__version__
    if needs_training:
        print("Training the KNN model (first run or scikit-learn version changed)...")
        import train_model  # imported here to avoid a circular import
        train_model.train_and_save()
    return joblib.load(MODEL_PATH), json.loads(METRICS_PATH.read_text())


def predict_traffic(model, metrics, values):
    """
    values: dict with the 4 feature numbers.
    Returns a dict: prediction, confidence, severity, summary, reasons, ...
    """
    row = pd.DataFrame([[values[name] for name in FEATURES]], columns=FEATURES)

    # predict_proba -> [chance of normal, chance of suspicious].
    # For KNN this is simply the share of the k nearest neighbours in each class.
    chance_suspicious = float(model.predict_proba(row)[0][1])
    is_suspicious = chance_suspicious >= 0.5
    k = int(metrics["best_k"])
    suspicious_neighbours = round(chance_suspicious * k)

    if is_suspicious:
        confidence = chance_suspicious
        severity = "high" if chance_suspicious >= 0.8 else "medium"
    else:
        confidence = 1 - chance_suspicious
        severity = "low"

    # Explain in plain words: which numbers are far above normal traffic?
    reasons = []
    unusual_features = []
    for name in FEATURES:
        typical_max = metrics["normal_profile"][name]["typical_max"]
        if values[name] > typical_max:
            info = FEATURE_INFO[name]
            unusual_features.append(info["label"].lower())
            reasons.append(
                f"{info['label']} ({values[name]:g} {info['unit']}) is higher than "
                f"almost all normal traffic in the training data (about {typical_max:g} or less)."
            )
    if is_suspicious and not reasons:
        reasons.append("No single value is extreme, but the combination of values "
                       "is closest to known suspicious traffic.")
    if is_suspicious:
        summary = (f"{suspicious_neighbours} of the {k} most similar known records "
                   f"were suspicious.")
    else:
        summary = "Traffic appears normal - it looks like known normal traffic."
        if suspicious_neighbours:
            summary += (f" ({suspicious_neighbours} of the {k} most similar records "
                        f"were suspicious, but most were normal.)")

    return {
        "prediction": "suspicious" if is_suspicious else "normal",
        "confidence": round(confidence * 100, 1),
        "severity": severity,
        "summary": summary,
        "reasons": reasons,
        "unusual_features": unusual_features,
        "neighbours_suspicious": suspicious_neighbours,
        "k": k,
    }
