"""RandomForest crop classifier (trained on 100% synthetic curves)."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from . import config
from .features import (
    CROP_IDS,
    CROP_LABELS,
    model_feature_names,
    model_feature_vector,
    pearson,
    two_wave_detected,
)

logger = logging.getLogger(__name__)
DATA_DIR = Path(__file__).resolve().parent / "data"
MODEL_PATH = DATA_DIR / "model.joblib"
PROFILES_PATH = DATA_DIR / "profiles.json"

_bundle_cache: dict | None = None


def load_bundle(force: bool = False) -> dict | None:
    global _bundle_cache
    if _bundle_cache is not None and not force:
        return _bundle_cache
    if not MODEL_PATH.exists():
        return None
    try:
        _bundle_cache = joblib.load(MODEL_PATH)
        return _bundle_cache
    except Exception as exc:  # corrupt model / sklearn version change
        logger.warning("Cannot load ML model: %s", exc)
        return None


def train_and_save(
    series: Sequence[Sequence[float]],
    labels: Sequence[str],
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    X = np.vstack([model_feature_vector(values) for values in series])
    y = np.asarray(labels)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    clf = RandomForestClassifier(
        n_estimators=320,
        max_depth=16,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    accuracy = float(accuracy_score(y_test, prediction))
    label_order = sorted(set(y.tolist()))
    matrix = confusion_matrix(y_test, prediction, labels=label_order).tolist()
    report = classification_report(y_test, prediction, zero_division=0, output_dict=True)
    bundle = {
        "model": clf,
        "classes": [str(c) for c in clf.classes_],
        "feature_names": model_feature_names(),
        "metrics": {
            "accuracy": round(accuracy, 4),
            "confusion_matrix": matrix,
            "labels": label_order,
            "classification_report": report,
            "n_samples": int(len(y)),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "dataset": "synthetic",
            "validation": "synthetic hold-out only; not validated on real fields",
        },
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    global _bundle_cache
    _bundle_cache = bundle
    return bundle


def _fallback_prediction(values: Sequence[float]) -> dict:
    """Rule-based backup if the pickle cannot be loaded.

    The demo must remain usable even if a joblib file is rebuilt by another
    scikit-learn version.  This fallback is intentionally simple.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size and float(np.nanmax(arr)) < 0.30:
        return _format_prediction(
            {"bare": 0.72, "wheat": 0.08, "barley": 0.07, "rapso": 0.07, "sunflower": 0.03, "double_wave": 0.03},
            source="fallback",
        )
    if two_wave_detected(arr):
        return _format_prediction(
            {"double_wave": 0.70, "wheat": 0.12, "barley": 0.08, "sunflower": 0.05, "rapso": 0.03, "bare": 0.02},
            source="fallback",
        )
    try:
        with PROFILES_PATH.open("r", encoding="utf-8") as fh:
            profiles = json.load(fh)
        corrs = {}
        for crop in CROP_IDS:
            ref = [float(x["ndvi"]) for x in profiles["crops"][crop]["series"]]
            corrs[crop] = max(0.001, pearson(arr, ref))
        exponent = {crop: float(np.exp(5.0 * value)) for crop, value in corrs.items()}
        total = sum(exponent.values()) + 1e-9
        probs = {crop: value / total for crop, value in exponent.items()}
        probs["bare"] = 0.03
        probs["double_wave"] = 0.02
        norm = sum(probs.values())
        probs = {crop: value / norm for crop, value in probs.items()}
        return _format_prediction(probs, source="fallback")
    except Exception:
        return _format_prediction(
            {"wheat": 0.3, "barley": 0.25, "sunflower": 0.2, "rapso": 0.15, "bare": 0.1},
            source="fallback",
        )


def _format_prediction(probabilities: dict[str, float], source: str = "model") -> dict:
    ordered = sorted(probabilities.items(), key=lambda pair: pair[1], reverse=True)
    top3 = [
        {
            "crop": crop,
            "label_ru": CROP_LABELS.get(crop, {}).get("ru", crop),
            "label_kz": CROP_LABELS.get(crop, {}).get("kz", crop),
            "probability": round(float(prob), 4),
        }
        for crop, prob in ordered[:3]
    ]
    return {
        "predicted": ordered[0][0],
        "probability": round(float(ordered[0][1]), 4),
        "top3": top3,
        "probabilities": {crop: round(float(prob), 4) for crop, prob in ordered},
        "source": source,
    }


def predict_crop(values: Sequence[float]) -> dict:
    bundle = load_bundle()
    if not bundle:
        return _fallback_prediction(values)
    try:
        vector = model_feature_vector(values).reshape(1, -1)
        clf = bundle["model"]
        probabilities = clf.predict_proba(vector)[0]
        classes = [str(c) for c in bundle["classes"]]
        prob_map = {crop: float(prob) for crop, prob in zip(classes, probabilities)}
        result = _format_prediction(prob_map)
        result["source"] = "model"
        return result
    except Exception as exc:
        logger.warning("ML prediction failed, using fallback: %s", exc)
        return _fallback_prediction(values)


def model_info() -> dict:
    bundle = load_bundle()
    if not bundle:
        return {"available": False}
    metrics = bundle.get("metrics", {})
    return {
        "available": True,
        "trained_at": bundle.get("trained_at"),
        "sklearn_version": bundle.get("sklearn_version"),
        "accuracy": metrics.get("accuracy"),
        "n_samples": metrics.get("n_samples"),
        "dataset": metrics.get("dataset"),
        "classes": bundle.get("classes", []),
    }
