"""Inference helper for the project's custom-trained grading model.

Preference order:
1) semantic_hybrid_grader.joblib (recommended; trained with Sentence-BERT features)
2) hybrid_grader.joblib (included offline lexical fallback)
"""
from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "models" / "artifacts"
SEMANTIC_MODEL_PATH = ARTIFACT_DIR / "semantic_hybrid_grader.joblib"
FALLBACK_MODEL_PATH = ARTIFACT_DIR / "hybrid_grader.joblib"
_artifact = None
_encoder = None
_mode = None


def _load():
    global _artifact, _encoder, _mode
    if _artifact is not None:
        return _artifact, _mode

    if SEMANTIC_MODEL_PATH.exists():
        from sentence_transformers import SentenceTransformer
        _artifact = joblib.load(SEMANTIC_MODEL_PATH)
        _encoder = SentenceTransformer(_artifact.get("encoder_name", "all-MiniLM-L6-v2"))
        _mode = "semantic_hybrid"
    elif FALLBACK_MODEL_PATH.exists():
        _artifact = joblib.load(FALLBACK_MODEL_PATH)
        _mode = "lexical_fallback"
    else:
        raise FileNotFoundError(
            "No trained grader artifact found. Run `python models/train_semantic_hybrid_grader.py` "
            "(recommended) or `python models/train_hybrid_grader.py`."
        )
    return _artifact, _mode


def trained_score(question_text, model_answer, student_answer, rubric_keywords="", max_score=10):
    artifact, mode = _load()
    row = pd.Series({
        "question_text": question_text,
        "model_answer": model_answer,
        "student_answer": student_answer,
        "rubric_keywords": rubric_keywords or "",
    })

    if mode == "semantic_hybrid":
        from train_semantic_hybrid_grader import features_for_row
        feature_values = features_for_row(row, _encoder)
    else:
        from train_hybrid_grader import make_features
        feature_values = make_features(row, artifact["vectorizer"])

    features = np.asarray([feature_values], dtype=float)
    pred = float(artifact["model"].predict(features)[0])
    pred = max(0.0, min(float(max_score), pred))
    return {
        "score": round(pred, 2),
        "max_score": max_score,
        "model_name": f"{artifact.get('model_name', 'trained_regressor')} ({mode})",
        "features": {
            name: round(float(value), 4)
            for name, value in zip(artifact.get("feature_names", []), features[0])
        },
    }
