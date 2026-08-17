"""Recommended trainer: pretrained Sentence-BERT features + our supervised regressor.

Run this on the project machine (where sentence-transformers is installed):
    python models/train_semantic_hybrid_grader.py

The Sentence-BERT encoder is pretrained; the score regressor is trained by this
project on labelled answer scores. This is the recommended final-project model.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, mean_squared_error, r2_score

from train_hybrid_grader import split_by_question

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "training_dataset.csv"
ARTIFACT_DIR = ROOT / "models" / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "semantic_hybrid_grader.joblib"
METRICS_PATH = ARTIFACT_DIR / "semantic_hybrid_metrics.json"
PREDICTIONS_PATH = ROOT / "data" / "semantic_hybrid_test_predictions.csv"
ENCODER_NAME = "all-MiniLM-L6-v2"

FEATURE_NAMES = [
    "reference_semantic_similarity",
    "question_semantic_similarity",
    "rubric_semantic_similarity",
    "keyword_coverage",
    "reference_token_recall",
    "student_token_precision",
    "length_ratio",
]


def normalize_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def lexical_keyword_coverage(answer: str, rubric: str) -> float:
    tokens = set(normalize_tokens(answer))
    keywords = [k.strip() for k in str(rubric or "").split(";") if k.strip()]
    if not keywords:
        return 0.0
    hits = 0
    for kw in keywords:
        kt = normalize_tokens(kw)
        if kt and all(t in tokens for t in kt):
            hits += 1
    return hits / len(keywords)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def features_for_row(row: pd.Series, encoder: SentenceTransformer) -> list[float]:
    q = str(row["question_text"])
    ref = str(row["model_answer"])
    ans = str(row["student_answer"])
    rubric = str(row.get("rubric_keywords", ""))
    keywords = [k.strip() for k in rubric.split(";") if k.strip()]

    texts = [ans, ref, q] + keywords
    embs = encoder.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    ans_e, ref_e, q_e = embs[0], embs[1], embs[2]
    kw_embs = embs[3:] if len(embs) > 3 else []

    ref_sim = max(0.0, cosine(ans_e, ref_e))
    q_sim = max(0.0, cosine(ans_e, q_e))
    rubric_sem = float(np.mean([max(0.0, cosine(ans_e, e)) for e in kw_embs])) if len(kw_embs) else 0.0

    ref_tokens = set(normalize_tokens(ref))
    ans_tokens = set(normalize_tokens(ans))
    overlap = ref_tokens & ans_tokens
    ref_recall = len(overlap) / max(1, len(ref_tokens))
    student_precision = len(overlap) / max(1, len(ans_tokens))
    length_ratio = min(len(ans_tokens) / max(1, len(ref_tokens)), 2.0) / 2.0

    return [
        ref_sim,
        q_sim,
        rubric_sem,
        lexical_keyword_coverage(ans, rubric),
        ref_recall,
        student_precision,
        length_ratio,
    ]


def metric_dict(y_true, y_pred):
    y_pred = np.clip(np.asarray(y_pred, float), 0, 10)
    y_true = np.asarray(y_true, float)
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(mean_squared_error(y_true, y_pred) ** 0.5), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "qwk": round(float(cohen_kappa_score(np.rint(y_true).astype(int), np.rint(y_pred).astype(int), weights="quadratic")), 4),
        "pearson": round(float(np.corrcoef(y_true, y_pred)[0, 1]), 4),
    }


def main():
    df = split_by_question(pd.read_csv(DATA_PATH).dropna(subset=["human_score"]).copy())
    encoder = SentenceTransformer(ENCODER_NAME)

    print("Encoding 100 labelled answers with Sentence-BERT...")
    X_all = np.asarray([features_for_row(r, encoder) for _, r in df.iterrows()], dtype=float)
    y_all = df["human_score"].to_numpy(float)

    train_mask = df["split"].eq("train").to_numpy()
    val_mask = df["split"].eq("validation").to_numpy()
    test_mask = df["split"].eq("test").to_numpy()

    candidates = {
        "Ridge": Ridge(alpha=0.7),
        "RandomForest": RandomForestRegressor(n_estimators=400, max_depth=6, min_samples_leaf=2, random_state=42),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=180, learning_rate=0.035, max_depth=2, min_samples_leaf=2, loss="huber", random_state=42),
    }

    val_results = {}
    for name, model in candidates.items():
        model.fit(X_all[train_mask], y_all[train_mask])
        val_results[name] = metric_dict(y_all[val_mask], model.predict(X_all[val_mask]))

    best_name = min(val_results, key=lambda n: val_results[n]["mae"])
    best_model = candidates[best_name]
    train_val = train_mask | val_mask
    best_model.fit(X_all[train_val], y_all[train_val])
    test_pred = np.clip(best_model.predict(X_all[test_mask]), 0, 10)
    test_metrics = metric_dict(y_all[test_mask], test_pred)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": best_model,
        "model_name": best_name,
        "encoder_name": ENCODER_NAME,
        "feature_names": FEATURE_NAMES,
        "score_min": 0.0,
        "score_max": 10.0,
    }, MODEL_PATH)

    summary = {
        "dataset_rows": int(len(df)),
        "questions": int(df.question_id.nunique()),
        "split": {k: int(df.split.eq(k).sum()) for k in ["train", "validation", "test"]},
        "encoder": ENCODER_NAME,
        "validation_models": val_results,
        "selected_model": best_name,
        "held_out_test": test_metrics,
        "note": "Prototype dataset contains 20 group-graded answers and 80 synthetic augmented answers. Replace synthetic examples with additional instructor/human-scored answers for stronger research claims.",
    }
    METRICS_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    pred = df.loc[test_mask, ["question_id", "student_answer", "human_score", "source"]].copy()
    pred["predicted_score"] = np.round(test_pred, 2)
    pred["absolute_error"] = np.round(np.abs(test_pred - y_all[test_mask]), 2)
    pred.to_csv(PREDICTIONS_PATH, index=False)

    print(json.dumps(summary, indent=2))
    print(f"Saved: {MODEL_PATH}")


if __name__ == "__main__":
    main()
