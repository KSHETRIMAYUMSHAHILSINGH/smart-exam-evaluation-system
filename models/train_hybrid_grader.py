"""Train the project's primary custom grading model.

Architecture:
    question + reference answer + rubric + student answer
        -> interpretable NLP similarity/coverage features
        -> supervised regression model trained on human/synthetic scores

This is intentionally lightweight so it can be retrained on a normal laptop.
The Sentence-Transformer scorer remains as a separate pretrained semantic
baseline in the application; this file trains the project's own score predictor.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    cohen_kappa_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "training_dataset.csv"
ARTIFACT_DIR = ROOT / "models" / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "hybrid_grader.joblib"
METRICS_PATH = ARTIFACT_DIR / "hybrid_metrics.json"
PREDICTIONS_PATH = ROOT / "data" / "hybrid_test_predictions.csv"
FEATURE_NAMES = [
    "reference_tfidf_similarity",
    "question_tfidf_similarity",
    "keyword_coverage",
    "reference_token_recall",
    "student_token_precision",
    "length_ratio",
    "answer_length_norm",
]


def normalize_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def keyword_coverage(answer: str, rubric: str) -> float:
    answer_tokens = set(normalize_tokens(answer))
    keywords = [k.strip() for k in str(rubric or "").split(";") if k.strip()]
    if not keywords:
        return 0.0
    matched = 0
    for keyword in keywords:
        kt = normalize_tokens(keyword)
        if kt and all(t in answer_tokens for t in kt):
            matched += 1
    return matched / len(keywords)


def _cosine(vectorizer: TfidfVectorizer, a: str, b: str) -> float:
    mat = vectorizer.transform([str(a), str(b)])
    return float(cosine_similarity(mat[0], mat[1])[0, 0])


def make_features(row: pd.Series | dict, vectorizer: TfidfVectorizer) -> list[float]:
    q = str(row["question_text"])
    ref = str(row["model_answer"])
    ans = str(row["student_answer"])
    rubric = str(row.get("rubric_keywords", ""))

    ref_sim = _cosine(vectorizer, ans, ref)
    q_sim = _cosine(vectorizer, ans, q)
    kw_cov = keyword_coverage(ans, rubric)

    ref_tokens = set(normalize_tokens(ref))
    ans_tokens = set(normalize_tokens(ans))
    overlap = ref_tokens & ans_tokens
    ref_recall = len(overlap) / max(1, len(ref_tokens))
    student_precision = len(overlap) / max(1, len(ans_tokens))

    ref_len = max(1, len(normalize_tokens(ref)))
    ans_len = len(normalize_tokens(ans))
    length_ratio = min(ans_len / ref_len, 2.0) / 2.0  # normalize to 0..1
    answer_length_norm = min(ans_len / 100.0, 1.0)

    return [
        ref_sim,
        q_sim,
        kw_cov,
        ref_recall,
        student_precision,
        length_ratio,
        answer_length_norm,
    ]


def split_by_question(df: pd.DataFrame) -> pd.DataFrame:
    """Create a repeatable 70/15/15 answer-level split for every question.

    Each question contributes one real/group-graded answer to validation and one
    to test, so the reported hold-out result is not synthetic-only.
    """
    parts = []
    for _, group in df.groupby("question_id", sort=True):
        human = group[group["source"] == "human_group"].sort_values("human_score").copy()
        synth = group[group["source"] == "synthetic_augmented"].sort_values(["human_score", "student_answer"]).copy()
        if len(human) < 4 or len(synth) < 16:
            raise ValueError("Expected 4 human-group and 16 synthetic examples per question.")

        human["split"] = "train"
        synth["split"] = "train"
        # Human scores are 7,8,9,10: hold out 8 for test and 9 for validation.
        human.iloc[1, human.columns.get_loc("split")] = "test"
        human.iloc[2, human.columns.get_loc("split")] = "validation"
        # Add low/mid synthetic examples to each hold-out set.
        synth.iloc[[1, 8], synth.columns.get_loc("split")] = "test"
        synth.iloc[[3, 11], synth.columns.get_loc("split")] = "validation"
        parts.extend([human, synth])
    return pd.concat(parts, ignore_index=True)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    clipped = np.clip(y_pred, 0, 10)
    rounded_true = np.clip(np.rint(y_true), 0, 10).astype(int)
    rounded_pred = np.clip(np.rint(clipped), 0, 10).astype(int)
    return {
        "mae": round(float(mean_absolute_error(y_true, clipped)), 4),
        "rmse": round(float(mean_squared_error(y_true, clipped) ** 0.5), 4),
        "r2": round(float(r2_score(y_true, clipped)), 4),
        "qwk": round(float(cohen_kappa_score(rounded_true, rounded_pred, weights="quadratic")), 4),
        "pearson": round(float(np.corrcoef(y_true, clipped)[0, 1]), 4) if len(y_true) > 1 else 0.0,
    }


def main() -> None:
    df = pd.read_csv(DATA_PATH).dropna(subset=["human_score"]).copy()
    df = split_by_question(df)

    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "validation"].copy()
    test = df[df["split"] == "test"].copy()

    corpus = pd.concat(
        [train["question_text"], train["model_answer"], train["student_answer"]],
        ignore_index=True,
    ).astype(str)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
        max_features=5000,
    )
    vectorizer.fit(corpus)

    def X(frame: pd.DataFrame) -> np.ndarray:
        return np.asarray([make_features(r, vectorizer) for _, r in frame.iterrows()], dtype=float)

    X_train, y_train = X(train), train["human_score"].to_numpy(float)
    X_val, y_val = X(val), val["human_score"].to_numpy(float)
    X_test, y_test = X(test), test["human_score"].to_numpy(float)

    candidates = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=400,
            max_depth=6,
            min_samples_leaf=2,
            random_state=42,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=180,
            learning_rate=0.035,
            max_depth=2,
            min_samples_leaf=2,
            loss="huber",
            random_state=42,
        ),
    }

    validation_results = {}
    trained = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        validation_results[name] = metrics(y_val, pred)
        trained[name] = model

    best_name = min(validation_results, key=lambda name: validation_results[name]["mae"])

    # Refit the selected model on train + validation, then evaluate once on held-out test.
    train_val = pd.concat([train, val], ignore_index=True)
    X_train_val = X(train_val)
    y_train_val = train_val["human_score"].to_numpy(float)
    best_model = candidates[best_name]
    best_model.fit(X_train_val, y_train_val)
    test_pred = np.clip(best_model.predict(X_test), 0, 10)
    test_metrics = metrics(y_test, test_pred)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": best_model,
            "vectorizer": vectorizer,
            "feature_names": FEATURE_NAMES,
            "model_name": best_name,
            "score_min": 0.0,
            "score_max": 10.0,
        },
        MODEL_PATH,
    )

    summary = {
        "dataset_rows": int(len(df)),
        "human_group_rows": int((df["source"] == "human_group").sum()),
        "synthetic_augmented_rows": int((df["source"] == "synthetic_augmented").sum()),
        "questions": int(df["question_id"].nunique()),
        "split": {"train": int(len(train)), "validation": int(len(val)), "test": int(len(test))},
        "validation_models": validation_results,
        "selected_model": best_name,
        "held_out_test": test_metrics,
        "note": "Prototype results use 20 group-graded answers plus 80 synthetic augmented answers; synthetic labels are not a substitute for a larger real human-graded dataset.",
    }
    METRICS_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    pred_df = test[["question_id", "student_answer", "human_score", "source"]].copy()
    pred_df["predicted_score"] = np.round(test_pred, 2)
    pred_df["absolute_error"] = np.round(np.abs(test_pred - y_test), 2)
    pred_df.to_csv(PREDICTIONS_PATH, index=False)

    print(json.dumps(summary, indent=2))
    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved predictions: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
