"""Research Grader V2A for the Smart Exam Evaluation project.

Goals
-----
1. Make grading rubric/concept aware instead of relying mainly on whole-answer
   semantic similarity and answer length.
2. Reduce the score boost caused by irrelevant verbosity, copied question text,
   and rubric-keyword stuffing.
3. Stay practical for the current low-data setting (140 labelled rows) by using
   frozen sentence embeddings + a shallow GradientBoosting regressor.

This script does NOT replace the restored baseline. It trains and saves a
separate experimental artifact:
    models/artifacts/research_grader_v2.joblib

It also writes:
    data/research_grader_v2_holdout.csv
    data/research_grader_v2_adversarial.csv
    data/research_grader_v2_feature_importance.csv
"""

from __future__ import annotations

import ast
import math
import os
import re
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "training_dataset.csv"
ARTIFACT_DIR = ROOT / "models" / "artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "research_grader_v2.joblib"
HOLDOUT_PATH = ROOT / "data" / "research_grader_v2_holdout.csv"
ADVERSARIAL_PATH = ROOT / "data" / "research_grader_v2_adversarial.csv"
IMPORTANCE_PATH = ROOT / "data" / "research_grader_v2_feature_importance.csv"

# Keep the encoder modest because the point of V2A is the rubric-aware
# representation, not a larger backbone. Override from the shell if desired:
#   GRADER_EMBED_MODEL=sentence-transformers/all-mpnet-base-v2 python ...
ENCODER_NAME = os.getenv(
    "GRADER_EMBED_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

RANDOM_STATE = 42
TEST_SIZE = 0.25

# Fixed thresholds for V2A. We will inspect their behaviour rather than tune
# them on the holdout set.
CONCEPT_SUPPORT = 0.45
CONCEPT_STRONG = 0.60
RUBRIC_SUPPORT = 0.50
IRRELEVANT_SENTENCE = 0.25

FEATURE_NAMES = [
    "answer_model_similarity",
    "answer_question_similarity",
    "model_minus_question_similarity",
    "length_adequacy_70",
    "length_excess_ratio",
    "word_count_log",
    "unique_token_ratio",
    "question_copy_ratio",
    "rubric_lexical_coverage",
    "rubric_semantic_mean",
    "rubric_semantic_min",
    "rubric_coverage_050",
    "concept_similarity_mean",
    "concept_similarity_min",
    "concept_similarity_top3",
    "concept_coverage_045",
    "concept_coverage_060",
    "concept_soft_coverage",
    "sentence_relevance_mean",
    "irrelevant_sentence_ratio",
    "evidence_density",
    "keyword_repetition_rate",
]

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "to",
    "of", "in", "on", "at", "by", "for", "from", "with", "as", "is",
    "are", "was", "were", "be", "been", "being", "it", "its", "this",
    "that", "these", "those", "can", "could", "may", "might", "will",
    "would", "should", "do", "does", "did", "have", "has", "had", "not",
    "what", "why", "how", "when", "where", "which", "who", "into", "about",
    "through", "their", "there", "they", "them", "we", "you", "your",
}

GENERIC_FLUFF = (
    "This is an important topic in modern technology and it has many uses in "
    "different situations. It can be useful for organizations and users in "
    "many real world applications. The topic has several advantages and is "
    "widely discussed in computer science."
)


class EmbeddingCache:
    """Small in-memory cache so repeated question/reference text is encoded once."""

    def __init__(self, model_name: str) -> None:
        print(f"Loading sentence encoder: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.cache: dict[str, np.ndarray] = {}

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        items = [clean_text(t) for t in texts]
        if not items:
            return np.zeros((0, 384), dtype=np.float32)

        missing = list(dict.fromkeys(t for t in items if t not in self.cache))
        if missing:
            vectors = self.model.encode(
                missing,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for text, vector in zip(missing, vectors):
                self.cache[text] = vector.astype(np.float32)

        return np.vstack([self.cache[t] for t in items])


def clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def word_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?", clean_text(text).lower())


def content_tokens(text: str) -> list[str]:
    return [
        token
        for token in word_tokens(text)
        if len(token) >= 3 and token not in STOPWORDS
    ]


def split_sentences(text: str) -> list[str]:
    """Split an answer/reference into evidence-sized units without NLTK/spaCy."""
    text = clean_text(text)
    if not text:
        return [""]

    # New lines and strong punctuation are the safest concept boundaries.
    parts = re.split(r"(?<=[.!?])\s+|\n+|\s*;\s*", text)
    parts = [clean_text(p) for p in parts if clean_text(p)]

    # If a reference is one very long sentence, split cautiously at colon and
    # a few coordinating boundaries. This keeps the method usable for rubrics
    # written as compact prose.
    if len(parts) == 1 and len(word_tokens(parts[0])) >= 28:
        parts = re.split(
            r"\s*:\s*|\s*,\s+(?=(?:and|while|whereas|which|that)\b)",
            parts[0],
            flags=re.IGNORECASE,
        )
        parts = [clean_text(p) for p in parts if len(word_tokens(p)) >= 3]

    return parts or [text]


def parse_rubric_terms(value: object) -> list[str]:
    """Accept comma/semicolon/pipe text or a stored Python/JSON-like list."""
    raw = clean_text(value)
    if not raw:
        return []

    if raw.startswith("[") and raw.endswith("]"):
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, (list, tuple)):
                return list(dict.fromkeys(clean_text(x) for x in parsed if clean_text(x)))
        except (ValueError, SyntaxError):
            pass

    terms = re.split(r"\s*[|;\n]+\s*|\s*,\s*", raw)
    terms = [clean_text(t).strip("-•* ") for t in terms]
    terms = [t for t in terms if t]
    return list(dict.fromkeys(terms))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    # Embeddings are normalized, so dot product is cosine similarity.
    return float(np.dot(a, b))


def safe_stats(values: np.ndarray) -> tuple[float, float, float]:
    if values.size == 0:
        return 0.0, 0.0, 0.0
    return float(values.mean()), float(values.min()), float(values.max())


def top_k_mean(values: np.ndarray, k: int = 3) -> float:
    if values.size == 0:
        return 0.0
    k = min(k, values.size)
    return float(np.sort(values)[-k:].mean())


def phrase_occurrences(text: str, phrase: str) -> int:
    text = clean_text(text).lower()
    phrase = clean_text(phrase).lower()
    if not phrase:
        return 0

    # Word boundaries are useful for simple terms. For punctuation-heavy
    # phrases, escaped substring matching is safer.
    if re.fullmatch(r"[a-z0-9 ]+", phrase):
        pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"
        return len(re.findall(pattern, text))
    return text.count(phrase)


def make_features(
    question_text: str,
    model_answer: str,
    student_answer: str,
    rubric_keywords: object,
    embedder: EmbeddingCache,
) -> dict[str, float]:
    question_text = clean_text(question_text)
    model_answer = clean_text(model_answer)
    student_answer = clean_text(student_answer)

    student_sentences = split_sentences(student_answer)
    concept_units = split_sentences(model_answer)
    rubric_terms = parse_rubric_terms(rubric_keywords)

    whole = embedder.encode([question_text, model_answer, student_answer])
    q_vec, model_vec, answer_vec = whole[0], whole[1], whole[2]

    answer_model_similarity = cosine(answer_vec, model_vec)
    answer_question_similarity = cosine(answer_vec, q_vec)

    student_sentence_vecs = embedder.encode(student_sentences)
    concept_vecs = embedder.encode(concept_units)

    if concept_vecs.size and student_sentence_vecs.size:
        concept_sentence_matrix = concept_vecs @ student_sentence_vecs.T
        concept_best = concept_sentence_matrix.max(axis=1)
        sentence_best = concept_sentence_matrix.max(axis=0)
    else:
        concept_best = np.zeros(1, dtype=np.float32)
        sentence_best = np.zeros(1, dtype=np.float32)

    concept_mean, concept_min, _ = safe_stats(concept_best)

    concept_coverage_045 = float(np.mean(concept_best >= CONCEPT_SUPPORT))
    concept_coverage_060 = float(np.mean(concept_best >= CONCEPT_STRONG))

    # Continuous concept support is less brittle than a single threshold.
    concept_soft = np.clip((concept_best - 0.25) / (0.70 - 0.25), 0.0, 1.0)
    concept_soft_coverage = float(concept_soft.mean())

    sentence_relevance_mean = float(sentence_best.mean())
    irrelevant_sentence_ratio = float(
        np.mean(sentence_best < IRRELEVANT_SENTENCE)
    )

    # Rubric descriptors receive their own semantic alignment features.
    if rubric_terms:
        rubric_vecs = embedder.encode(rubric_terms)
        rubric_matrix = rubric_vecs @ student_sentence_vecs.T
        rubric_best = rubric_matrix.max(axis=1)
        rubric_mean, rubric_min, _ = safe_stats(rubric_best)
        rubric_coverage_050 = float(np.mean(rubric_best >= RUBRIC_SUPPORT))
        matched_lexical = sum(
            1 for term in rubric_terms if phrase_occurrences(student_answer, term) > 0
        )
        rubric_lexical_coverage = matched_lexical / len(rubric_terms)

        total_term_hits = sum(
            phrase_occurrences(student_answer, term) for term in rubric_terms
        )
        extra_term_hits = max(0, total_term_hits - matched_lexical)
    else:
        rubric_mean = 0.0
        rubric_min = 0.0
        rubric_coverage_050 = 0.0
        rubric_lexical_coverage = 0.0
        extra_term_hits = 0

    answer_words = word_tokens(student_answer)
    model_words = word_tokens(model_answer)
    answer_word_count = len(answer_words)
    model_word_count = max(1, len(model_words))

    # Same named length feature as the restored baseline: full credit for
    # reaching 70% of reference length, but no bonus for continuing to grow.
    length_adequacy_70 = min(
        1.0,
        answer_word_count / max(1.0, 0.70 * model_word_count),
    )

    length_ratio = answer_word_count / model_word_count
    length_excess_ratio = max(0.0, length_ratio - 1.0)

    unique_token_ratio = (
        len(set(answer_words)) / answer_word_count if answer_word_count else 0.0
    )

    q_content = set(content_tokens(question_text))
    a_content = content_tokens(student_answer)
    question_copy_ratio = (
        sum(token in q_content for token in a_content) / len(a_content)
        if a_content
        else 0.0
    )

    # Coverage divided by a mild verbosity penalty: adding unrelated text can
    # make the answer longer, but it cannot create new rubric evidence.
    evidence_density = concept_soft_coverage / (1.0 + 0.35 * length_excess_ratio)

    keyword_repetition_rate = extra_term_hits / max(1, answer_word_count)

    result = {
        "answer_model_similarity": answer_model_similarity,
        "answer_question_similarity": answer_question_similarity,
        "model_minus_question_similarity": (
            answer_model_similarity - answer_question_similarity
        ),
        "length_adequacy_70": length_adequacy_70,
        "length_excess_ratio": length_excess_ratio,
        "word_count_log": math.log1p(answer_word_count),
        "unique_token_ratio": unique_token_ratio,
        "question_copy_ratio": question_copy_ratio,
        "rubric_lexical_coverage": float(rubric_lexical_coverage),
        "rubric_semantic_mean": rubric_mean,
        "rubric_semantic_min": rubric_min,
        "rubric_coverage_050": rubric_coverage_050,
        "concept_similarity_mean": concept_mean,
        "concept_similarity_min": concept_min,
        "concept_similarity_top3": top_k_mean(concept_best, 3),
        "concept_coverage_045": concept_coverage_045,
        "concept_coverage_060": concept_coverage_060,
        "concept_soft_coverage": concept_soft_coverage,
        "sentence_relevance_mean": sentence_relevance_mean,
        "irrelevant_sentence_ratio": irrelevant_sentence_ratio,
        "evidence_density": evidence_density,
        "keyword_repetition_rate": keyword_repetition_rate,
    }

    return result


def build_feature_frame(
    df: pd.DataFrame,
    embedder: EmbeddingCache,
    show_progress: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    total = len(df)

    for i, row in enumerate(df.itertuples(index=False), start=1):
        rows.append(
            make_features(
                question_text=getattr(row, "question_text"),
                model_answer=getattr(row, "model_answer"),
                student_answer=getattr(row, "student_answer"),
                rubric_keywords=getattr(row, "rubric_keywords"),
                embedder=embedder,
            )
        )
        if show_progress and (i == 1 or i % 20 == 0 or i == total):
            print(f"Feature extraction: {i}/{total}")

    out = pd.DataFrame(rows, columns=FEATURE_NAMES)
    out = out.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return out


def make_regressor() -> GradientBoostingRegressor:
    # Shallow trees + Huber loss are deliberately conservative for 140 rows.
    return GradientBoostingRegressor(
        loss="huber",
        n_estimators=140,
        learning_rate=0.035,
        max_depth=2,
        min_samples_leaf=4,
        subsample=0.90,
        random_state=RANDOM_STATE,
    )


def sample_weights(df: pd.DataFrame) -> np.ndarray:
    """Give the small set of genuine human-labelled rows more influence."""
    if "source" not in df.columns:
        return np.ones(len(df), dtype=float)

    return np.where(
        df["source"].fillna("").astype(str).eq("human_group"),
        3.0,
        1.0,
    ).astype(float)


def clipped_predict(model: GradientBoostingRegressor, X: pd.DataFrame) -> np.ndarray:
    return np.clip(model.predict(X), 0.0, 10.0)


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "within_1_mark": float(np.mean(np.abs(y_true - y_pred) <= 1.0)),
        "within_2_marks": float(np.mean(np.abs(y_true - y_pred) <= 2.0)),
    }


def build_stratify_labels(df: pd.DataFrame) -> pd.Series | None:
    score_bin = pd.cut(
        df["human_score"].astype(float),
        bins=[-0.01, 2.5, 5.5, 7.5, 10.01],
        labels=["low", "mid", "good", "high"],
        include_lowest=True,
    ).astype(str)

    if "source" in df.columns:
        composite = df["source"].fillna("unknown").astype(str) + "_" + score_bin
        if composite.value_counts().min() >= 2:
            return composite

        source = df["source"].fillna("unknown").astype(str)
        if source.value_counts().min() >= 2:
            return source

    if score_bin.value_counts().min() >= 2:
        return score_bin

    return None


def choose_irrelevant_reference(row: pd.Series, full_df: pd.DataFrame) -> str:
    other = full_df[full_df["question_id"].astype(str) != str(row["question_id"])]
    if other.empty:
        return GENERIC_FLUFF
    return clean_text(other.iloc[0]["model_answer"])


def adversarial_variants(row: pd.Series, full_df: pd.DataFrame) -> dict[str, str]:
    base = clean_text(row["student_answer"])
    question = clean_text(row["question_text"])
    rubric_terms = parse_rubric_terms(row["rubric_keywords"])
    stuffing = " ".join(rubric_terms * 4) if rubric_terms else "important relevant correct " * 8
    irrelevant = choose_irrelevant_reference(row, full_df)

    return {
        "generic_verbose_padding": f"{base} {GENERIC_FLUFF} {GENERIC_FLUFF}",
        "question_copy": f"{base} {question} {question} {question}",
        "rubric_keyword_stuffing": f"{base} {stuffing}",
        "irrelevant_topic_padding": f"{base} {irrelevant} {irrelevant}",
        "combined_long_weak": (
            f"{base} {question} {stuffing} {GENERIC_FLUFF} {irrelevant}"
        ),
    }


def run_adversarial_evaluation(
    model: GradientBoostingRegressor,
    test_df: pd.DataFrame,
    full_df: pd.DataFrame,
    embedder: EmbeddingCache,
) -> pd.DataFrame:
    # Focus on answers that should not receive a large score merely because
    # they were padded. Prefer <=4; fall back to the lowest-score test rows.
    weak = test_df[test_df["human_score"].astype(float) <= 4.0].copy()
    if weak.empty:
        weak = test_df.nsmallest(min(8, len(test_df)), "human_score").copy()

    records: list[dict[str, object]] = []

    for _, row in weak.iterrows():
        base_features = pd.DataFrame(
            [
                make_features(
                    row["question_text"],
                    row["model_answer"],
                    row["student_answer"],
                    row["rubric_keywords"],
                    embedder,
                )
            ],
            columns=FEATURE_NAMES,
        )
        base_score = float(clipped_predict(model, base_features)[0])

        for attack, mutated_answer in adversarial_variants(row, full_df).items():
            attack_features = pd.DataFrame(
                [
                    make_features(
                        row["question_text"],
                        row["model_answer"],
                        mutated_answer,
                        row["rubric_keywords"],
                        embedder,
                    )
                ],
                columns=FEATURE_NAMES,
            )
            attacked_score = float(clipped_predict(model, attack_features)[0])

            records.append(
                {
                    "question_id": row["question_id"],
                    "human_score": float(row["human_score"]),
                    "attack": attack,
                    "base_prediction": base_score,
                    "attacked_prediction": attacked_score,
                    "score_delta": attacked_score - base_score,
                    "base_answer": clean_text(row["student_answer"]),
                    "attacked_answer": mutated_answer,
                }
            )

    return pd.DataFrame(records)


def print_metrics(title: str, metrics: dict[str, float]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(f"MAE:            {metrics['mae']:.3f}")
    print(f"RMSE:           {metrics['rmse']:.3f}")
    print(f"R2:             {metrics['r2']:.3f}")
    print(f"Within 1 mark:  {metrics['within_1_mark']:.1%}")
    print(f"Within 2 marks: {metrics['within_2_marks']:.1%}")


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DATA_PATH}. The restored training dataset must exist first."
        )

    df = pd.read_csv(DATA_PATH)

    required = {
        "question_id",
        "question_text",
        "model_answer",
        "student_answer",
        "human_score",
        "rubric_keywords",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"training_dataset.csv is missing columns: {sorted(missing)}")

    df = df.dropna(subset=["student_answer", "human_score"]).copy()
    df["human_score"] = pd.to_numeric(df["human_score"], errors="raise").astype(float)
    df = df[df["human_score"].between(0.0, 10.0)].reset_index(drop=True)

    print("=" * 72)
    print("RESEARCH GRADER V2A")
    print("=" * 72)
    print(f"Dataset rows: {len(df)}")
    print(f"Questions:    {df['question_id'].nunique()}")
    print(f"Encoder:      {ENCODER_NAME}")
    print("Model:        GradientBoostingRegressor (Huber, shallow trees)")
    print("Length feature: length_adequacy_70")

    if "source" in df.columns:
        print("\nSources:")
        print(df["source"].fillna("unknown").value_counts().to_string())

    embedder = EmbeddingCache(ENCODER_NAME)
    X = build_feature_frame(df, embedder)
    y = df["human_score"].to_numpy(dtype=float)

    stratify = build_stratify_labels(df)
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y[train_idx]
    y_test = y[test_idx]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    model = make_regressor()
    model.fit(X_train, y_train, sample_weight=sample_weights(train_df))

    test_pred = clipped_predict(model, X_test)
    metrics = metric_dict(y_test, test_pred)
    print_metrics("Holdout performance", metrics)

    low_mask = y_test <= 4.0
    if low_mask.any():
        low_metrics = metric_dict(y_test[low_mask], test_pred[low_mask])
        print_metrics("Holdout performance: weak answers (<=4)", low_metrics)
    else:
        low_metrics = {}

    human_metrics: dict[str, float] = {}
    if "source" in test_df.columns:
        human_mask = test_df["source"].fillna("").eq("human_group").to_numpy()
        if human_mask.any():
            human_metrics = metric_dict(y_test[human_mask], test_pred[human_mask])
            print_metrics("Holdout performance: human_group only", human_metrics)

    holdout = test_df.copy()
    holdout["predicted_score"] = test_pred
    holdout["absolute_error"] = np.abs(
        holdout["human_score"].astype(float).to_numpy() - test_pred
    )
    holdout.to_csv(HOLDOUT_PATH, index=False)

    adversarial = run_adversarial_evaluation(
        model=model,
        test_df=test_df,
        full_df=df,
        embedder=embedder,
    )
    adversarial.to_csv(ADVERSARIAL_PATH, index=False)

    print("\nAdversarial robustness")
    print("----------------------")
    if adversarial.empty:
        print("No adversarial probes were generated.")
        adv_summary = {}
    else:
        positive = adversarial["score_delta"].clip(lower=0.0)
        adv_summary = {
            "mean_delta": float(adversarial["score_delta"].mean()),
            "mean_positive_delta": float(positive.mean()),
            "max_delta": float(adversarial["score_delta"].max()),
            "failure_rate_gt_0_5": float(np.mean(adversarial["score_delta"] > 0.5)),
            "failure_rate_gt_1_0": float(np.mean(adversarial["score_delta"] > 1.0)),
        }
        print(f"Mean score delta:          {adv_summary['mean_delta']:+.3f}")
        print(f"Mean positive delta:       {adv_summary['mean_positive_delta']:+.3f}")
        print(f"Worst score increase:      {adv_summary['max_delta']:+.3f}")
        print(f"Failures > +0.5 mark:      {adv_summary['failure_rate_gt_0_5']:.1%}")
        print(f"Failures > +1.0 mark:      {adv_summary['failure_rate_gt_1_0']:.1%}")

        by_attack = (
            adversarial.groupby("attack")["score_delta"]
            .agg(["mean", "max", "count"])
            .sort_values("max", ascending=False)
        )
        print("\nBy attack:")
        print(by_attack.round(3).to_string())

    importance = pd.DataFrame(
        {
            "feature": FEATURE_NAMES,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(IMPORTANCE_PATH, index=False)

    print("\nTop feature importances")
    print("-----------------------")
    print(importance.head(12).to_string(index=False, formatters={"importance": "{:.4f}".format}))

    # Train final V2A on all 140 rows after the untouched holdout evaluation.
    final_model = make_regressor()
    final_model.fit(X, y, sample_weight=sample_weights(df))

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "version": "research_grader_v2a",
        "model_name": "ResearchGraderV2A-GradientBoosting",
        "model": final_model,
        "encoder_name": ENCODER_NAME,
        "feature_names": FEATURE_NAMES,
        "length_feature": "length_adequacy_70",
        "thresholds": {
            "concept_support": CONCEPT_SUPPORT,
            "concept_strong": CONCEPT_STRONG,
            "rubric_support": RUBRIC_SUPPORT,
            "irrelevant_sentence": IRRELEVANT_SENTENCE,
        },
        "dataset_rows": int(len(df)),
        "holdout_metrics": metrics,
        "low_score_holdout_metrics": low_metrics,
        "human_holdout_metrics": human_metrics,
        "adversarial_summary": adv_summary,
    }
    joblib.dump(artifact, ARTIFACT_PATH)

    print("\nSaved")
    print("-----")
    print(f"Artifact:           {ARTIFACT_PATH}")
    print(f"Holdout predictions:{HOLDOUT_PATH}")
    print(f"Adversarial report: {ADVERSARIAL_PATH}")
    print(f"Feature importance: {IMPORTANCE_PATH}")
    print("\nV2A is experimental; the restored baseline has not been modified.")


if __name__ == "__main__":
    main()
