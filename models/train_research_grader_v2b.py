"""Research Grader V2B for the Smart Exam Evaluation project.

Why V2B exists
--------------
V2A exposed a strong shortcut: raw answer length dominated feature importance and
adversarially padded weak answers gained several marks. V2B attacks that failure
mode directly.

V2B changes
-----------
1. Removes raw word-count and raw total-length features from the predictive set.
2. Replaces total-length adequacy with *evidence-only* length adequacy: only text
   that is semantically supported by the reference concepts contributes.
3. Adds explicit padding, question-copy, repetition, and keyword-stuffing signals.
4. Uses adversarial/hard-negative augmentation on the TRAINING SPLIT ONLY. Each
   padded version keeps the original score, teaching score invariance to gaming.
5. Trains a clean-only ablation and the robust model on the exact same split so
   we can measure what adversarial training actually buys us.
6. Keeps the restored baseline and V2A artifacts untouched.

Outputs
-------
models/artifacts/research_grader_v2b.joblib
data/research_grader_v2b_holdout.csv
data/research_grader_v2b_adversarial.csv
data/research_grader_v2b_feature_importance.csv
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
ARTIFACT_PATH = ARTIFACT_DIR / "research_grader_v2b.joblib"
HOLDOUT_PATH = ROOT / "data" / "research_grader_v2b_holdout.csv"
ADVERSARIAL_PATH = ROOT / "data" / "research_grader_v2b_adversarial.csv"
IMPORTANCE_PATH = ROOT / "data" / "research_grader_v2b_feature_importance.csv"

ENCODER_NAME = os.getenv(
    "GRADER_EMBED_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

RANDOM_STATE = 42
TEST_SIZE = 0.25

# Semantic thresholds. These are fixed before seeing the V2B holdout results.
CONCEPT_SUPPORT = 0.42
CONCEPT_STRONG = 0.58
RUBRIC_SUPPORT = 0.48
SENTENCE_SUPPORT_LOW = 0.24
SENTENCE_SUPPORT_HIGH = 0.58

# Each original training row gets five invariance attacks. A 0.25 multiplier
# means all attacks together have roughly the same total weight as 1.25 clean
# examples, rather than overwhelming the clean data.
ADVERSARIAL_WEIGHT_MULTIPLIER = 0.25

FEATURE_NAMES = [
    # Whole-answer semantic signals are retained, but no raw length feature is.
    "answer_model_similarity",
    "answer_question_similarity",
    "model_minus_question_similarity",

    # Reference-concept evidence.
    "concept_similarity_mean",
    "concept_similarity_min",
    "concept_similarity_top3",
    "concept_coverage_042",
    "concept_coverage_058",
    "concept_soft_coverage",

    # Rubric evidence.
    "rubric_semantic_mean",
    "rubric_semantic_min",
    "rubric_coverage_048",
    "rubric_lexical_coverage",

    # Evidence-vs-padding features.
    "sentence_relevance_mean",
    "supported_sentence_ratio",
    "irrelevant_sentence_ratio",
    "unsupported_word_ratio",
    "evidence_density",
    "evidence_length_adequacy_70",

    # Anti-gaming features.
    "question_copy_ratio",
    "unique_token_ratio",
    "repeated_bigram_ratio",
    "keyword_repetition_rate",
    "keyword_stuffing_score",
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
    """Small cache so repeated question/reference strings are encoded once."""

    def __init__(self, model_name: str) -> None:
        print(f"Loading sentence encoder: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.cache: dict[str, np.ndarray] = {}
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        items = [clean_text(t) for t in texts]
        if not items:
            return np.zeros((0, self.dim), dtype=np.float32)

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
    text = clean_text(text)
    if not text:
        return [""]

    parts = re.split(r"(?<=[.!?])\s+|\n+|\s*;\s*", text)
    parts = [clean_text(p) for p in parts if clean_text(p)]
    return parts or [text]


def split_reference_concepts(text: str) -> list[str]:
    """Create concept-sized reference units rather than one giant sentence."""
    text = clean_text(text)
    if not text:
        return [""]

    parts = re.split(
        r"\n+|\s*[;•]\s*|(?<=[.!?])\s+",
        text,
    )
    parts = [clean_text(p).strip("-•* ") for p in parts if clean_text(p)]

    expanded: list[str] = []
    for part in parts:
        # Only split long clauses. Short references stay intact.
        if len(word_tokens(part)) >= 24:
            clauses = re.split(
                r"\s*,\s+(?=(?:and|but|while|whereas|which|that)\b)",
                part,
                flags=re.IGNORECASE,
            )
            clauses = [clean_text(c) for c in clauses if len(word_tokens(c)) >= 4]
            expanded.extend(clauses or [part])
        else:
            expanded.append(part)

    return expanded or [text]


def parse_rubric_terms(value: object) -> list[str]:
    raw = clean_text(value)
    if not raw:
        return []

    if raw.startswith("[") and raw.endswith("]"):
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, (list, tuple)):
                return list(
                    dict.fromkeys(clean_text(x) for x in parsed if clean_text(x))
                )
        except (ValueError, SyntaxError):
            pass

    terms = re.split(r"\s*[|;\n]+\s*|\s*,\s*", raw)
    terms = [clean_text(t).strip("-•* ") for t in terms]
    terms = [t for t in terms if t]
    return list(dict.fromkeys(terms))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
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
    if re.fullmatch(r"[a-z0-9 ]+", phrase):
        pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"
        return len(re.findall(pattern, text))
    return text.count(phrase)


def repeated_bigram_ratio(tokens: list[str]) -> float:
    if len(tokens) < 2:
        return 0.0
    bigrams = list(zip(tokens[:-1], tokens[1:]))
    return 1.0 - (len(set(bigrams)) / len(bigrams))


def sentence_support_weights(sentence_best: np.ndarray) -> np.ndarray:
    """0..1 evidence weight for each student sentence."""
    denom = SENTENCE_SUPPORT_HIGH - SENTENCE_SUPPORT_LOW
    return np.clip((sentence_best - SENTENCE_SUPPORT_LOW) / denom, 0.0, 1.0)


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
    concept_units = split_reference_concepts(model_answer)
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
        sentence_best = np.zeros(len(student_sentences), dtype=np.float32)

    concept_mean, concept_min, _ = safe_stats(concept_best)
    concept_coverage_042 = float(np.mean(concept_best >= CONCEPT_SUPPORT))
    concept_coverage_058 = float(np.mean(concept_best >= CONCEPT_STRONG))

    concept_soft = np.clip((concept_best - 0.22) / (0.68 - 0.22), 0.0, 1.0)
    concept_soft_coverage = float(concept_soft.mean())

    support_weights = sentence_support_weights(sentence_best)
    sentence_relevance_mean = float(sentence_best.mean())
    supported_sentence_ratio = float(np.mean(support_weights >= 0.50))
    irrelevant_sentence_ratio = float(np.mean(support_weights <= 0.05))

    sentence_word_counts = np.asarray(
        [len(word_tokens(s)) for s in student_sentences],
        dtype=float,
    )
    total_words = float(sentence_word_counts.sum())
    supported_words = float(np.sum(sentence_word_counts * support_weights))
    unsupported_words = max(0.0, total_words - supported_words)
    unsupported_word_ratio = unsupported_words / max(1.0, total_words)

    model_word_count = max(1, len(word_tokens(model_answer)))

    # Critically, generic/irrelevant padding does not earn length credit unless
    # it looks like actual reference-supported evidence.
    evidence_length_adequacy_70 = min(
        1.0,
        supported_words / max(1.0, 0.70 * model_word_count),
    )

    # Coverage should be dense in supported evidence. This drops when large
    # amounts of unrelated material are appended.
    evidence_density = concept_soft_coverage * (1.0 - unsupported_word_ratio)

    if rubric_terms:
        rubric_vecs = embedder.encode(rubric_terms)
        rubric_matrix = rubric_vecs @ student_sentence_vecs.T
        rubric_best = rubric_matrix.max(axis=1)
        rubric_mean, rubric_min, _ = safe_stats(rubric_best)
        rubric_coverage_048 = float(np.mean(rubric_best >= RUBRIC_SUPPORT))

        matched_lexical = sum(
            1 for term in rubric_terms if phrase_occurrences(student_answer, term) > 0
        )
        rubric_lexical_coverage = matched_lexical / len(rubric_terms)

        total_term_hits = sum(
            phrase_occurrences(student_answer, term) for term in rubric_terms
        )
        extra_term_hits = max(0, total_term_hits - matched_lexical)
        keyword_stuffing_score = min(
            1.0,
            extra_term_hits / max(1.0, 2.0 * len(rubric_terms)),
        )
    else:
        rubric_mean = 0.0
        rubric_min = 0.0
        rubric_coverage_048 = 0.0
        rubric_lexical_coverage = 0.0
        extra_term_hits = 0
        keyword_stuffing_score = 0.0

    answer_words = word_tokens(student_answer)
    answer_word_count = len(answer_words)

    q_content = set(content_tokens(question_text))
    a_content = content_tokens(student_answer)
    question_copy_ratio = (
        sum(token in q_content for token in a_content) / len(a_content)
        if a_content
        else 0.0
    )

    unique_token_ratio = (
        len(set(answer_words)) / answer_word_count if answer_word_count else 0.0
    )

    keyword_repetition_rate = extra_term_hits / max(1, answer_word_count)

    result = {
        "answer_model_similarity": answer_model_similarity,
        "answer_question_similarity": answer_question_similarity,
        "model_minus_question_similarity": (
            answer_model_similarity - answer_question_similarity
        ),
        "concept_similarity_mean": concept_mean,
        "concept_similarity_min": concept_min,
        "concept_similarity_top3": top_k_mean(concept_best, 3),
        "concept_coverage_042": concept_coverage_042,
        "concept_coverage_058": concept_coverage_058,
        "concept_soft_coverage": concept_soft_coverage,
        "rubric_semantic_mean": rubric_mean,
        "rubric_semantic_min": rubric_min,
        "rubric_coverage_048": rubric_coverage_048,
        "rubric_lexical_coverage": float(rubric_lexical_coverage),
        "sentence_relevance_mean": sentence_relevance_mean,
        "supported_sentence_ratio": supported_sentence_ratio,
        "irrelevant_sentence_ratio": irrelevant_sentence_ratio,
        "unsupported_word_ratio": unsupported_word_ratio,
        "evidence_density": evidence_density,
        "evidence_length_adequacy_70": evidence_length_adequacy_70,
        "question_copy_ratio": question_copy_ratio,
        "unique_token_ratio": unique_token_ratio,
        "repeated_bigram_ratio": repeated_bigram_ratio(answer_words),
        "keyword_repetition_rate": keyword_repetition_rate,
        "keyword_stuffing_score": keyword_stuffing_score,
    }

    return result


def build_feature_frame(
    df: pd.DataFrame,
    embedder: EmbeddingCache,
    show_progress: bool = True,
    label: str = "Feature extraction",
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
        if show_progress and (i == 1 or i % 50 == 0 or i == total):
            print(f"{label}: {i}/{total}")

    out = pd.DataFrame(rows, columns=FEATURE_NAMES)
    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def make_regressor() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        loss="huber",
        n_estimators=180,
        learning_rate=0.030,
        max_depth=2,
        min_samples_leaf=5,
        subsample=0.90,
        random_state=RANDOM_STATE,
    )


def row_base_weight(row: pd.Series) -> float:
    if str(row.get("source", "")) == "human_group":
        return 4.0
    return 1.0


def sample_weights(df: pd.DataFrame, adversarial: bool = False) -> np.ndarray:
    weights = np.asarray([row_base_weight(row) for _, row in df.iterrows()], dtype=float)
    if adversarial:
        weights *= ADVERSARIAL_WEIGHT_MULTIPLIER
    return weights


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


def choose_irrelevant_reference(row: pd.Series, pool_df: pd.DataFrame) -> str:
    other = pool_df[
        pool_df["question_id"].astype(str) != str(row["question_id"])
    ]
    if other.empty:
        return GENERIC_FLUFF
    # Deterministic choice keeps experiments reproducible.
    return clean_text(other.iloc[0]["model_answer"])


def adversarial_variants(row: pd.Series, pool_df: pd.DataFrame) -> dict[str, str]:
    base = clean_text(row["student_answer"])
    question = clean_text(row["question_text"])
    rubric_terms = parse_rubric_terms(row["rubric_keywords"])
    stuffing = (
        " ".join(rubric_terms * 4)
        if rubric_terms
        else "important relevant correct " * 8
    )
    irrelevant = choose_irrelevant_reference(row, pool_df)

    return {
        "generic_verbose_padding": f"{base} {GENERIC_FLUFF} {GENERIC_FLUFF}",
        "question_copy": f"{base} {question} {question} {question}",
        "rubric_keyword_stuffing": f"{base} {stuffing}",
        "irrelevant_topic_padding": f"{base} {irrelevant} {irrelevant}",
        "combined_long_weak": (
            f"{base} {question} {stuffing} {GENERIC_FLUFF} {irrelevant}"
        ),
    }


def build_adversarial_training_frame(train_df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for _, row in train_df.iterrows():
        variants = adversarial_variants(row, train_df)
        for attack_name, attacked_answer in variants.items():
            record = row.to_dict()
            record["student_answer"] = attacked_answer
            record["_attack"] = attack_name
            records.append(record)

    return pd.DataFrame(records)


def run_adversarial_evaluation(
    model: GradientBoostingRegressor,
    test_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    embedder: EmbeddingCache,
    model_label: str,
) -> pd.DataFrame:
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

        for attack, mutated_answer in adversarial_variants(row, pool_df).items():
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
                    "model": model_label,
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


def summarize_adversarial(adversarial: pd.DataFrame) -> dict[str, float]:
    if adversarial.empty:
        return {}

    positive = adversarial["score_delta"].clip(lower=0.0)
    return {
        "mean_delta": float(adversarial["score_delta"].mean()),
        "mean_positive_delta": float(positive.mean()),
        "max_delta": float(adversarial["score_delta"].max()),
        "failure_rate_gt_0_5": float(np.mean(adversarial["score_delta"] > 0.5)),
        "failure_rate_gt_1_0": float(np.mean(adversarial["score_delta"] > 1.0)),
    }


def print_metrics(title: str, metrics: dict[str, float]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(f"MAE:            {metrics['mae']:.3f}")
    print(f"RMSE:           {metrics['rmse']:.3f}")
    print(f"R2:             {metrics['r2']:.3f}")
    print(f"Within 1 mark:  {metrics['within_1_mark']:.1%}")
    print(f"Within 2 marks: {metrics['within_2_marks']:.1%}")


def print_adversarial_summary(title: str, adv: pd.DataFrame) -> dict[str, float]:
    summary = summarize_adversarial(adv)

    print(f"\n{title}")
    print("-" * len(title))

    if not summary:
        print("No adversarial probes were generated.")
        return summary

    print(f"Mean score delta:          {summary['mean_delta']:+.3f}")
    print(f"Mean positive delta:       {summary['mean_positive_delta']:+.3f}")
    print(f"Worst score increase:      {summary['max_delta']:+.3f}")
    print(f"Failures > +0.5 mark:      {summary['failure_rate_gt_0_5']:.1%}")
    print(f"Failures > +1.0 mark:      {summary['failure_rate_gt_1_0']:.1%}")

    by_attack = (
        adv.groupby("attack")["score_delta"]
        .agg(["mean", "max", "count"])
        .sort_values("max", ascending=False)
    )
    print("\nBy attack:")
    print(by_attack.round(3).to_string())
    return summary


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

    print("=" * 76)
    print("RESEARCH GRADER V2B — ADVERSARIAL HARD-NEGATIVE TRAINING")
    print("=" * 76)
    print(f"Dataset rows: {len(df)}")
    print(f"Questions:    {df['question_id'].nunique()}")
    print(f"Encoder:      {ENCODER_NAME}")
    print("Model:        GradientBoostingRegressor (Huber, shallow trees)")
    print("Raw word-count feature: REMOVED")
    print("Raw total-length feature: REMOVED")
    print("Length signal: evidence_length_adequacy_70 only")
    print("Training defense: same-score adversarial invariance augmentation")

    if "source" in df.columns:
        print("\nSources:")
        print(df["source"].fillna("unknown").value_counts().to_string())

    embedder = EmbeddingCache(ENCODER_NAME)

    # Split ORIGINAL rows first. This prevents attacked versions of a held-out
    # answer from leaking into the training set.
    stratify = build_stratify_labels(df)
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    print(f"\nClean train rows: {len(train_df)}")
    print(f"Untouched holdout rows: {len(test_df)}")

    X_train_clean = build_feature_frame(
        train_df,
        embedder,
        label="Clean feature extraction",
    )
    X_test = build_feature_frame(
        test_df,
        embedder,
        label="Holdout feature extraction",
    )

    y_train_clean = train_df["human_score"].to_numpy(dtype=float)
    y_test = test_df["human_score"].to_numpy(dtype=float)

    # Ablation: safe V2B features WITHOUT hard-negative training.
    clean_only_model = make_regressor()
    clean_only_model.fit(
        X_train_clean,
        y_train_clean,
        sample_weight=sample_weights(train_df, adversarial=False),
    )

    clean_holdout_pred = clipped_predict(clean_only_model, X_test)
    clean_metrics = metric_dict(y_test, clean_holdout_pred)
    print_metrics("V2B features only — holdout", clean_metrics)

    clean_adv = run_adversarial_evaluation(
        model=clean_only_model,
        test_df=test_df,
        pool_df=train_df,
        embedder=embedder,
        model_label="v2b_features_only",
    )
    clean_adv_summary = print_adversarial_summary(
        "V2B features only — adversarial robustness",
        clean_adv,
    )

    # Hard-negative / invariance augmentation on TRAIN rows only.
    attack_train_df = build_adversarial_training_frame(train_df)
    print(f"\nAdversarial train rows: {len(attack_train_df)}")

    X_train_adv = build_feature_frame(
        attack_train_df,
        embedder,
        label="Adversarial feature extraction",
    )
    y_train_adv = attack_train_df["human_score"].to_numpy(dtype=float)

    X_robust_train = pd.concat(
        [X_train_clean, X_train_adv],
        ignore_index=True,
    )
    y_robust_train = np.concatenate([y_train_clean, y_train_adv])
    w_robust_train = np.concatenate(
        [
            sample_weights(train_df, adversarial=False),
            sample_weights(attack_train_df, adversarial=True),
        ]
    )

    robust_model = make_regressor()
    robust_model.fit(
        X_robust_train,
        y_robust_train,
        sample_weight=w_robust_train,
    )

    robust_holdout_pred = clipped_predict(robust_model, X_test)
    robust_metrics = metric_dict(y_test, robust_holdout_pred)
    print_metrics("V2B robust model — holdout", robust_metrics)

    low_mask = y_test <= 4.0
    if low_mask.any():
        low_metrics = metric_dict(y_test[low_mask], robust_holdout_pred[low_mask])
        print_metrics("V2B robust model — weak answers (<=4)", low_metrics)
    else:
        low_metrics = {}

    human_metrics: dict[str, float] = {}
    if "source" in test_df.columns:
        human_mask = test_df["source"].fillna("").eq("human_group").to_numpy()
        if human_mask.any():
            human_metrics = metric_dict(
                y_test[human_mask],
                robust_holdout_pred[human_mask],
            )
            print_metrics("V2B robust model — human_group only", human_metrics)

    robust_adv = run_adversarial_evaluation(
        model=robust_model,
        test_df=test_df,
        pool_df=train_df,
        embedder=embedder,
        model_label="v2b_robust",
    )
    robust_adv_summary = print_adversarial_summary(
        "V2B robust model — adversarial robustness",
        robust_adv,
    )

    if clean_adv_summary and robust_adv_summary:
        print("\nAdversarial-training effect")
        print("---------------------------")
        print(
            "Mean positive delta: "
            f"{clean_adv_summary['mean_positive_delta']:+.3f} -> "
            f"{robust_adv_summary['mean_positive_delta']:+.3f}"
        )
        print(
            "Worst increase:      "
            f"{clean_adv_summary['max_delta']:+.3f} -> "
            f"{robust_adv_summary['max_delta']:+.3f}"
        )
        print(
            "Failures > +0.5:     "
            f"{clean_adv_summary['failure_rate_gt_0_5']:.1%} -> "
            f"{robust_adv_summary['failure_rate_gt_0_5']:.1%}"
        )

    holdout = test_df.copy()
    holdout["v2b_features_only_prediction"] = clean_holdout_pred
    holdout["v2b_robust_prediction"] = robust_holdout_pred
    holdout["robust_absolute_error"] = np.abs(
        holdout["human_score"].astype(float).to_numpy() - robust_holdout_pred
    )
    holdout.to_csv(HOLDOUT_PATH, index=False)

    combined_adv = pd.concat([clean_adv, robust_adv], ignore_index=True)
    combined_adv.to_csv(ADVERSARIAL_PATH, index=False)

    importance = pd.DataFrame(
        {
            "feature": FEATURE_NAMES,
            "importance": robust_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(IMPORTANCE_PATH, index=False)

    print("\nTop robust-model feature importances")
    print("------------------------------------")
    print(
        importance.head(14).to_string(
            index=False,
            formatters={"importance": "{:.4f}".format},
        )
    )

    # Final artifact: rebuild clean + adversarial features using ALL 140 rows.
    # This happens only after the untouched holdout evaluation above.
    print("\nTraining final V2B artifact on all rows + invariance attacks...")
    X_all_clean = build_feature_frame(
        df,
        embedder,
        label="Final clean feature extraction",
    )
    all_attack_df = build_adversarial_training_frame(df)
    X_all_adv = build_feature_frame(
        all_attack_df,
        embedder,
        label="Final adversarial feature extraction",
    )

    X_final = pd.concat([X_all_clean, X_all_adv], ignore_index=True)
    y_final = np.concatenate(
        [
            df["human_score"].to_numpy(dtype=float),
            all_attack_df["human_score"].to_numpy(dtype=float),
        ]
    )
    w_final = np.concatenate(
        [
            sample_weights(df, adversarial=False),
            sample_weights(all_attack_df, adversarial=True),
        ]
    )

    final_model = make_regressor()
    final_model.fit(X_final, y_final, sample_weight=w_final)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "version": "research_grader_v2b",
        "model_name": "ResearchGraderV2B-AdversarialGradientBoosting",
        "model": final_model,
        "encoder_name": ENCODER_NAME,
        "feature_names": FEATURE_NAMES,
        "raw_word_count_feature": False,
        "raw_total_length_feature": False,
        "length_feature": "evidence_length_adequacy_70",
        "adversarial_training": True,
        "adversarial_weight_multiplier": ADVERSARIAL_WEIGHT_MULTIPLIER,
        "dataset_rows": int(len(df)),
        "final_training_rows": int(len(X_final)),
        "holdout_metrics": robust_metrics,
        "low_score_holdout_metrics": low_metrics,
        "human_holdout_metrics": human_metrics,
        "features_only_holdout_metrics": clean_metrics,
        "features_only_adversarial_summary": clean_adv_summary,
        "robust_adversarial_summary": robust_adv_summary,
        "thresholds": {
            "concept_support": CONCEPT_SUPPORT,
            "concept_strong": CONCEPT_STRONG,
            "rubric_support": RUBRIC_SUPPORT,
            "sentence_support_low": SENTENCE_SUPPORT_LOW,
            "sentence_support_high": SENTENCE_SUPPORT_HIGH,
        },
    }
    joblib.dump(artifact, ARTIFACT_PATH)

    print("\nSaved")
    print("-----")
    print(f"Artifact:           {ARTIFACT_PATH}")
    print(f"Holdout predictions:{HOLDOUT_PATH}")
    print(f"Adversarial report: {ADVERSARIAL_PATH}")
    print(f"Feature importance: {IMPORTANCE_PATH}")
    print("\nV1 baseline and V2A remain untouched.")


if __name__ == "__main__":
    main()
