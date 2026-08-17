"""Research Grader V2C — atomic rubric + NLI evidence grading.

Project
-------
ML Assisted Grading / Smart Exam Evaluation

V2C goals
---------
1. Grade explicit rubric concepts instead of rewarding whole-answer length.
2. Use Natural Language Inference (NLI) to distinguish:
       entailment / neutral / contradiction
   between student evidence and each atomic rubric criterion.
3. Keep the learned calibration layer small, interpretable, and suitable for
   the project's low-data setting.
4. Prevent obvious gaming shortcuts by:
       - using no raw answer-length feature,
       - using no whole-answer semantic-similarity feature,
       - using positive-constrained regression over "higher-is-better" features,
       - adding same-score invariance attacks during training.
5. Evaluate every one of the 20 human_group answers out-of-fold:
       each of four folds holds out one human response per question.
6. Evaluate adversarial robustness only on held-out human answers, so the
   robustness numbers are not training-set numbers.

Expected inputs
---------------
data/training_dataset_v2c.csv
data/rubric_concepts_v2c.json

Created by
----------
scripts/prepare_research_dataset_v2c.py

Outputs
-------
models/artifacts/research_grader_v2c.joblib
data/research_grader_v2c_human_oof.csv
data/research_grader_v2c_adversarial.csv
data/research_grader_v2c_coefficients.csv
data/research_grader_v2c_concept_diagnostics.csv

Notes
-----
The original data/training_dataset.csv, V1, V2A and V2B are NOT modified.
"""

from __future__ import annotations

import ast
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Paths and experiment configuration
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "training_dataset_v2c.csv"
RUBRIC_PATH = ROOT / "data" / "rubric_concepts_v2c.json"

ARTIFACT_DIR = ROOT / "models" / "artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "research_grader_v2c.joblib"

HUMAN_OOF_PATH = ROOT / "data" / "research_grader_v2c_human_oof.csv"
ADVERSARIAL_PATH = ROOT / "data" / "research_grader_v2c_adversarial.csv"
COEFFICIENT_PATH = ROOT / "data" / "research_grader_v2c_coefficients.csv"
CONCEPT_DIAGNOSTICS_PATH = (
    ROOT / "data" / "research_grader_v2c_concept_diagnostics.csv"
)

NLI_MODEL_NAME = os.getenv(
    "GRADER_NLI_MODEL",
    "cross-encoder/nli-deberta-v3-xsmall",
)

NLI_DEVICE = os.getenv("GRADER_NLI_DEVICE") or None
NLI_BATCH_SIZE = int(os.getenv("GRADER_NLI_BATCH_SIZE", "32"))

RANDOM_STATE = 42

# Human answers are much more valuable supervision than synthetic answers.
HUMAN_SAMPLE_WEIGHT = 5.0
SYNTHETIC_SAMPLE_WEIGHT = 1.0

# Same-score invariance attacks should influence the fit without overwhelming
# the clean examples.
ADVERSARIAL_WEIGHT_MULTIPLIER = 0.35

# Ridge regularization is intentionally fixed instead of tuned against our
# tiny 20-answer human benchmark.
RIDGE_ALPHA = 4.0

# NLI evidence thresholds. These are explicit experiment constants.
ENTAIL_SUPPORT = 0.50
ENTAIL_STRONG = 0.70
SENTENCE_SUPPORT = 0.50

# All predictive features are constructed so that HIGHER = BETTER.
# With Ridge(positive=True), the calibrator cannot learn that obvious gaming
# signals such as repetition should increase the score.
FEATURE_NAMES = [
    "criterion_evidence_mean",
    "criterion_evidence_min",
    "criterion_evidence_top3",
    "criterion_coverage_050",
    "criterion_coverage_070",
    "criterion_entailment_mean",
    "contradiction_safety_mean",
    "contradiction_safety_min",
    "supported_sentence_ratio",
    "evidence_density",
    "no_bigram_repetition",
    "no_keyword_stuffing",
]

LABEL_MAPPING = ["contradiction", "entailment", "neutral"]
CONTRADICTION_INDEX = 0
ENTAILMENT_INDEX = 1
NEUTRAL_INDEX = 2

GENERIC_FLUFF = (
    "This is an important topic in modern technology and it has many uses in "
    "different situations. It can be useful for organizations and users in "
    "many real world applications. The topic has several advantages and is "
    "widely discussed in computer science."
)

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "to",
    "of", "in", "on", "at", "by", "for", "from", "with", "as", "is",
    "are", "was", "were", "be", "been", "being", "it", "its", "this",
    "that", "these", "those", "can", "could", "may", "might", "will",
    "would", "should", "do", "does", "did", "have", "has", "had", "not",
    "what", "why", "how", "when", "where", "which", "who", "into", "about",
    "through", "their", "there", "they", "them", "we", "you", "your",
}


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def clean_text(value: object) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return re.sub(r"\s+", " ", str(value)).strip()


def word_tokens(text: str) -> list[str]:
    return re.findall(
        r"[A-Za-z0-9]+(?:'[A-Za-z]+)?",
        clean_text(text).lower(),
    )


def content_tokens(text: str) -> list[str]:
    return [
        token
        for token in word_tokens(text)
        if len(token) >= 3 and token not in STOPWORDS
    ]


def split_sentences(text: str) -> list[str]:
    """Split an answer into reasonably self-contained evidence units."""
    text = clean_text(text)
    if not text:
        return [""]

    parts = re.split(
        r"(?<=[.!?])\s+|\n+|\s*;\s*",
        text,
    )
    parts = [clean_text(part) for part in parts if clean_text(part)]
    return parts or [text]


def parse_rubric_terms(value: object) -> list[str]:
    raw = clean_text(value)
    if not raw:
        return []

    if raw.startswith("[") and raw.endswith("]"):
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, (list, tuple)):
                return list(
                    dict.fromkeys(
                        clean_text(x)
                        for x in parsed
                        if clean_text(x)
                    )
                )
        except (ValueError, SyntaxError):
            pass

    terms = re.split(r"\s*[|;\n]+\s*|\s*,\s*", raw)
    terms = [clean_text(term).strip("-•* ") for term in terms]
    return list(dict.fromkeys(term for term in terms if term))


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


def top_k_mean(values: np.ndarray, k: int = 3) -> float:
    if values.size == 0:
        return 0.0

    k = min(k, values.size)
    return float(np.sort(values)[-k:].mean())


# ---------------------------------------------------------------------------
# NLI cache
# ---------------------------------------------------------------------------

class NLICache:
    """Batched cached NLI inference.

    For each pair:
        premise    = student evidence sentence
        hypothesis = atomic rubric criterion

    The selected model's documented output order is:
        contradiction, entailment, neutral
    """

    def __init__(
        self,
        model_name: str,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        print(f"Loading NLI cross-encoder: {model_name}")
        if device:
            print(f"NLI device override: {device}")

        self.model_name = model_name
        self.batch_size = batch_size
        self.model = CrossEncoder(
            model_name,
            device=device,
            max_length=256,
        )
        self.cache: dict[tuple[str, str], np.ndarray] = {}

        num_labels = int(getattr(self.model, "num_labels", 3))
        if num_labels != 3:
            raise RuntimeError(
                f"Expected a 3-label NLI model, got num_labels={num_labels}"
            )

    def probabilities(
        self,
        pairs: Iterable[tuple[str, str]],
    ) -> np.ndarray:
        normalized_pairs = [
            (clean_text(premise), clean_text(hypothesis))
            for premise, hypothesis in pairs
        ]

        if not normalized_pairs:
            return np.zeros((0, 3), dtype=np.float32)

        missing = list(
            dict.fromkeys(
                pair
                for pair in normalized_pairs
                if pair not in self.cache
            )
        )

        if missing:
            scores = self.model.predict(
                missing,
                batch_size=self.batch_size,
                show_progress_bar=False,
                apply_softmax=True,
                convert_to_numpy=True,
            )
            scores = np.asarray(scores, dtype=np.float32)

            if scores.ndim != 2 or scores.shape[1] != 3:
                raise RuntimeError(
                    "Unexpected NLI output shape: "
                    f"{scores.shape}; expected (N, 3)."
                )

            for pair, prob in zip(missing, scores):
                self.cache[pair] = prob

        return np.vstack(
            [self.cache[pair] for pair in normalized_pairs]
        )


# ---------------------------------------------------------------------------
# Rubric evidence extraction
# ---------------------------------------------------------------------------

@dataclass
class FeatureResult:
    features: dict[str, float]
    diagnostics: list[dict[str, object]]


def criterion_evidence(
    entailment: np.ndarray,
    contradiction: np.ndarray,
) -> np.ndarray:
    """Convert NLI probabilities into criterion evidence.

    A criterion receives high evidence only if:
      - at least one student sentence entails it, and
      - the answer does not strongly contradict it elsewhere.

    This intentionally punishes internally inconsistent answers.
    """
    return entailment * (1.0 - contradiction)


def make_features(
    row: pd.Series,
    rubric_concepts: dict[str, dict[str, object]],
    nli: NLICache,
) -> FeatureResult:
    question_id = str(row["question_id"])
    student_answer = clean_text(row["student_answer"])

    if question_id not in rubric_concepts:
        raise KeyError(
            f"No V2C rubric concepts configured for {question_id}"
        )

    criteria = [
        clean_text(x)
        for x in rubric_concepts[question_id]["criteria"]
        if clean_text(x)
    ]

    if not criteria:
        raise RuntimeError(
            f"Rubric for {question_id} has no criteria."
        )

    sentences = split_sentences(student_answer)

    # Matrix shape:
    #   criteria x student sentences x NLI labels
    pairs = [
        (sentence, criterion)
        for criterion in criteria
        for sentence in sentences
    ]
    probs = nli.probabilities(pairs)
    probs = probs.reshape(len(criteria), len(sentences), 3)

    entail_matrix = probs[:, :, ENTAILMENT_INDEX]
    contradiction_matrix = probs[:, :, CONTRADICTION_INDEX]

    # Criterion support is based on the best entailing sentence.
    best_entail = entail_matrix.max(axis=1)

    # Any strong contradiction anywhere in the answer should matter.
    max_contradiction = contradiction_matrix.max(axis=1)

    evidence = criterion_evidence(
        best_entail,
        max_contradiction,
    )

    criterion_evidence_mean = float(evidence.mean())
    criterion_evidence_min = float(evidence.min())
    criterion_evidence_top3 = top_k_mean(evidence, 3)

    criterion_coverage_050 = float(
        np.mean(evidence >= ENTAIL_SUPPORT)
    )
    criterion_coverage_070 = float(
        np.mean(evidence >= ENTAIL_STRONG)
    )

    criterion_entailment_mean = float(best_entail.mean())

    contradiction_safety = 1.0 - max_contradiction
    contradiction_safety_mean = float(
        contradiction_safety.mean()
    )
    contradiction_safety_min = float(
        contradiction_safety.min()
    )

    # Sentence-level support: each sentence gets credit only from its strongest
    # rubric entailment, discounted by its strongest contradiction.
    sentence_best_entail = entail_matrix.max(axis=0)
    sentence_max_contradiction = contradiction_matrix.max(axis=0)

    sentence_evidence = (
        sentence_best_entail *
        (1.0 - sentence_max_contradiction)
    )

    supported_sentence_ratio = float(
        np.mean(sentence_evidence >= SENTENCE_SUPPORT)
    )

    sentence_word_counts = np.asarray(
        [max(1, len(word_tokens(sentence))) for sentence in sentences],
        dtype=float,
    )

    # Evidence density is a ratio, not a length feature:
    # adding unsupported padding makes it go down.
    evidence_density = float(
        np.average(
            np.clip(sentence_evidence, 0.0, 1.0),
            weights=sentence_word_counts,
        )
    )

    answer_words = word_tokens(student_answer)

    no_bigram_repetition = float(
        1.0 - repeated_bigram_ratio(answer_words)
    )

    rubric_terms = parse_rubric_terms(
        row.get("rubric_keywords", "")
    )

    if rubric_terms:
        unique_hit_count = sum(
            1
            for term in rubric_terms
            if phrase_occurrences(student_answer, term) > 0
        )
        total_hits = sum(
            phrase_occurrences(student_answer, term)
            for term in rubric_terms
        )
        extra_hits = max(0, total_hits - unique_hit_count)

        stuffing = min(
            1.0,
            extra_hits / max(1.0, 2.0 * len(rubric_terms)),
        )
        no_keyword_stuffing = 1.0 - stuffing
    else:
        no_keyword_stuffing = 1.0

    features = {
        "criterion_evidence_mean": criterion_evidence_mean,
        "criterion_evidence_min": criterion_evidence_min,
        "criterion_evidence_top3": criterion_evidence_top3,
        "criterion_coverage_050": criterion_coverage_050,
        "criterion_coverage_070": criterion_coverage_070,
        "criterion_entailment_mean": criterion_entailment_mean,
        "contradiction_safety_mean": contradiction_safety_mean,
        "contradiction_safety_min": contradiction_safety_min,
        "supported_sentence_ratio": supported_sentence_ratio,
        "evidence_density": evidence_density,
        "no_bigram_repetition": no_bigram_repetition,
        "no_keyword_stuffing": float(no_keyword_stuffing),
    }

    diagnostics: list[dict[str, object]] = []

    for criterion_index, criterion in enumerate(criteria):
        best_sentence_index = int(
            np.argmax(entail_matrix[criterion_index])
        )

        diagnostics.append(
            {
                "criterion_index": criterion_index + 1,
                "criterion": criterion,
                "best_student_sentence": sentences[
                    best_sentence_index
                ],
                "entailment": float(
                    best_entail[criterion_index]
                ),
                "contradiction": float(
                    max_contradiction[criterion_index]
                ),
                "evidence": float(
                    evidence[criterion_index]
                ),
            }
        )

    return FeatureResult(
        features=features,
        diagnostics=diagnostics,
    )


def build_feature_frame(
    df: pd.DataFrame,
    rubric_concepts: dict[str, dict[str, object]],
    nli: NLICache,
    label: str,
    diagnostics: list[dict[str, object]] | None = None,
) -> pd.DataFrame:
    feature_rows: list[dict[str, float]] = []
    total = len(df)

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        result = make_features(
            row=row,
            rubric_concepts=rubric_concepts,
            nli=nli,
        )
        feature_rows.append(result.features)

        if diagnostics is not None:
            row_id = int(row["_row_id"])
            for item in result.diagnostics:
                diagnostics.append(
                    {
                        "_row_id": row_id,
                        "question_id": row["question_id"],
                        "source": row["source"],
                        "human_score": float(
                            row["human_score"]
                        ),
                        "student_answer": row[
                            "student_answer"
                        ],
                        **item,
                    }
                )

        if i == 1 or i % 25 == 0 or i == total:
            print(f"{label}: {i}/{total}")

    out = pd.DataFrame(
        feature_rows,
        columns=FEATURE_NAMES,
        index=df.index,
    )

    return (
        out
        .replace([np.inf, -np.inf], 0.0)
        .fillna(0.0)
        .astype(float)
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def make_calibrator() -> Pipeline:
    """Small monotonic-direction calibrator.

    Every feature is higher-is-better and Ridge has positive coefficients.
    This means the model cannot learn a positive reward for repetition,
    keyword stuffing, contradiction, or unsupported padding.
    """
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "ridge",
                Ridge(
                    alpha=RIDGE_ALPHA,
                    fit_intercept=True,
                    positive=True,
                ),
            ),
        ]
    )


def base_sample_weights(df: pd.DataFrame) -> np.ndarray:
    return np.asarray(
        [
            HUMAN_SAMPLE_WEIGHT
            if str(source) == "human_group"
            else SYNTHETIC_SAMPLE_WEIGHT
            for source in df["source"]
        ],
        dtype=float,
    )


def clipped_predict(
    model: Pipeline,
    X: pd.DataFrame,
) -> np.ndarray:
    return np.clip(
        model.predict(X),
        0.0,
        10.0,
    )


def metric_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    result = {
        "mae": float(
            mean_absolute_error(y_true, y_pred)
        ),
        "rmse": float(
            math.sqrt(
                mean_squared_error(y_true, y_pred)
            )
        ),
        "within_1_mark": float(
            np.mean(np.abs(y_true - y_pred) <= 1.0)
        ),
        "within_2_marks": float(
            np.mean(np.abs(y_true - y_pred) <= 2.0)
        ),
    }

    if len(np.unique(y_true)) >= 2:
        result["r2"] = float(r2_score(y_true, y_pred))
    else:
        result["r2"] = float("nan")

    return result


def print_metrics(
    title: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    metrics = metric_dict(y_true, y_pred)

    print(f"\n## {title}")
    print(f"MAE:            {metrics['mae']:.3f}")
    print(f"RMSE:           {metrics['rmse']:.3f}")
    print(f"R2:             {metrics['r2']:.3f}")
    print(
        f"Within 1 mark:  "
        f"{metrics['within_1_mark'] * 100:.1f}%"
    )
    print(
        f"Within 2 marks: "
        f"{metrics['within_2_marks'] * 100:.1f}%"
    )


def direct_rubric_score(
    features: pd.DataFrame,
) -> np.ndarray:
    """No-learning NLI rubric baseline for comparison."""
    return np.clip(
        10.0 * features["criterion_evidence_mean"].to_numpy(),
        0.0,
        10.0,
    )


# ---------------------------------------------------------------------------
# Adversarial examples
# ---------------------------------------------------------------------------

def choose_irrelevant_answer(
    row: pd.Series,
    pool_df: pd.DataFrame,
) -> str:
    other = pool_df[
        pool_df["question_id"].astype(str)
        != str(row["question_id"])
    ]

    if other.empty:
        return GENERIC_FLUFF

    return clean_text(other.iloc[0]["model_answer"])


def contradiction_text(
    question_id: str,
    rubric_concepts: dict[str, dict[str, object]],
) -> str:
    criteria = rubric_concepts[question_id]["criteria"]

    # This is evaluation-only. Unlike padding attacks, it changes answer
    # correctness, so we do NOT use it as a same-score training example.
    selected = criteria[:2]

    return " ".join(
        f"It is false that {clean_text(criterion).rstrip('.').lower()}."
        for criterion in selected
    )


def adversarial_variants(
    row: pd.Series,
    pool_df: pd.DataFrame,
    rubric_concepts: dict[str, dict[str, object]],
    include_contradiction: bool,
) -> dict[str, str]:
    base = clean_text(row["student_answer"])
    question = clean_text(row["question_text"])

    rubric_terms = parse_rubric_terms(
        row.get("rubric_keywords", "")
    )
    stuffing = (
        " ".join(rubric_terms * 5)
        if rubric_terms
        else "important relevant correct " * 10
    )

    irrelevant = choose_irrelevant_answer(
        row,
        pool_df,
    )

    variants = {
        "generic_verbose_padding": (
            f"{base} {GENERIC_FLUFF} {GENERIC_FLUFF}"
        ),
        "question_copy": (
            f"{base} {question} {question} {question}"
        ),
        "rubric_keyword_stuffing": (
            f"{base} {stuffing}"
        ),
        "irrelevant_topic_padding": (
            f"{base} {irrelevant} {irrelevant}"
        ),
        "combined_long_weak": (
            f"{base} {question} {stuffing} "
            f"{GENERIC_FLUFF} {irrelevant}"
        ),
    }

    if include_contradiction:
        variants["contradiction_injection"] = (
            f"{base} "
            f"{contradiction_text(str(row['question_id']), rubric_concepts)}"
        )

    return variants


def build_invariance_attack_frame(
    df: pd.DataFrame,
    pool_df: pd.DataFrame,
    rubric_concepts: dict[str, dict[str, object]],
) -> pd.DataFrame:
    """Create same-score attacks for weak answers only."""
    weak = df[df["human_score"].astype(float) <= 4.0]

    records: list[dict[str, object]] = []

    for _, row in weak.iterrows():
        variants = adversarial_variants(
            row=row,
            pool_df=pool_df,
            rubric_concepts=rubric_concepts,
            include_contradiction=False,
        )

        for attack_name, attacked_answer in variants.items():
            record = row.to_dict()
            record["student_answer"] = attacked_answer
            record["_attack"] = attack_name
            record["_parent_row_id"] = int(row["_row_id"])
            records.append(record)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Human out-of-fold protocol
# ---------------------------------------------------------------------------

def assign_human_folds(
    df: pd.DataFrame,
) -> pd.Series:
    """Each fold holds out one human response per question.

    The prepared dataset contains exactly four human_group rows per question,
    so this produces four folds and evaluates all 20 human answers exactly once.
    """
    fold = pd.Series(
        -1,
        index=df.index,
        dtype=int,
    )

    human = df[df["source"].astype(str) == "human_group"]

    counts = human.groupby("question_id").size()
    unique_counts = sorted(counts.unique().tolist())

    if unique_counts != [4]:
        raise RuntimeError(
            "V2C expects exactly four human_group answers per question "
            f"for the 4-fold protocol, found: {counts.to_dict()}"
        )

    for _, group in human.groupby(
        "question_id",
        sort=True,
    ):
        ordered_indices = list(group.index)
        for fold_id, idx in enumerate(ordered_indices):
            fold.loc[idx] = fold_id

    return fold


def training_matrix_for_fold(
    clean_df: pd.DataFrame,
    clean_X: pd.DataFrame,
    adversarial_df: pd.DataFrame,
    adversarial_X: pd.DataFrame,
    heldout_row_ids: set[int],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    clean_train_mask = ~clean_df["_row_id"].isin(
        heldout_row_ids
    )
    clean_train = clean_df.loc[clean_train_mask]
    X_clean_train = clean_X.loc[clean_train.index]

    # Exclude attacks derived from held-out human answers.
    if not adversarial_df.empty:
        attack_train_mask = ~adversarial_df[
            "_parent_row_id"
        ].isin(heldout_row_ids)
        attack_train = adversarial_df.loc[
            attack_train_mask
        ]
        X_attack_train = adversarial_X.loc[
            attack_train.index
        ]
    else:
        attack_train = adversarial_df
        X_attack_train = adversarial_X

    if attack_train.empty:
        X_train = X_clean_train.copy()
        y_train = clean_train[
            "human_score"
        ].to_numpy(dtype=float)
        weights = base_sample_weights(clean_train)
        return X_train, y_train, weights

    X_train = pd.concat(
        [X_clean_train, X_attack_train],
        axis=0,
        ignore_index=True,
    )

    y_train = np.concatenate(
        [
            clean_train[
                "human_score"
            ].to_numpy(dtype=float),
            attack_train[
                "human_score"
            ].to_numpy(dtype=float),
        ]
    )

    clean_weights = base_sample_weights(
        clean_train
    )

    attack_weights = (
        base_sample_weights(attack_train)
        * ADVERSARIAL_WEIGHT_MULTIPLIER
    )

    weights = np.concatenate(
        [clean_weights, attack_weights]
    )

    return X_train, y_train, weights


# ---------------------------------------------------------------------------
# Coefficients
# ---------------------------------------------------------------------------

def coefficient_frame(
    model: Pipeline,
) -> pd.DataFrame:
    ridge: Ridge = model.named_steps["ridge"]

    coef = np.asarray(
        ridge.coef_,
        dtype=float,
    )

    return (
        pd.DataFrame(
            {
                "feature": FEATURE_NAMES,
                "standardized_coefficient": coef,
            }
        )
        .sort_values(
            "standardized_coefficient",
            ascending=False,
        )
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 78)
    print(
        "RESEARCH GRADER V2C — ATOMIC RUBRIC + NLI CONTRADICTION SCORING"
    )
    print("=" * 78)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing prepared dataset: {DATA_PATH}\n"
            "Run scripts/prepare_research_dataset_v2c.py first."
        )

    if not RUBRIC_PATH.exists():
        raise FileNotFoundError(
            f"Missing V2C rubric file: {RUBRIC_PATH}\n"
            "Run scripts/prepare_research_dataset_v2c.py first."
        )

    df = pd.read_csv(DATA_PATH)

    with RUBRIC_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        rubric_concepts = json.load(f)

    required = {
        "question_id",
        "question_text",
        "model_answer",
        "student_answer",
        "human_score",
        "rubric_keywords",
        "source",
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"V2C dataset is missing columns: {sorted(missing)}"
        )

    df = df.copy().reset_index(drop=True)
    df["_row_id"] = np.arange(len(df), dtype=int)
    df["human_score"] = pd.to_numeric(
        df["human_score"],
        errors="raise",
    ).astype(float)

    human_count = int(
        (df["source"].astype(str) == "human_group").sum()
    )
    synthetic_count = int(
        (df["source"].astype(str) == "synthetic_augmented").sum()
    )

    print(f"Dataset rows:       {len(df)}")
    print(f"Synthetic rows:     {synthetic_count}")
    print(f"Human gold rows:    {human_count}")
    print(f"Questions:          {df['question_id'].nunique()}")
    print(f"NLI model:          {NLI_MODEL_NAME}")
    print(f"Calibrator:         Ridge(alpha={RIDGE_ALPHA}, positive=True)")
    print("Raw word count:     REMOVED")
    print("Whole-answer sim:   REMOVED")
    print("Total-length signal: REMOVED")
    print("Human evaluation:   4-fold, one human answer/question/fold")

    nli = NLICache(
        model_name=NLI_MODEL_NAME,
        device=NLI_DEVICE,
        batch_size=NLI_BATCH_SIZE,
    )

    # ---------------------------------------------------------------
    # Extract clean features once.
    # ---------------------------------------------------------------
    concept_diagnostics: list[
        dict[str, object]
    ] = []

    clean_X = build_feature_frame(
        df=df,
        rubric_concepts=rubric_concepts,
        nli=nli,
        label="Clean NLI feature extraction",
        diagnostics=concept_diagnostics,
    )

    diagnostics_df = pd.DataFrame(
        concept_diagnostics
    )

    # ---------------------------------------------------------------
    # Build and extract same-score weak-answer attacks once.
    # ---------------------------------------------------------------
    adversarial_train_df = build_invariance_attack_frame(
        df=df,
        pool_df=df,
        rubric_concepts=rubric_concepts,
    )

    print(
        f"\nWeak-answer invariance attacks: "
        f"{len(adversarial_train_df)}"
    )

    if not adversarial_train_df.empty:
        adversarial_train_X = build_feature_frame(
            df=adversarial_train_df,
            rubric_concepts=rubric_concepts,
            nli=nli,
            label="Attack NLI feature extraction",
        )
    else:
        adversarial_train_X = pd.DataFrame(
            columns=FEATURE_NAMES
        )

    # ---------------------------------------------------------------
    # Human out-of-fold evaluation.
    # ---------------------------------------------------------------
    human_fold = assign_human_folds(df)
    df["_human_fold"] = human_fold

    oof_records: list[dict[str, object]] = []
    adversarial_eval_records: list[
        dict[str, object]
    ] = []

    for fold_id in range(4):
        test_mask = df["_human_fold"] == fold_id
        test_df = df.loc[test_mask].copy()

        if len(test_df) != 5:
            raise RuntimeError(
                f"Fold {fold_id} should contain 5 held-out human rows, "
                f"found {len(test_df)}"
            )

        heldout_row_ids = set(
            test_df["_row_id"].astype(int).tolist()
        )

        X_train, y_train, train_weights = (
            training_matrix_for_fold(
                clean_df=df,
                clean_X=clean_X,
                adversarial_df=adversarial_train_df,
                adversarial_X=adversarial_train_X,
                heldout_row_ids=heldout_row_ids,
            )
        )

        model = make_calibrator()
        model.fit(
            X_train,
            y_train,
            ridge__sample_weight=train_weights,
        )

        X_test = clean_X.loc[
            test_df.index
        ]
        predictions = clipped_predict(
            model,
            X_test,
        )
        direct_predictions = direct_rubric_score(
            X_test
        )

        for local_pos, (idx, row) in enumerate(
            test_df.iterrows()
        ):
            pred = float(predictions[local_pos])
            direct_pred = float(
                direct_predictions[local_pos]
            )

            oof_records.append(
                {
                    "fold": fold_id,
                    "_row_id": int(row["_row_id"]),
                    "question_id": row["question_id"],
                    "human_score": float(
                        row["human_score"]
                    ),
                    "prediction": pred,
                    "direct_nli_rubric_score": direct_pred,
                    "absolute_error": abs(
                        float(row["human_score"]) - pred
                    ),
                    "student_answer": row[
                        "student_answer"
                    ],
                }
            )

            # Robustness evaluation is focused on genuinely weak answers.
            if float(row["human_score"]) > 4.0:
                continue

            base_score = pred

            variants = adversarial_variants(
                row=row,
                pool_df=df,
                rubric_concepts=rubric_concepts,
                include_contradiction=True,
            )

            for attack_name, attacked_answer in variants.items():
                attacked_row = row.copy()
                attacked_row[
                    "student_answer"
                ] = attacked_answer

                attack_result = make_features(
                    row=attacked_row,
                    rubric_concepts=rubric_concepts,
                    nli=nli,
                )

                attack_X = pd.DataFrame(
                    [attack_result.features],
                    columns=FEATURE_NAMES,
                )

                attacked_score = float(
                    clipped_predict(
                        model,
                        attack_X,
                    )[0]
                )

                adversarial_eval_records.append(
                    {
                        "fold": fold_id,
                        "_row_id": int(
                            row["_row_id"]
                        ),
                        "question_id": row[
                            "question_id"
                        ],
                        "human_score": float(
                            row["human_score"]
                        ),
                        "attack": attack_name,
                        "base_prediction": base_score,
                        "attacked_prediction": attacked_score,
                        "score_delta": (
                            attacked_score - base_score
                        ),
                        "base_answer": row[
                            "student_answer"
                        ],
                        "attacked_answer": attacked_answer,
                    }
                )

        print(
            f"Human OOF fold {fold_id + 1}/4 complete "
            f"(train clean={len(df) - len(test_df)}, "
            f"held out={len(test_df)})"
        )

    oof_df = pd.DataFrame(
        oof_records
    ).sort_values(
        ["question_id", "_row_id"]
    ).reset_index(drop=True)

    adversarial_eval_df = pd.DataFrame(
        adversarial_eval_records
    )

    if len(oof_df) != 20:
        raise RuntimeError(
            "Expected 20 human OOF predictions, "
            f"found {len(oof_df)}"
        )

    y_human = oof_df[
        "human_score"
    ].to_numpy(dtype=float)
    y_pred = oof_df[
        "prediction"
    ].to_numpy(dtype=float)
    y_direct = oof_df[
        "direct_nli_rubric_score"
    ].to_numpy(dtype=float)

    print_metrics(
        "V2C — HUMAN GOLD OOF",
        y_human,
        y_pred,
    )

    print_metrics(
        "Direct NLI rubric score — HUMAN GOLD OOF",
        y_human,
        y_direct,
    )

    weak_oof = oof_df[
        oof_df["human_score"] <= 4.0
    ]

    if not weak_oof.empty:
        print_metrics(
            "V2C — HUMAN GOLD WEAK ANSWERS (<=4)",
            weak_oof[
                "human_score"
            ].to_numpy(dtype=float),
            weak_oof[
                "prediction"
            ].to_numpy(dtype=float),
        )

    print("\n## Human OOF MAE by question")
    per_question = (
        oof_df.groupby("question_id")
        .apply(
            lambda g: mean_absolute_error(
                g["human_score"],
                g["prediction"],
            ),
            include_groups=False,
        )
        .rename("mae")
    )
    print(per_question.round(3).to_string())

    # ---------------------------------------------------------------
    # Adversarial OOF report.
    # ---------------------------------------------------------------
    print("\n## V2C — HELD-OUT HUMAN ADVERSARIAL ROBUSTNESS")

    if adversarial_eval_df.empty:
        print("No human answers <=4 were available for robustness testing.")
    else:
        padding_attacks = adversarial_eval_df[
            adversarial_eval_df["attack"]
            != "contradiction_injection"
        ].copy()

        positive_delta = padding_attacks[
            "score_delta"
        ].clip(lower=0.0)

        print(
            f"Weak human base answers:      "
            f"{padding_attacks['_row_id'].nunique()}"
        )
        print(
            f"Mean score delta:             "
            f"{padding_attacks['score_delta'].mean():+.3f}"
        )
        print(
            f"Mean positive delta:          "
            f"{positive_delta.mean():+.3f}"
        )
        print(
            f"Worst score increase:         "
            f"{padding_attacks['score_delta'].max():+.3f}"
        )
        print(
            f"Failures > +0.5 mark:         "
            f"{(padding_attacks['score_delta'] > 0.5).mean() * 100:.1f}%"
        )
        print(
            f"Failures > +1.0 mark:         "
            f"{(padding_attacks['score_delta'] > 1.0).mean() * 100:.1f}%"
        )

        print("\nPadding/gaming attacks:")
        by_attack = (
            padding_attacks.groupby("attack")[
                "score_delta"
            ]
            .agg(["mean", "max", "count"])
            .sort_values(
                "mean",
                ascending=False,
            )
        )
        print(by_attack.round(3).to_string())

        contradiction_rows = adversarial_eval_df[
            adversarial_eval_df["attack"]
            == "contradiction_injection"
        ]

        if not contradiction_rows.empty:
            print("\nContradiction injection:")
            print(
                f"Mean score delta: "
                f"{contradiction_rows['score_delta'].mean():+.3f}"
            )
            print(
                f"Worst score increase: "
                f"{contradiction_rows['score_delta'].max():+.3f}"
            )
            print(
                "Desired behavior: contradiction should not raise the score."
            )

    # ---------------------------------------------------------------
    # Train final artifact on all clean data + all weak invariance attacks.
    # ---------------------------------------------------------------
    print(
        "\nTraining final V2C artifact on all repaired rows "
        "+ weak-answer invariance attacks..."
    )

    if adversarial_train_df.empty:
        X_final = clean_X.copy()
        y_final = df[
            "human_score"
        ].to_numpy(dtype=float)
        w_final = base_sample_weights(df)
    else:
        X_final = pd.concat(
            [
                clean_X.reset_index(drop=True),
                adversarial_train_X.reset_index(
                    drop=True
                ),
            ],
            axis=0,
            ignore_index=True,
        )

        y_final = np.concatenate(
            [
                df[
                    "human_score"
                ].to_numpy(dtype=float),
                adversarial_train_df[
                    "human_score"
                ].to_numpy(dtype=float),
            ]
        )

        w_final = np.concatenate(
            [
                base_sample_weights(df),
                base_sample_weights(
                    adversarial_train_df
                )
                * ADVERSARIAL_WEIGHT_MULTIPLIER,
            ]
        )

    final_model = make_calibrator()
    final_model.fit(
        X_final,
        y_final,
        ridge__sample_weight=w_final,
    )

    coefficients = coefficient_frame(
        final_model
    )

    print("\n## Final standardized coefficients")
    print(
        coefficients.to_string(
            index=False,
            formatters={
                "standardized_coefficient": (
                    lambda x: f"{x:.4f}"
                )
            },
        )
    )

    # ---------------------------------------------------------------
    # Save experiment artifacts.
    # ---------------------------------------------------------------
    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    HUMAN_OOF_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifact = {
        "version": "research_grader_v2c",
        "model": final_model,
        "feature_names": FEATURE_NAMES,
        "nli_model_name": NLI_MODEL_NAME,
        "nli_label_mapping": LABEL_MAPPING,
        "rubric_concepts": rubric_concepts,
        "entail_support": ENTAIL_SUPPORT,
        "entail_strong": ENTAIL_STRONG,
        "sentence_support": SENTENCE_SUPPORT,
        "ridge_alpha": RIDGE_ALPHA,
        "human_sample_weight": HUMAN_SAMPLE_WEIGHT,
        "synthetic_sample_weight": SYNTHETIC_SAMPLE_WEIGHT,
        "adversarial_weight_multiplier": (
            ADVERSARIAL_WEIGHT_MULTIPLIER
        ),
        "training_rows_clean": int(len(df)),
        "training_rows_adversarial": int(
            len(adversarial_train_df)
        ),
        "design_notes": {
            "raw_word_count_feature": False,
            "raw_length_feature": False,
            "whole_answer_similarity_feature": False,
            "atomic_rubric_nli": True,
            "contradiction_aware": True,
            "positive_constrained_calibrator": True,
        },
    }

    joblib.dump(
        artifact,
        ARTIFACT_PATH,
    )

    oof_df.to_csv(
        HUMAN_OOF_PATH,
        index=False,
    )

    adversarial_eval_df.to_csv(
        ADVERSARIAL_PATH,
        index=False,
    )

    coefficients.to_csv(
        COEFFICIENT_PATH,
        index=False,
    )

    diagnostics_df.to_csv(
        CONCEPT_DIAGNOSTICS_PATH,
        index=False,
    )

    print("\n## Saved")
    print(f"Artifact:            {ARTIFACT_PATH}")
    print(f"Human OOF:           {HUMAN_OOF_PATH}")
    print(f"Adversarial report:  {ADVERSARIAL_PATH}")
    print(f"Coefficients:        {COEFFICIENT_PATH}")
    print(f"Concept diagnostics: {CONCEPT_DIAGNOSTICS_PATH}")

    print(
        "\nV1 baseline, V2A, V2B, and the original 140-row dataset "
        "remain untouched."
    )


if __name__ == "__main__":
    main()
