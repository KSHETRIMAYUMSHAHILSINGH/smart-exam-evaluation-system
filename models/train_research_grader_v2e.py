"""Research Grader V2E — evidence bottleneck around V2B.

Motivation
----------
V2D improved weak-answer MAE, but it did not solve the core padding problem:
weak answers still gained large amounts under combined-long attacks.

The reason is structural:
- V2B and V2D both use max-over-sentences semantic matching.
- When extra sentences are appended, a per-concept maximum can only stay the
  same or increase.
- Therefore a rubric ceiling built from those maxima can become MORE permissive
  after padding.

V2E changes the guard from "score the whole answer, then cap it" to:

    full student answer
            |
            v
    sentence-level rubric evidence extraction
            |
            v
    evidence-only answer
            |
            v
    V2B score on evidence only
            |
            +----> hard upper bound on full-answer V2B score
            |
            v
    contradiction penalty
            |
            v
        final score

Unsupported padding is discarded before it can contribute to the score.

Research design
---------------
V2E is deliberately hybrid:
- V2B remains the primary learned scorer.
- MiniLM embeddings find candidate evidence for atomic rubric criteria.
- NLI is NOT used as the direct grader.
  It only validates candidate evidence and detects contradiction.
- Repetition/question-copy/keyword-only text is downweighted before evidence
  selection.
- The final guard can only LOWER the base score; it can never raise it.

Evaluation
----------
A. HUMAN_OOF:
   Same 20 human_group answers, same 4-fold protocol.

B. WEAK_OOF:
   All score <= 4 rows, 5-fold question-stratified OOF.

For both sets:
    V2B vs V2E
and the same six attacks:
    generic_verbose_padding
    question_copy
    rubric_keyword_stuffing
    irrelevant_topic_padding
    combined_long_weak
    contradiction_injection

Inputs
------
data/training_dataset_v2c.csv
data/rubric_concepts_v2c.json
models/train_research_grader_v2b.py
models/train_research_grader_v2c.py

Outputs
-------
models/artifacts/research_grader_v2e.joblib
data/research_grader_v2e_human_oof.csv
data/research_grader_v2e_weak_oof.csv
data/research_grader_v2e_adversarial.csv
data/research_grader_v2e_summary.csv
data/research_grader_v2e_evidence_diagnostics.csv

V1/V2A/V2B/V2C/V2D and the original dataset are not modified.
"""

from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import StratifiedKFold


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

import train_research_grader_v2b as v2b  # noqa: E402
import train_research_grader_v2c as v2c  # noqa: E402


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_PATH = ROOT / "data" / "training_dataset_v2c.csv"
RUBRIC_PATH = ROOT / "data" / "rubric_concepts_v2c.json"

ARTIFACT_PATH = (
    ROOT / "models" / "artifacts" / "research_grader_v2e.joblib"
)

HUMAN_OOF_PATH = (
    ROOT / "data" / "research_grader_v2e_human_oof.csv"
)

WEAK_OOF_PATH = (
    ROOT / "data" / "research_grader_v2e_weak_oof.csv"
)

ADVERSARIAL_PATH = (
    ROOT / "data" / "research_grader_v2e_adversarial.csv"
)

SUMMARY_PATH = (
    ROOT / "data" / "research_grader_v2e_summary.csv"
)

EVIDENCE_DIAGNOSTICS_PATH = (
    ROOT / "data" / "research_grader_v2e_evidence_diagnostics.csv"
)


# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------

RANDOM_STATE = 42

# Embedding similarity -> 0..1 semantic support.
SEMANTIC_LOW = 0.32
SEMANTIC_HIGH = 0.70

# Candidate sentence must have at least this semantic support to a rubric
# criterion before NLI validation is considered.
MIN_SEMANTIC_SUPPORT = 0.34

# Validated support used to count a rubric concept as represented.
VALIDATED_CONCEPT_SUPPORT = 0.36

# A student sentence is retained in the evidence-only answer if its strongest
# validated rubric support is at least this value.
EVIDENCE_SENTENCE_THRESHOLD = 0.30

# Very high lexical overlap with the QUESTION but little new content is likely
# question-copying rather than evidence.
QUESTION_COPY_START = 0.72

# Evidence-only score is allowed a small amount of slack so concise wording or
# imperfect extraction does not over-penalize strong legitimate answers.
EVIDENCE_SCORE_SLACK = 0.85

# A second monotonic cap learned from validated rubric coverage. This is only a
# safety backstop; the evidence-only V2B score is the primary V2E ceiling.
COVERAGE_CAP_SLACK = 1.00
MIN_COVERAGE_CAP = 1.5

# NLI contradiction is used only as a high-confidence penalty.
CONTRADICTION_START = 0.72
MAX_CONTRADICTION_PENALTY = 2.5

# NLI is a validator, not the direct score.
# Even modest entailment retains some semantic evidence; strong contradiction
# suppresses it aggressively.
NLI_ENTAILMENT_FLOOR = 0.45

# Near-duplicate evidence sentences are collapsed so repetition cannot create
# artificial evidence volume.
EVIDENCE_DEDUP_COSINE = 0.94


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
# Generic helpers
# ---------------------------------------------------------------------------

def clean_text(value: object) -> str:
    return v2c.clean_text(value)


def word_tokens(text: str) -> list[str]:
    return v2b.word_tokens(text)


def content_tokens(text: str) -> list[str]:
    return [
        token
        for token in word_tokens(text)
        if len(token) >= 3 and token not in STOPWORDS
    ]


def normalized_semantic_support(
    cosine_values: np.ndarray,
) -> np.ndarray:
    denom = SEMANTIC_HIGH - SEMANTIC_LOW

    if denom <= 0:
        raise RuntimeError("Invalid semantic thresholds.")

    return np.clip(
        (cosine_values - SEMANTIC_LOW) / denom,
        0.0,
        1.0,
    )


def lexical_question_copy_ratio(
    sentence: str,
    question: str,
) -> float:
    s_tokens = content_tokens(sentence)
    q_tokens = set(content_tokens(question))

    if not s_tokens or not q_tokens:
        return 0.0

    return float(
        sum(token in q_tokens for token in s_tokens)
        / len(s_tokens)
    )


def question_novelty_factor(
    sentence: str,
    question: str,
) -> float:
    """1.0 for normal evidence, dropping toward 0 for near-question copies."""
    overlap = lexical_question_copy_ratio(
        sentence,
        question,
    )

    if overlap <= QUESTION_COPY_START:
        return 1.0

    return float(
        np.clip(
            1.0
            - (
                (overlap - QUESTION_COPY_START)
                / max(1e-9, 1.0 - QUESTION_COPY_START)
            ),
            0.0,
            1.0,
        )
    )


def sentence_form_factor(sentence: str) -> float:
    """Conservative anti-stuffing signal.

    This does NOT reward length. It only discounts obvious token lists and
    extreme repetition.
    """
    tokens = word_tokens(sentence)

    if not tokens:
        return 0.0

    unique_ratio = len(set(tokens)) / len(tokens)
    repeat_penalty = 1.0 - v2b.repeated_bigram_ratio(tokens)

    # Very short fragments can still be valid evidence, so keep a generous
    # lower bound.
    if len(tokens) <= 2:
        completeness = 0.55
    elif len(tokens) <= 4:
        completeness = 0.75
    else:
        completeness = 1.0

    return float(
        np.clip(
            0.45 * unique_ratio
            + 0.35 * repeat_penalty
            + 0.20 * completeness,
            0.0,
            1.0,
        )
    )


def metric_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    result = {
        "mae": float(
            mean_absolute_error(
                y_true,
                y_pred,
            )
        ),
        "rmse": float(
            math.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred,
                )
            )
        ),
        "within_1_mark": float(
            np.mean(
                np.abs(y_true - y_pred) <= 1.0
            )
        ),
        "within_2_marks": float(
            np.mean(
                np.abs(y_true - y_pred) <= 2.0
            )
        ),
    }

    if len(np.unique(y_true)) >= 2:
        result["r2"] = float(
            r2_score(
                y_true,
                y_pred,
            )
        )
    else:
        result["r2"] = float("nan")

    return result


def print_metrics(
    title: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    m = metric_dict(
        y_true,
        y_pred,
    )

    print(f"\n## {title}")
    print(f"MAE:            {m['mae']:.3f}")
    print(f"RMSE:           {m['rmse']:.3f}")
    print(f"R2:             {m['r2']:.3f}")
    print(
        f"Within 1 mark:  "
        f"{m['within_1_mark'] * 100:.1f}%"
    )
    print(
        f"Within 2 marks: "
        f"{m['within_2_marks'] * 100:.1f}%"
    )

    return m


# ---------------------------------------------------------------------------
# Human folds
# ---------------------------------------------------------------------------

def assign_human_folds(
    df: pd.DataFrame,
) -> pd.Series:
    fold = pd.Series(
        -1,
        index=df.index,
        dtype=int,
    )

    human = df[
        df["source"]
        .astype(str)
        .eq("human_group")
    ]

    counts = human.groupby(
        "question_id"
    ).size()

    if not (counts == 4).all():
        raise RuntimeError(
            "Expected four human_group rows per question. "
            f"Found: {counts.to_dict()}"
        )

    for _, group in human.groupby(
        "question_id",
        sort=True,
    ):
        for fold_id, idx in enumerate(group.index):
            fold.loc[idx] = fold_id

    return fold


# ---------------------------------------------------------------------------
# Evidence bottleneck
# ---------------------------------------------------------------------------

@dataclass
class EvidenceProfile:
    evidence_text: str
    selected_sentences: list[str]
    criterion_support: np.ndarray
    coverage_index: float
    supported_count: int
    max_contradiction: float
    original_sentence_count: int
    selected_sentence_count: int
    evidence_word_ratio: float
    diagnostics: list[dict[str, object]]


def deduplicate_evidence_sentences(
    sentences: list[str],
    embedder: v2b.EmbeddingCache,
) -> list[str]:
    if not sentences:
        return []

    # Preserve order while removing exact duplicates first.
    exact_unique = list(
        dict.fromkeys(
            clean_text(sentence)
            for sentence in sentences
            if clean_text(sentence)
        )
    )

    if len(exact_unique) <= 1:
        return exact_unique

    vecs = embedder.encode(
        exact_unique
    )

    kept: list[int] = []

    for idx in range(len(exact_unique)):
        if not kept:
            kept.append(idx)
            continue

        sims = vecs[kept] @ vecs[idx]

        if float(np.max(sims)) < EVIDENCE_DEDUP_COSINE:
            kept.append(idx)

    return [
        exact_unique[idx]
        for idx in kept
    ]


def extract_evidence_profile(
    row: pd.Series,
    rubric_concepts: dict[str, dict[str, object]],
    embedder: v2b.EmbeddingCache,
    nli: v2c.NLICache,
) -> EvidenceProfile:
    qid = str(row["question_id"])

    if qid not in rubric_concepts:
        raise KeyError(
            f"Missing rubric concepts for {qid}"
        )

    question = clean_text(
        row["question_text"]
    )
    answer = clean_text(
        row["student_answer"]
    )

    criteria = [
        clean_text(x)
        for x in rubric_concepts[qid]["criteria"]
        if clean_text(x)
    ]

    sentences = [
        clean_text(s)
        for s in v2b.split_sentences(answer)
        if clean_text(s)
    ]

    if not sentences:
        sentences = [""]

    criterion_vecs = embedder.encode(
        criteria
    )
    sentence_vecs = embedder.encode(
        sentences
    )

    cosine_matrix = (
        criterion_vecs
        @ sentence_vecs.T
    )

    semantic_matrix = normalized_semantic_support(
        cosine_matrix
    )

    # We use NLI only for semantically plausible pairs. For weak semantic
    # matches, validated support is zero without spending NLI compute.
    candidate_pairs: list[tuple[str, str]] = []
    candidate_positions: list[tuple[int, int]] = []

    for ci in range(len(criteria)):
        for si in range(len(sentences)):
            if (
                semantic_matrix[ci, si]
                >= MIN_SEMANTIC_SUPPORT
            ):
                candidate_pairs.append(
                    (
                        sentences[si],
                        criteria[ci],
                    )
                )
                candidate_positions.append(
                    (ci, si)
                )

    entail_matrix = np.zeros_like(
        semantic_matrix,
        dtype=np.float32,
    )
    contradiction_matrix = np.zeros_like(
        semantic_matrix,
        dtype=np.float32,
    )

    if candidate_pairs:
        probs = nli.probabilities(
            candidate_pairs
        )

        for (
            (ci, si),
            prob,
        ) in zip(
            candidate_positions,
            probs,
        ):
            contradiction_matrix[
                ci,
                si,
            ] = float(
                prob[v2c.CONTRADICTION_INDEX]
            )
            entail_matrix[
                ci,
                si,
            ] = float(
                prob[v2c.ENTAILMENT_INDEX]
            )

    sentence_novelty = np.asarray(
        [
            question_novelty_factor(
                sentence,
                question,
            )
            for sentence in sentences
        ],
        dtype=np.float32,
    )

    sentence_form = np.asarray(
        [
            sentence_form_factor(
                sentence
            )
            for sentence in sentences
        ],
        dtype=np.float32,
    )

    # NLI modulates semantic evidence instead of directly defining the grade.
    nli_entail_factor = (
        NLI_ENTAILMENT_FLOOR
        + (
            1.0 - NLI_ENTAILMENT_FLOOR
        )
        * entail_matrix
    )

    validated = (
        semantic_matrix
        * nli_entail_factor
        * (1.0 - contradiction_matrix)
        * sentence_novelty[np.newaxis, :]
        * sentence_form[np.newaxis, :]
    )

    # Pairs below semantic threshold had entailment=0 but would otherwise
    # retain the NLI floor. Explicitly zero them.
    validated[
        semantic_matrix < MIN_SEMANTIC_SUPPORT
    ] = 0.0

    criterion_support = validated.max(
        axis=1
    )

    sentence_support = validated.max(
        axis=0
    )

    selected_indices = [
        idx
        for idx, support in enumerate(
            sentence_support
        )
        if float(support)
        >= EVIDENCE_SENTENCE_THRESHOLD
    ]

    selected_sentences = [
        sentences[idx]
        for idx in selected_indices
    ]

    selected_sentences = deduplicate_evidence_sentences(
        selected_sentences,
        embedder,
    )

    evidence_text = clean_text(
        " ".join(selected_sentences)
    )

    supported_count = int(
        np.sum(
            criterion_support
            >= VALIDATED_CONCEPT_SUPPORT
        )
    )

    hard_coverage = (
        supported_count
        / max(1, len(criteria))
    )

    mean_support = float(
        criterion_support.mean()
    )

    coverage_index = float(
        0.65 * mean_support
        + 0.35 * hard_coverage
    )

    max_contradiction = float(
        contradiction_matrix.max()
        if contradiction_matrix.size
        else 0.0
    )

    total_words = max(
        1,
        len(
            word_tokens(answer)
        ),
    )

    evidence_words = len(
        word_tokens(evidence_text)
    )

    evidence_word_ratio = float(
        evidence_words / total_words
    )

    diagnostics: list[
        dict[str, object]
    ] = []

    for ci, criterion in enumerate(
        criteria
    ):
        best_si = int(
            np.argmax(
                validated[ci]
            )
        )

        diagnostics.append(
            {
                "question_id": qid,
                "criterion_index": ci + 1,
                "criterion": criterion,
                "best_sentence": sentences[
                    best_si
                ],
                "semantic_cosine": float(
                    cosine_matrix[
                        ci,
                        best_si,
                    ]
                ),
                "semantic_support": float(
                    semantic_matrix[
                        ci,
                        best_si,
                    ]
                ),
                "entailment": float(
                    entail_matrix[
                        ci,
                        best_si,
                    ]
                ),
                "contradiction": float(
                    contradiction_matrix[
                        ci,
                        best_si,
                    ]
                ),
                "validated_support": float(
                    validated[
                        ci,
                        best_si,
                    ]
                ),
            }
        )

    return EvidenceProfile(
        evidence_text=evidence_text,
        selected_sentences=selected_sentences,
        criterion_support=criterion_support,
        coverage_index=coverage_index,
        supported_count=supported_count,
        max_contradiction=max_contradiction,
        original_sentence_count=len(
            sentences
        ),
        selected_sentence_count=len(
            selected_sentences
        ),
        evidence_word_ratio=evidence_word_ratio,
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Coverage backstop
# ---------------------------------------------------------------------------

def fit_coverage_guard(
    train_df: pd.DataFrame,
    coverage_by_row_id: dict[int, float],
) -> IsotonicRegression:
    x = np.asarray(
        [
            coverage_by_row_id[
                int(row_id)
            ]
            for row_id in train_df[
                "_row_id"
            ]
        ],
        dtype=float,
    )

    y = train_df[
        "human_score"
    ].to_numpy(
        dtype=float
    )

    weights = v2b.sample_weights(
        train_df,
        adversarial=False,
    )

    iso = IsotonicRegression(
        y_min=0.0,
        y_max=10.0,
        increasing=True,
        out_of_bounds="clip",
    )

    iso.fit(
        x,
        y,
        sample_weight=weights,
    )

    return iso


def coverage_ceiling(
    guard: IsotonicRegression,
    coverage_index: float,
) -> float:
    expected = float(
        guard.predict(
            np.asarray(
                [coverage_index],
                dtype=float,
            )
        )[0]
    )

    return float(
        np.clip(
            max(
                MIN_COVERAGE_CAP,
                expected
                + COVERAGE_CAP_SLACK,
            ),
            0.0,
            10.0,
        )
    )


# ---------------------------------------------------------------------------
# Contradiction penalty
# ---------------------------------------------------------------------------

def contradiction_penalty(
    probability: float,
) -> float:
    if probability <= CONTRADICTION_START:
        return 0.0

    scaled = (
        probability
        - CONTRADICTION_START
    ) / max(
        1e-9,
        1.0 - CONTRADICTION_START,
    )

    return float(
        MAX_CONTRADICTION_PENALTY
        * np.clip(
            scaled,
            0.0,
            1.0,
        )
    )


# ---------------------------------------------------------------------------
# V2B base training
# ---------------------------------------------------------------------------

def build_v2b_attack_frame(
    df: pd.DataFrame,
) -> pd.DataFrame:
    records: list[
        dict[str, object]
    ] = []

    for _, row in df.iterrows():
        variants = v2b.adversarial_variants(
            row,
            df,
        )

        for attack_name, attacked_answer in variants.items():
            record = row.to_dict()
            record[
                "student_answer"
            ] = attacked_answer
            record[
                "_attack"
            ] = attack_name
            record[
                "_parent_row_id"
            ] = int(
                row["_row_id"]
            )
            records.append(
                record
            )

    return pd.DataFrame(
        records
    ).reset_index(
        drop=True
    )


def build_v2b_feature_frame(
    df: pd.DataFrame,
    embedder: v2b.EmbeddingCache,
    label: str,
) -> pd.DataFrame:
    out = v2b.build_feature_frame(
        df.reset_index(
            drop=True
        ),
        embedder=embedder,
        show_progress=True,
        label=label,
    )

    out.index = df.index
    return out


def train_v2b_fold(
    clean_df: pd.DataFrame,
    clean_X: pd.DataFrame,
    attack_df: pd.DataFrame,
    attack_X: pd.DataFrame,
    heldout_ids: set[int],
):
    clean_mask = ~clean_df[
        "_row_id"
    ].isin(
        heldout_ids
    )

    train_clean = clean_df.loc[
        clean_mask
    ]

    X_clean = clean_X.loc[
        train_clean.index
    ]

    attack_mask = ~attack_df[
        "_parent_row_id"
    ].isin(
        heldout_ids
    )

    train_attack = attack_df.loc[
        attack_mask
    ]

    X_attack = attack_X.loc[
        train_attack.index
    ]

    X_train = pd.concat(
        [
            X_clean.reset_index(
                drop=True
            ),
            X_attack.reset_index(
                drop=True
            ),
        ],
        ignore_index=True,
    )

    y_train = np.concatenate(
        [
            train_clean[
                "human_score"
            ].to_numpy(
                dtype=float
            ),
            train_attack[
                "human_score"
            ].to_numpy(
                dtype=float
            ),
        ]
    )

    weights = np.concatenate(
        [
            v2b.sample_weights(
                train_clean,
                adversarial=False,
            ),
            v2b.sample_weights(
                train_attack,
                adversarial=True,
            ),
        ]
    )

    model = v2b.make_regressor()

    model.fit(
        X_train,
        y_train,
        sample_weight=weights,
    )

    return (
        model,
        train_clean,
    )


def v2b_predict_one(
    model,
    row: pd.Series,
    embedder: v2b.EmbeddingCache,
) -> float:
    features = v2b.make_features(
        question_text=row[
            "question_text"
        ],
        model_answer=row[
            "model_answer"
        ],
        student_answer=row[
            "student_answer"
        ],
        rubric_keywords=row[
            "rubric_keywords"
        ],
        embedder=embedder,
    )

    X = pd.DataFrame(
        [features],
        columns=v2b.FEATURE_NAMES,
    )

    return float(
        np.clip(
            model.predict(
                X
            )[0],
            0.0,
            10.0,
        )
    )


# ---------------------------------------------------------------------------
# V2E prediction
# ---------------------------------------------------------------------------

def v2e_predict_one(
    base_model,
    coverage_guard: IsotonicRegression,
    row: pd.Series,
    rubric_concepts: dict[str, dict[str, object]],
    embedder: v2b.EmbeddingCache,
    nli: v2c.NLICache,
) -> dict[str, object]:
    base_score = v2b_predict_one(
        model=base_model,
        row=row,
        embedder=embedder,
    )

    profile = extract_evidence_profile(
        row=row,
        rubric_concepts=rubric_concepts,
        embedder=embedder,
        nli=nli,
    )

    if profile.evidence_text:
        evidence_row = row.copy()
        evidence_row[
            "student_answer"
        ] = profile.evidence_text

        evidence_score = v2b_predict_one(
            model=base_model,
            row=evidence_row,
            embedder=embedder,
        )
    else:
        evidence_score = 0.0

    evidence_ceiling = float(
        np.clip(
            evidence_score
            + EVIDENCE_SCORE_SLACK,
            0.0,
            10.0,
        )
    )

    rubric_ceiling = coverage_ceiling(
        guard=coverage_guard,
        coverage_index=profile.coverage_index,
    )

    pre_contradiction = min(
        base_score,
        evidence_ceiling,
        rubric_ceiling,
    )

    penalty = contradiction_penalty(
        profile.max_contradiction
    )

    final_score = float(
        np.clip(
            pre_contradiction
            - penalty,
            0.0,
            10.0,
        )
    )

    return {
        "base_score": float(
            base_score
        ),
        "evidence_score": float(
            evidence_score
        ),
        "evidence_ceiling": float(
            evidence_ceiling
        ),
        "rubric_ceiling": float(
            rubric_ceiling
        ),
        "pre_contradiction_score": float(
            pre_contradiction
        ),
        "contradiction_penalty": float(
            penalty
        ),
        "max_contradiction": float(
            profile.max_contradiction
        ),
        "final_score": final_score,
        "coverage_index": float(
            profile.coverage_index
        ),
        "supported_count": int(
            profile.supported_count
        ),
        "evidence_word_ratio": float(
            profile.evidence_word_ratio
        ),
        "original_sentence_count": int(
            profile.original_sentence_count
        ),
        "selected_sentence_count": int(
            profile.selected_sentence_count
        ),
        "evidence_text": profile.evidence_text,
        "diagnostics": profile.diagnostics,
    }


# ---------------------------------------------------------------------------
# Attacks
# ---------------------------------------------------------------------------

def all_eval_attacks(
    row: pd.Series,
    df: pd.DataFrame,
    rubric_concepts: dict[str, dict[str, object]],
) -> dict[str, str]:
    return v2c.adversarial_variants(
        row=row,
        pool_df=df,
        rubric_concepts=rubric_concepts,
        include_contradiction=True,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_rows(
    evaluation_name: str,
    folds: list[
        tuple[
            set[int],
            pd.DataFrame,
        ]
    ],
    df: pd.DataFrame,
    clean_X: pd.DataFrame,
    attack_df: pd.DataFrame,
    attack_X: pd.DataFrame,
    clean_profiles: dict[
        int,
        EvidenceProfile,
    ],
    rubric_concepts: dict[str, dict[str, object]],
    embedder: v2b.EmbeddingCache,
    nli: v2c.NLICache,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    prediction_records: list[
        dict[str, object]
    ] = []

    attack_records: list[
        dict[str, object]
    ] = []

    diagnostic_records: list[
        dict[str, object]
    ] = []

    coverage_by_row_id = {
        row_id: profile.coverage_index
        for row_id, profile in clean_profiles.items()
    }

    for fold_num, (
        heldout_ids,
        test_df,
    ) in enumerate(
        folds,
        start=1,
    ):
        base_model, train_clean = train_v2b_fold(
            clean_df=df,
            clean_X=clean_X,
            attack_df=attack_df,
            attack_X=attack_X,
            heldout_ids=heldout_ids,
        )

        coverage_guard = fit_coverage_guard(
            train_df=train_clean,
            coverage_by_row_id=coverage_by_row_id,
        )

        for _, row in test_df.iterrows():
            result = v2e_predict_one(
                base_model=base_model,
                coverage_guard=coverage_guard,
                row=row,
                rubric_concepts=rubric_concepts,
                embedder=embedder,
                nli=nli,
            )

            truth = float(
                row["human_score"]
            )

            for model_name, prediction in [
                (
                    "V2B",
                    result["base_score"],
                ),
                (
                    "V2E",
                    result["final_score"],
                ),
            ]:
                prediction_records.append(
                    {
                        "evaluation": evaluation_name,
                        "fold": fold_num,
                        "model": model_name,
                        "_row_id": int(
                            row["_row_id"]
                        ),
                        "question_id": row[
                            "question_id"
                        ],
                        "source": row[
                            "source"
                        ],
                        "human_score": truth,
                        "prediction": float(
                            prediction
                        ),
                        "absolute_error": abs(
                            truth
                            - float(
                                prediction
                            )
                        ),
                        "evidence_score": result[
                            "evidence_score"
                        ],
                        "evidence_ceiling": result[
                            "evidence_ceiling"
                        ],
                        "rubric_ceiling": result[
                            "rubric_ceiling"
                        ],
                        "coverage_index": result[
                            "coverage_index"
                        ],
                        "supported_count": result[
                            "supported_count"
                        ],
                        "evidence_word_ratio": result[
                            "evidence_word_ratio"
                        ],
                        "max_contradiction": result[
                            "max_contradiction"
                        ],
                        "contradiction_penalty": result[
                            "contradiction_penalty"
                        ],
                        "student_answer": row[
                            "student_answer"
                        ],
                        "evidence_text": result[
                            "evidence_text"
                        ],
                    }
                )

            for diag in result[
                "diagnostics"
            ]:
                diagnostic_records.append(
                    {
                        "evaluation": evaluation_name,
                        "fold": fold_num,
                        "_row_id": int(
                            row["_row_id"]
                        ),
                        "human_score": truth,
                        "student_answer": row[
                            "student_answer"
                        ],
                        "evidence_text": result[
                            "evidence_text"
                        ],
                        **diag,
                    }
                )

            attacks = all_eval_attacks(
                row=row,
                df=df,
                rubric_concepts=rubric_concepts,
            )

            base_v2b = float(
                result["base_score"]
            )
            base_v2e = float(
                result["final_score"]
            )

            for (
                attack_name,
                attacked_answer,
            ) in attacks.items():
                attacked_row = row.copy()
                attacked_row[
                    "student_answer"
                ] = attacked_answer

                attacked = v2e_predict_one(
                    base_model=base_model,
                    coverage_guard=coverage_guard,
                    row=attacked_row,
                    rubric_concepts=rubric_concepts,
                    embedder=embedder,
                    nli=nli,
                )

                for (
                    model_name,
                    clean_prediction,
                    attacked_prediction,
                ) in [
                    (
                        "V2B",
                        base_v2b,
                        attacked[
                            "base_score"
                        ],
                    ),
                    (
                        "V2E",
                        base_v2e,
                        attacked[
                            "final_score"
                        ],
                    ),
                ]:
                    attack_records.append(
                        {
                            "evaluation": evaluation_name,
                            "fold": fold_num,
                            "model": model_name,
                            "_row_id": int(
                                row["_row_id"]
                            ),
                            "question_id": row[
                                "question_id"
                            ],
                            "human_score": truth,
                            "attack": attack_name,
                            "base_prediction": float(
                                clean_prediction
                            ),
                            "attacked_prediction": float(
                                attacked_prediction
                            ),
                            "score_delta": float(
                                attacked_prediction
                                - clean_prediction
                            ),
                            "attacked_evidence_score": attacked[
                                "evidence_score"
                            ],
                            "attacked_evidence_ceiling": attacked[
                                "evidence_ceiling"
                            ],
                            "attacked_rubric_ceiling": attacked[
                                "rubric_ceiling"
                            ],
                            "attacked_coverage_index": attacked[
                                "coverage_index"
                            ],
                            "attacked_supported_count": attacked[
                                "supported_count"
                            ],
                            "attacked_evidence_word_ratio": attacked[
                                "evidence_word_ratio"
                            ],
                            "attacked_max_contradiction": attacked[
                                "max_contradiction"
                            ],
                            "attacked_contradiction_penalty": attacked[
                                "contradiction_penalty"
                            ],
                            "base_answer": row[
                                "student_answer"
                            ],
                            "attacked_answer": attacked_answer,
                            "attacked_evidence_text": attacked[
                                "evidence_text"
                            ],
                        }
                    )

        print(
            f"{evaluation_name}: fold "
            f"{fold_num}/{len(folds)} complete "
            f"(held out={len(test_df)})"
        )

    return (
        pd.DataFrame(
            prediction_records
        ),
        pd.DataFrame(
            attack_records
        ),
        pd.DataFrame(
            diagnostic_records
        ),
    )


def print_adversarial(
    title: str,
    adv_df: pd.DataFrame,
    model_name: str,
) -> dict[str, float]:
    part = adv_df[
        adv_df["model"].eq(
            model_name
        )
    ].copy()

    padding = part[
        ~part["attack"].eq(
            "contradiction_injection"
        )
    ].copy()

    contradiction = part[
        part["attack"].eq(
            "contradiction_injection"
        )
    ].copy()

    positive = padding[
        "score_delta"
    ].clip(
        lower=0.0
    )

    print(
        f"\n## {title} — {model_name}"
    )

    print(
        f"Base answers:                 "
        f"{padding['_row_id'].nunique()}"
    )
    print(
        f"Padding mean delta:           "
        f"{padding['score_delta'].mean():+.3f}"
    )
    print(
        f"Padding mean positive delta:  "
        f"{positive.mean():+.3f}"
    )
    print(
        f"Padding worst increase:       "
        f"{padding['score_delta'].max():+.3f}"
    )
    print(
        f"Failures > +0.5 mark:         "
        f"{(padding['score_delta'] > 0.5).mean() * 100:.1f}%"
    )
    print(
        f"Failures > +1.0 mark:         "
        f"{(padding['score_delta'] > 1.0).mean() * 100:.1f}%"
    )

    print("\nBy padding attack:")
    print(
        padding.groupby(
            "attack"
        )["score_delta"]
        .agg(
            [
                "mean",
                "max",
                "count",
            ]
        )
        .sort_values(
            "mean",
            ascending=False,
        )
        .round(3)
        .to_string()
    )

    if not contradiction.empty:
        print(
            "\nContradiction injection:"
        )
        print(
            f"Mean delta:                   "
            f"{contradiction['score_delta'].mean():+.3f}"
        )
        print(
            f"Worst increase:               "
            f"{contradiction['score_delta'].max():+.3f}"
        )
        print(
            f"Raised score > +0.5:          "
            f"{(contradiction['score_delta'] > 0.5).mean() * 100:.1f}%"
        )

    return {
        "padding_mean_delta": float(
            padding[
                "score_delta"
            ].mean()
        ),
        "padding_mean_positive_delta": float(
            positive.mean()
        ),
        "padding_worst_increase": float(
            padding[
                "score_delta"
            ].max()
        ),
        "padding_fail_gt_0_5": float(
            np.mean(
                padding[
                    "score_delta"
                ] > 0.5
            )
        ),
        "padding_fail_gt_1_0": float(
            np.mean(
                padding[
                    "score_delta"
                ] > 1.0
            )
        ),
        "contradiction_mean_delta": float(
            contradiction[
                "score_delta"
            ].mean()
        )
        if not contradiction.empty
        else float("nan"),
        "contradiction_worst_increase": float(
            contradiction[
                "score_delta"
            ].max()
        )
        if not contradiction.empty
        else float("nan"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 80)
    print(
        "RESEARCH GRADER V2E — RUBRIC EVIDENCE BOTTLENECK"
    )
    print("=" * 80)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing prepared dataset: {DATA_PATH}"
        )

    if not RUBRIC_PATH.exists():
        raise FileNotFoundError(
            f"Missing rubric concepts: {RUBRIC_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH
    ).reset_index(
        drop=True
    )

    with RUBRIC_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        rubric_concepts = json.load(
            f
        )

    df["_row_id"] = np.arange(
        len(df),
        dtype=int,
    )

    df["human_score"] = pd.to_numeric(
        df["human_score"],
        errors="raise",
    ).astype(float)

    human = df[
        df["source"]
        .astype(str)
        .eq("human_group")
    ]

    weak = df[
        df["human_score"] <= 4.0
    ]

    print(f"Dataset rows:          {len(df)}")
    print(f"Human gold rows:       {len(human)}")
    print(
        f"Human score range:     "
        f"{human['human_score'].min():.1f}"
        f"–{human['human_score'].max():.1f}"
    )
    print(f"Weak rows <=4:         {len(weak)}")
    print(f"Base scorer:           V2B GradientBoosting")
    print(f"Embedding encoder:     {v2b.ENCODER_NAME}")
    print(f"NLI validator:         {v2c.NLI_MODEL_NAME}")
    print(
        "Raw answer length:     not added by V2E"
    )
    print(
        "Whole-answer NLI:      not used as score"
    )
    print(
        "Guard direction:       can lower V2B only"
    )
    print(
        f"Evidence-score slack:  +{EVIDENCE_SCORE_SLACK:.2f}"
    )

    # ---------------------------------------------------------------
    # Models
    # ---------------------------------------------------------------
    print(
        "\nLoading sentence encoder..."
    )

    embedder = v2b.EmbeddingCache(
        v2b.ENCODER_NAME
    )

    print(
        "\nLoading NLI validator..."
    )

    nli = v2c.NLICache(
        model_name=v2c.NLI_MODEL_NAME,
        device=v2c.NLI_DEVICE,
        batch_size=v2c.NLI_BATCH_SIZE,
    )

    # ---------------------------------------------------------------
    # Clean V2B features
    # ---------------------------------------------------------------
    clean_X = build_v2b_feature_frame(
        df=df,
        embedder=embedder,
        label="V2B clean feature extraction",
    )

    # ---------------------------------------------------------------
    # Precompute CLEAN evidence profiles once.
    # Fold training only needs the clean coverage values; test attacks are
    # still extracted fold-by-fold with the same deterministic extractor.
    # ---------------------------------------------------------------
    print(
        "\nExtracting clean rubric evidence profiles..."
    )

    clean_profiles: dict[
        int,
        EvidenceProfile,
    ] = {}

    for i, (_, row) in enumerate(
        df.iterrows(),
        start=1,
    ):
        clean_profiles[
            int(
                row["_row_id"]
            )
        ] = extract_evidence_profile(
            row=row,
            rubric_concepts=rubric_concepts,
            embedder=embedder,
            nli=nli,
        )

        if (
            i == 1
            or i % 25 == 0
            or i == len(df)
        ):
            print(
                f"Clean evidence extraction: "
                f"{i}/{len(df)}"
            )

    # ---------------------------------------------------------------
    # V2B invariance training attacks
    # ---------------------------------------------------------------
    print(
        "\nBuilding V2B invariance attacks..."
    )

    attack_df = build_v2b_attack_frame(
        df
    )

    attack_X = build_v2b_feature_frame(
        df=attack_df,
        embedder=embedder,
        label="V2B attack feature extraction",
    )

    # ---------------------------------------------------------------
    # Human OOF
    # ---------------------------------------------------------------
    human_fold_series = assign_human_folds(
        df
    )

    human_folds: list[
        tuple[
            set[int],
            pd.DataFrame,
        ]
    ] = []

    for fold_id in range(4):
        test_df = df[
            human_fold_series.eq(
                fold_id
            )
        ].copy()

        heldout = set(
            test_df[
                "_row_id"
            ].astype(
                int
            ).tolist()
        )

        human_folds.append(
            (
                heldout,
                test_df,
            )
        )

    (
        human_pred,
        human_adv,
        human_diag,
    ) = evaluate_rows(
        evaluation_name="HUMAN_OOF",
        folds=human_folds,
        df=df,
        clean_X=clean_X,
        attack_df=attack_df,
        attack_X=attack_X,
        clean_profiles=clean_profiles,
        rubric_concepts=rubric_concepts,
        embedder=embedder,
        nli=nli,
    )

    # ---------------------------------------------------------------
    # Weak OOF
    # ---------------------------------------------------------------
    weak_indices = weak.index.to_numpy()

    weak_strata = weak[
        "question_id"
    ].astype(
        str
    ).to_numpy()

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    weak_folds: list[
        tuple[
            set[int],
            pd.DataFrame,
        ]
    ] = []

    for _, test_pos in skf.split(
        weak_indices,
        weak_strata,
    ):
        test_indices = weak_indices[
            test_pos
        ]

        test_df = df.loc[
            test_indices
        ].copy()

        heldout = set(
            test_df[
                "_row_id"
            ].astype(
                int
            ).tolist()
        )

        weak_folds.append(
            (
                heldout,
                test_df,
            )
        )

    (
        weak_pred,
        weak_adv,
        weak_diag,
    ) = evaluate_rows(
        evaluation_name="WEAK_OOF",
        folds=weak_folds,
        df=df,
        clean_X=clean_X,
        attack_df=attack_df,
        attack_X=attack_X,
        clean_profiles=clean_profiles,
        rubric_concepts=rubric_concepts,
        embedder=embedder,
        nli=nli,
    )

    # ---------------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------------
    summary_records: list[
        dict[str, object]
    ] = []

    for (
        evaluation_name,
        pred_df,
        adv_df,
    ) in [
        (
            "HUMAN_OOF",
            human_pred,
            human_adv,
        ),
        (
            "WEAK_OOF",
            weak_pred,
            weak_adv,
        ),
    ]:
        print(
            "\n"
            + "=" * 72
            + f"\n{evaluation_name}\n"
            + "=" * 72
        )

        for model_name in [
            "V2B",
            "V2E",
        ]:
            part = pred_df[
                pred_df[
                    "model"
                ].eq(
                    model_name
                )
            ]

            y_true = part[
                "human_score"
            ].to_numpy(
                dtype=float
            )

            y_pred = part[
                "prediction"
            ].to_numpy(
                dtype=float
            )

            m = print_metrics(
                f"{evaluation_name} — {model_name}",
                y_true,
                y_pred,
            )

            a = print_adversarial(
                title=(
                    f"{evaluation_name} ADVERSARIAL"
                ),
                adv_df=adv_df,
                model_name=model_name,
            )

            summary_records.append(
                {
                    "evaluation": evaluation_name,
                    "model": model_name,
                    **m,
                    **a,
                }
            )

    # ---------------------------------------------------------------
    # Guard diagnostics
    # ---------------------------------------------------------------
    for name, pred_df in [
        (
            "HUMAN_OOF",
            human_pred,
        ),
        (
            "WEAK_OOF",
            weak_pred,
        ),
    ]:
        v2e_rows = pred_df[
            pred_df["model"].eq(
                "V2E"
            )
        ].copy()

        print(
            f"\n## V2E evidence bottleneck — {name}"
        )

        base_reduction = (
            v2e_rows["evidence_ceiling"]
            < (
                v2e_rows[
                    "prediction"
                ]
                + 999.0
            )
        )

        # The useful intervention metric is computed from the paired V2B/V2E
        # prediction difference.
        paired = (
            pred_df.pivot_table(
                index="_row_id",
                columns="model",
                values="prediction",
                aggfunc="first",
            )
            .dropna()
        )

        if {
            "V2B",
            "V2E",
        }.issubset(
            paired.columns
        ):
            reduction = (
                paired["V2B"]
                - paired["V2E"]
            )

            print(
                f"Guard changed score:      "
                f"{(reduction > 1e-6).mean() * 100:.1f}%"
            )
            print(
                f"Mean score reduction:     "
                f"{reduction.clip(lower=0.0).mean():.3f}"
            )
            print(
                f"Max score reduction:      "
                f"{reduction.max():.3f}"
            )

        print(
            f"Mean evidence word ratio: "
            f"{v2e_rows['evidence_word_ratio'].mean():.3f}"
        )
        print(
            f"Mean supported concepts:  "
            f"{v2e_rows['supported_count'].mean():.2f}/6"
        )
        print(
            f"Contradiction penalty on: "
            f"{(v2e_rows['contradiction_penalty'] > 1e-6).mean() * 100:.1f}%"
        )

    # ---------------------------------------------------------------
    # Final artifact
    # ---------------------------------------------------------------
    print(
        "\nTraining final V2E base artifact "
        "on all repaired data + V2B invariance attacks..."
    )

    final_base_model, final_clean = train_v2b_fold(
        clean_df=df,
        clean_X=clean_X,
        attack_df=attack_df,
        attack_X=attack_X,
        heldout_ids=set(),
    )

    coverage_by_row_id = {
        row_id: profile.coverage_index
        for row_id, profile in clean_profiles.items()
    }

    final_coverage_guard = fit_coverage_guard(
        train_df=final_clean,
        coverage_by_row_id=coverage_by_row_id,
    )

    artifact = {
        "version": "research_grader_v2e",
        "architecture": (
            "V2B base + sentence-level atomic-rubric evidence bottleneck "
            "+ evidence-only score ceiling + NLI contradiction penalty"
        ),
        "base_model": final_base_model,
        "base_feature_names": v2b.FEATURE_NAMES,
        "embedding_model_name": v2b.ENCODER_NAME,
        "nli_model_name": v2c.NLI_MODEL_NAME,
        "rubric_concepts": rubric_concepts,
        "coverage_guard": final_coverage_guard,
        "parameters": {
            "semantic_low": SEMANTIC_LOW,
            "semantic_high": SEMANTIC_HIGH,
            "min_semantic_support": MIN_SEMANTIC_SUPPORT,
            "validated_concept_support": VALIDATED_CONCEPT_SUPPORT,
            "evidence_sentence_threshold": EVIDENCE_SENTENCE_THRESHOLD,
            "question_copy_start": QUESTION_COPY_START,
            "evidence_score_slack": EVIDENCE_SCORE_SLACK,
            "coverage_cap_slack": COVERAGE_CAP_SLACK,
            "min_coverage_cap": MIN_COVERAGE_CAP,
            "contradiction_start": CONTRADICTION_START,
            "max_contradiction_penalty": MAX_CONTRADICTION_PENALTY,
            "nli_entailment_floor": NLI_ENTAILMENT_FLOOR,
            "evidence_dedup_cosine": EVIDENCE_DEDUP_COSINE,
        },
        "training_rows": int(
            len(df)
        ),
        "human_rows": int(
            len(human)
        ),
        "weak_rows_le_4": int(
            len(weak)
        ),
        "design_properties": {
            "guard_can_raise_base_score": False,
            "raw_length_added_by_v2e": False,
            "whole_answer_nli_direct_score": False,
            "padding_removed_before_guard_score": True,
            "duplicate_evidence_collapsed": True,
            "question_copy_downweighted": True,
        },
    }

    ARTIFACT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        artifact,
        ARTIFACT_PATH,
    )

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------
    human_pred.to_csv(
        HUMAN_OOF_PATH,
        index=False,
    )

    weak_pred.to_csv(
        WEAK_OOF_PATH,
        index=False,
    )

    combined_adv = pd.concat(
        [
            human_adv,
            weak_adv,
        ],
        ignore_index=True,
    )

    combined_adv.to_csv(
        ADVERSARIAL_PATH,
        index=False,
    )

    summary_df = pd.DataFrame(
        summary_records
    )

    summary_df.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    diagnostics_df = pd.concat(
        [
            human_diag,
            weak_diag,
        ],
        ignore_index=True,
    )

    diagnostics_df.to_csv(
        EVIDENCE_DIAGNOSTICS_PATH,
        index=False,
    )

    print(
        "\n## Side-by-side summary"
    )

    print(
        summary_df[
            [
                "evaluation",
                "model",
                "mae",
                "rmse",
                "r2",
                "within_1_mark",
                "within_2_marks",
                "padding_mean_positive_delta",
                "padding_worst_increase",
                "padding_fail_gt_0_5",
                "padding_fail_gt_1_0",
                "contradiction_mean_delta",
                "contradiction_worst_increase",
            ]
        ]
        .round(3)
        .to_string(
            index=False
        )
    )

    print("\n## Saved")
    print(
        f"Artifact:      {ARTIFACT_PATH}"
    )
    print(
        f"Human OOF:     {HUMAN_OOF_PATH}"
    )
    print(
        f"Weak OOF:      {WEAK_OOF_PATH}"
    )
    print(
        f"Adversarial:   {ADVERSARIAL_PATH}"
    )
    print(
        f"Summary:       {SUMMARY_PATH}"
    )
    print(
        f"Evidence diag: {EVIDENCE_DIAGNOSTICS_PATH}"
    )

    print(
        "\nV1, V2A, V2B, V2C, V2D, and the original dataset "
        "were not modified."
    )


if __name__ == "__main__":
    main()
