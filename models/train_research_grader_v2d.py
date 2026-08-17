"""Research Grader V2D — hybrid base scorer + rubric guard.

Core idea
---------
Use the component that currently works best for each job:

    V2B GradientBoosting
        -> primary score prediction

    Atomic rubric semantic coverage
        -> score CEILING only
           (the guard can lower an implausibly high score, never raise it)

    V2C NLI contradiction detector
        -> contradiction PENALTY only
           (NLI is not used as the main scoring engine)

Why
---
The fair 20-human-answer benchmark showed:
- V2B human OOF MAE: 1.453
- V2C human OOF MAE: 1.951
- V2B was highly stable to padding on high-scoring human answers.
- V2C reacted much more strongly to contradiction injection.

V2D therefore avoids replacing V2B with NLI. Instead it treats the rubric
and NLI components as safety/robustness guards around the stronger base score.

Evaluation
----------
A. HUMAN GOLD:
   4 folds, one held-out human answer per question per fold.
   All 20 human_group answers receive an OOF prediction.

B. LOW-SCORE STRESS SET:
   All rows with score <= 4 are evaluated out-of-fold in 5 folds,
   stratified by question.
   Importantly, attacks derived from a held-out weak answer are excluded
   from that fold's training data.

For both evaluations we compare:
    V2B base
    V2D hybrid

Attacks:
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
models/artifacts/research_grader_v2d.joblib
data/research_grader_v2d_human_oof.csv
data/research_grader_v2d_weak_oof.csv
data/research_grader_v2d_adversarial.csv
data/research_grader_v2d_summary.csv

Nothing from V1/V2A/V2B/V2C is overwritten.
"""

from __future__ import annotations

import json
import math
import sys
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


DATA_PATH = ROOT / "data" / "training_dataset_v2c.csv"
RUBRIC_PATH = ROOT / "data" / "rubric_concepts_v2c.json"

ARTIFACT_PATH = (
    ROOT / "models" / "artifacts" / "research_grader_v2d.joblib"
)
HUMAN_OOF_PATH = (
    ROOT / "data" / "research_grader_v2d_human_oof.csv"
)
WEAK_OOF_PATH = (
    ROOT / "data" / "research_grader_v2d_weak_oof.csv"
)
ADVERSARIAL_PATH = (
    ROOT / "data" / "research_grader_v2d_adversarial.csv"
)
SUMMARY_PATH = (
    ROOT / "data" / "research_grader_v2d_summary.csv"
)

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Rubric guard constants
# ---------------------------------------------------------------------------

# Map cosine similarity to a 0..1 semantic-support scale.
SEMANTIC_LOW = 0.34
SEMANTIC_HIGH = 0.68

# Hard concept-support threshold on the normalized 0..1 scale.
HARD_SUPPORT = 0.55

# The isotonic model predicts the expected score from rubric coverage.
# The guard receives this much upward slack before it can cap V2B.
#
# This makes the guard deliberately conservative: it should only intervene
# when V2B is substantially above what rubric evidence supports.
COVERAGE_CAP_SLACK = 1.25

# Do not let the rubric ceiling fall below this value. Very weak answers are
# still handled primarily by the base model; the guard is not meant to invent
# an absolute 0/10 scale by itself.
MIN_COVERAGE_CAP = 2.0

# Generic NLI is useful as a contradiction detector but was a poor direct
# scorer in V2C. Therefore V2D applies no penalty until contradiction
# probability is high.
CONTRADICTION_START = 0.65
MAX_CONTRADICTION_PENALTY = 2.0


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def clean_text(value: object) -> str:
    return v2c.clean_text(value)


def metric_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    result = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(
            math.sqrt(mean_squared_error(y_true, y_pred))
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
) -> dict[str, float]:
    m = metric_dict(y_true, y_pred)

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


def human_folds(df: pd.DataFrame) -> pd.Series:
    """Same four-fold human protocol used by V2C/fair benchmark."""
    fold = pd.Series(-1, index=df.index, dtype=int)

    human = df[
        df["source"].astype(str).eq("human_group")
    ]

    counts = human.groupby("question_id").size()

    if not (counts == 4).all():
        raise RuntimeError(
            "Expected exactly four human_group rows per question. "
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
# Atomic rubric semantic guard
# ---------------------------------------------------------------------------

def normalized_semantic_support(cosine: np.ndarray) -> np.ndarray:
    denom = SEMANTIC_HIGH - SEMANTIC_LOW

    if denom <= 0:
        raise RuntimeError("Invalid semantic support thresholds.")

    return np.clip(
        (cosine - SEMANTIC_LOW) / denom,
        0.0,
        1.0,
    )


def atomic_rubric_semantic_features(
    row: pd.Series,
    rubric_concepts: dict[str, dict[str, object]],
    embedder: v2b.EmbeddingCache,
) -> dict[str, float]:
    qid = str(row["question_id"])

    if qid not in rubric_concepts:
        raise KeyError(f"Missing rubric concepts for {qid}")

    criteria = [
        clean_text(x)
        for x in rubric_concepts[qid]["criteria"]
        if clean_text(x)
    ]

    answer = clean_text(row["student_answer"])
    sentences = v2b.split_sentences(answer)

    if not criteria or not sentences:
        return {
            "atomic_semantic_mean": 0.0,
            "atomic_semantic_min": 0.0,
            "atomic_hard_coverage": 0.0,
            "atomic_supported_count": 0.0,
            "atomic_coverage_index": 0.0,
        }

    criterion_vecs = embedder.encode(criteria)
    sentence_vecs = embedder.encode(sentences)

    similarity = criterion_vecs @ sentence_vecs.T
    best_cosine = similarity.max(axis=1)

    support = normalized_semantic_support(best_cosine)

    semantic_mean = float(support.mean())
    semantic_min = float(support.min())

    hard_coverage = float(
        np.mean(support >= HARD_SUPPORT)
    )

    supported_count = float(
        np.sum(support >= HARD_SUPPORT)
    )

    # Mean support is smooth; hard coverage resists the case where every
    # criterion is only weakly similar. Both are rubric-based, not length-based.
    coverage_index = float(
        0.70 * semantic_mean
        + 0.30 * hard_coverage
    )

    return {
        "atomic_semantic_mean": semantic_mean,
        "atomic_semantic_min": semantic_min,
        "atomic_hard_coverage": hard_coverage,
        "atomic_supported_count": supported_count,
        "atomic_coverage_index": coverage_index,
    }


def build_atomic_frame(
    df: pd.DataFrame,
    rubric_concepts: dict[str, dict[str, object]],
    embedder: v2b.EmbeddingCache,
    label: str,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    total = len(df)

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        rows.append(
            atomic_rubric_semantic_features(
                row=row,
                rubric_concepts=rubric_concepts,
                embedder=embedder,
            )
        )

        if i == 1 or i % 50 == 0 or i == total:
            print(f"{label}: {i}/{total}")

    return pd.DataFrame(rows, index=df.index)


def fit_coverage_guard(
    train_df: pd.DataFrame,
    atomic_X: pd.DataFrame,
) -> IsotonicRegression:
    x = atomic_X.loc[
        train_df.index,
        "atomic_coverage_index",
    ].to_numpy(dtype=float)

    y = train_df[
        "human_score"
    ].to_numpy(dtype=float)

    # Same preference used by V2B: real human labels carry more weight.
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
            np.asarray([coverage_index], dtype=float)
        )[0]
    )

    return float(
        np.clip(
            max(
                MIN_COVERAGE_CAP,
                expected + COVERAGE_CAP_SLACK,
            ),
            0.0,
            10.0,
        )
    )


# ---------------------------------------------------------------------------
# Contradiction guard
# ---------------------------------------------------------------------------

def max_rubric_contradiction(
    row: pd.Series,
    rubric_concepts: dict[str, dict[str, object]],
    nli: v2c.NLICache,
) -> float:
    """Highest contradiction probability against any atomic rubric criterion."""
    qid = str(row["question_id"])

    criteria = [
        clean_text(x)
        for x in rubric_concepts[qid]["criteria"]
        if clean_text(x)
    ]

    sentences = v2c.split_sentences(
        clean_text(row["student_answer"])
    )

    pairs = [
        (sentence, criterion)
        for criterion in criteria
        for sentence in sentences
    ]

    if not pairs:
        return 0.0

    probs = nli.probabilities(pairs)

    if probs.size == 0:
        return 0.0

    contradiction = probs[:, v2c.CONTRADICTION_INDEX]
    return float(np.max(contradiction))


def contradiction_penalty(
    max_contradiction: float,
) -> float:
    if max_contradiction <= CONTRADICTION_START:
        return 0.0

    scaled = (
        max_contradiction - CONTRADICTION_START
    ) / max(1e-9, 1.0 - CONTRADICTION_START)

    return float(
        MAX_CONTRADICTION_PENALTY
        * np.clip(scaled, 0.0, 1.0)
    )


# ---------------------------------------------------------------------------
# Hybrid scoring
# ---------------------------------------------------------------------------

def v2b_predict_one(
    model,
    row: pd.Series,
    embedder: v2b.EmbeddingCache,
) -> tuple[float, dict[str, float]]:
    features = v2b.make_features(
        question_text=row["question_text"],
        model_answer=row["model_answer"],
        student_answer=row["student_answer"],
        rubric_keywords=row["rubric_keywords"],
        embedder=embedder,
    )

    X = pd.DataFrame(
        [features],
        columns=v2b.FEATURE_NAMES,
    )

    pred = float(
        np.clip(
            model.predict(X)[0],
            0.0,
            10.0,
        )
    )

    return pred, features


def v2d_predict_one(
    base_model,
    coverage_guard: IsotonicRegression,
    row: pd.Series,
    rubric_concepts: dict[str, dict[str, object]],
    embedder: v2b.EmbeddingCache,
    nli: v2c.NLICache,
) -> dict[str, float]:
    base_score, base_features = v2b_predict_one(
        model=base_model,
        row=row,
        embedder=embedder,
    )

    atomic = atomic_rubric_semantic_features(
        row=row,
        rubric_concepts=rubric_concepts,
        embedder=embedder,
    )

    cap = coverage_ceiling(
        coverage_guard,
        atomic["atomic_coverage_index"],
    )

    capped_score = min(
        base_score,
        cap,
    )

    max_contra = max_rubric_contradiction(
        row=row,
        rubric_concepts=rubric_concepts,
        nli=nli,
    )

    contra_penalty = contradiction_penalty(
        max_contra
    )

    final_score = float(
        np.clip(
            capped_score - contra_penalty,
            0.0,
            10.0,
        )
    )

    return {
        "base_score": base_score,
        "coverage_cap": cap,
        "coverage_cap_reduction": (
            base_score - capped_score
        ),
        "max_contradiction": max_contra,
        "contradiction_penalty": contra_penalty,
        "final_score": final_score,
        **atomic,
        # Useful V2B diagnostics for later error analysis.
        "v2b_concept_soft_coverage": float(
            base_features["concept_soft_coverage"]
        ),
        "v2b_evidence_density": float(
            base_features["evidence_density"]
        ),
        "v2b_unsupported_word_ratio": float(
            base_features["unsupported_word_ratio"]
        ),
        "v2b_keyword_stuffing_score": float(
            base_features["keyword_stuffing_score"]
        ),
    }


# ---------------------------------------------------------------------------
# V2B training helpers
# ---------------------------------------------------------------------------

def build_v2b_attack_frame(
    df: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for _, row in df.iterrows():
        variants = v2b.adversarial_variants(
            row,
            df,
        )

        for attack_name, attacked_answer in variants.items():
            record = row.to_dict()
            record["student_answer"] = attacked_answer
            record["_attack"] = attack_name
            record["_parent_row_id"] = int(
                row["_row_id"]
            )
            records.append(record)

    return pd.DataFrame(records).reset_index(
        drop=True
    )


def build_v2b_feature_frame(
    df: pd.DataFrame,
    embedder: v2b.EmbeddingCache,
    label: str,
) -> pd.DataFrame:
    X = v2b.build_feature_frame(
        df.reset_index(drop=True),
        embedder=embedder,
        show_progress=True,
        label=label,
    )
    X.index = df.index
    return X


def train_v2b_fold(
    clean_df: pd.DataFrame,
    clean_X: pd.DataFrame,
    attack_df: pd.DataFrame,
    attack_X: pd.DataFrame,
    heldout_ids: set[int],
):
    clean_mask = ~clean_df[
        "_row_id"
    ].isin(heldout_ids)

    train_clean = clean_df.loc[
        clean_mask
    ]
    X_clean = clean_X.loc[
        train_clean.index
    ]

    attack_mask = ~attack_df[
        "_parent_row_id"
    ].isin(heldout_ids)

    train_attack = attack_df.loc[
        attack_mask
    ]
    X_attack = attack_X.loc[
        train_attack.index
    ]

    X_train = pd.concat(
        [
            X_clean.reset_index(drop=True),
            X_attack.reset_index(drop=True),
        ],
        ignore_index=True,
    )

    y_train = np.concatenate(
        [
            train_clean[
                "human_score"
            ].to_numpy(dtype=float),
            train_attack[
                "human_score"
            ].to_numpy(dtype=float),
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

    return model, train_clean


# ---------------------------------------------------------------------------
# Attack generation
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
    folds: list[tuple[set[int], pd.DataFrame]],
    df: pd.DataFrame,
    clean_X: pd.DataFrame,
    attack_df: pd.DataFrame,
    attack_X: pd.DataFrame,
    atomic_X: pd.DataFrame,
    rubric_concepts: dict[str, dict[str, object]],
    embedder: v2b.EmbeddingCache,
    nli: v2c.NLICache,
    run_attacks: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_records: list[dict[str, object]] = []
    attack_records: list[dict[str, object]] = []

    for fold_number, (heldout_ids, test_df) in enumerate(
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

        guard = fit_coverage_guard(
            train_df=train_clean,
            atomic_X=atomic_X,
        )

        for _, row in test_df.iterrows():
            result = v2d_predict_one(
                base_model=base_model,
                coverage_guard=guard,
                row=row,
                rubric_concepts=rubric_concepts,
                embedder=embedder,
                nli=nli,
            )

            truth = float(row["human_score"])

            for model_name, prediction in [
                ("V2B", result["base_score"]),
                ("V2D", result["final_score"]),
            ]:
                prediction_records.append(
                    {
                        "evaluation": evaluation_name,
                        "fold": fold_number,
                        "model": model_name,
                        "_row_id": int(
                            row["_row_id"]
                        ),
                        "question_id": row[
                            "question_id"
                        ],
                        "source": row["source"],
                        "human_score": truth,
                        "prediction": prediction,
                        "absolute_error": abs(
                            truth - prediction
                        ),
                        "coverage_cap": result[
                            "coverage_cap"
                        ],
                        "coverage_cap_reduction": result[
                            "coverage_cap_reduction"
                        ],
                        "atomic_coverage_index": result[
                            "atomic_coverage_index"
                        ],
                        "atomic_supported_count": result[
                            "atomic_supported_count"
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
                    }
                )

            if not run_attacks:
                continue

            attacks = all_eval_attacks(
                row=row,
                df=df,
                rubric_concepts=rubric_concepts,
            )

            base_v2b = result["base_score"]
            base_v2d = result["final_score"]

            for attack_name, attacked_answer in attacks.items():
                attacked_row = row.copy()
                attacked_row[
                    "student_answer"
                ] = attacked_answer

                attacked_result = v2d_predict_one(
                    base_model=base_model,
                    coverage_guard=guard,
                    row=attacked_row,
                    rubric_concepts=rubric_concepts,
                    embedder=embedder,
                    nli=nli,
                )

                for model_name, clean_pred, attacked_pred in [
                    (
                        "V2B",
                        base_v2b,
                        attacked_result["base_score"],
                    ),
                    (
                        "V2D",
                        base_v2d,
                        attacked_result["final_score"],
                    ),
                ]:
                    attack_records.append(
                        {
                            "evaluation": evaluation_name,
                            "fold": fold_number,
                            "model": model_name,
                            "_row_id": int(
                                row["_row_id"]
                            ),
                            "question_id": row[
                                "question_id"
                            ],
                            "human_score": truth,
                            "attack": attack_name,
                            "base_prediction": clean_pred,
                            "attacked_prediction": attacked_pred,
                            "score_delta": (
                                attacked_pred
                                - clean_pred
                            ),
                            "coverage_cap": attacked_result[
                                "coverage_cap"
                            ],
                            "coverage_cap_reduction": (
                                attacked_result[
                                    "coverage_cap_reduction"
                                ]
                            ),
                            "atomic_coverage_index": (
                                attacked_result[
                                    "atomic_coverage_index"
                                ]
                            ),
                            "max_contradiction": (
                                attacked_result[
                                    "max_contradiction"
                                ]
                            ),
                            "contradiction_penalty": (
                                attacked_result[
                                    "contradiction_penalty"
                                ]
                            ),
                            "base_answer": row[
                                "student_answer"
                            ],
                            "attacked_answer": attacked_answer,
                        }
                    )

        print(
            f"{evaluation_name}: fold "
            f"{fold_number}/{len(folds)} complete "
            f"(held out={len(test_df)})"
        )

    return (
        pd.DataFrame(prediction_records),
        pd.DataFrame(attack_records),
    )


def print_adversarial(
    title: str,
    adv_df: pd.DataFrame,
    model_name: str,
) -> dict[str, float]:
    part = adv_df[
        adv_df["model"].eq(model_name)
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
    ].clip(lower=0.0)

    print(f"\n## {title} — {model_name}")
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
        padding.groupby("attack")[
            "score_delta"
        ]
        .agg(["mean", "max", "count"])
        .sort_values(
            "mean",
            ascending=False,
        )
        .round(3)
        .to_string()
    )

    if not contradiction.empty:
        print("\nContradiction injection:")
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
            padding["score_delta"].mean()
        ),
        "padding_mean_positive_delta": float(
            positive.mean()
        ),
        "padding_worst_increase": float(
            padding["score_delta"].max()
        ),
        "padding_fail_gt_0_5": float(
            np.mean(
                padding["score_delta"] > 0.5
            )
        ),
        "padding_fail_gt_1_0": float(
            np.mean(
                padding["score_delta"] > 1.0
            )
        ),
        "contradiction_mean_delta": float(
            contradiction["score_delta"].mean()
        ) if not contradiction.empty else float("nan"),
        "contradiction_worst_increase": float(
            contradiction["score_delta"].max()
        ) if not contradiction.empty else float("nan"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 80)
    print(
        "RESEARCH GRADER V2D — V2B BASE + RUBRIC CEILING + NLI GUARD"
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
    ).reset_index(drop=True)

    with RUBRIC_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        rubric_concepts = json.load(f)

    df["_row_id"] = np.arange(
        len(df),
        dtype=int,
    )
    df["human_score"] = pd.to_numeric(
        df["human_score"],
        errors="raise",
    ).astype(float)

    human = df[
        df["source"].astype(str).eq(
            "human_group"
        )
    ]

    weak = df[
        df["human_score"] <= 4.0
    ]

    print(f"Dataset rows:       {len(df)}")
    print(f"Human gold rows:    {len(human)}")
    print(
        f"Human score range:  "
        f"{human['human_score'].min():.1f}"
        f"–{human['human_score'].max():.1f}"
    )
    print(f"Weak rows <=4:      {len(weak)}")
    print(f"V2B encoder:        {v2b.ENCODER_NAME}")
    print(f"NLI guard:          {v2c.NLI_MODEL_NAME}")
    print(
        "Hybrid rule:        final = min(V2B, rubric ceiling) "
        "- high-confidence contradiction penalty"
    )
    print(
        f"Coverage slack:      +{COVERAGE_CAP_SLACK:.2f} marks"
    )
    print(
        f"Contradiction starts at p={CONTRADICTION_START:.2f}"
    )

    # ---------------------------------------------------------------
    # Shared feature engines
    # ---------------------------------------------------------------
    print("\nLoading V2B sentence encoder...")
    embedder = v2b.EmbeddingCache(
        v2b.ENCODER_NAME
    )

    print("\nLoading NLI contradiction encoder...")
    nli = v2c.NLICache(
        model_name=v2c.NLI_MODEL_NAME,
        device=v2c.NLI_DEVICE,
        batch_size=v2c.NLI_BATCH_SIZE,
    )

    # ---------------------------------------------------------------
    # Precompute clean V2B and atomic rubric features
    # ---------------------------------------------------------------
    clean_X = build_v2b_feature_frame(
        df=df,
        embedder=embedder,
        label="V2B clean feature extraction",
    )

    atomic_X = build_atomic_frame(
        df=df,
        rubric_concepts=rubric_concepts,
        embedder=embedder,
        label="Atomic rubric extraction",
    )

    # ---------------------------------------------------------------
    # Precompute V2B training attacks
    # ---------------------------------------------------------------
    print("\nBuilding V2B invariance attacks...")
    attack_df = build_v2b_attack_frame(
        df
    )

    attack_X = build_v2b_feature_frame(
        df=attack_df,
        embedder=embedder,
        label="V2B attack feature extraction",
    )

    # ---------------------------------------------------------------
    # A. Human OOF evaluation
    # ---------------------------------------------------------------
    hfold_series = human_folds(df)
    human_eval_folds: list[
        tuple[set[int], pd.DataFrame]
    ] = []

    for fold_id in range(4):
        test_df = df[
            hfold_series.eq(fold_id)
        ].copy()

        heldout = set(
            test_df[
                "_row_id"
            ].astype(int).tolist()
        )

        human_eval_folds.append(
            (heldout, test_df)
        )

    human_pred, human_adv = evaluate_rows(
        evaluation_name="HUMAN_OOF",
        folds=human_eval_folds,
        df=df,
        clean_X=clean_X,
        attack_df=attack_df,
        attack_X=attack_X,
        atomic_X=atomic_X,
        rubric_concepts=rubric_concepts,
        embedder=embedder,
        nli=nli,
        run_attacks=True,
    )

    # ---------------------------------------------------------------
    # B. Weak <=4 OOF evaluation
    # ---------------------------------------------------------------
    weak_indices = weak.index.to_numpy()

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    weak_eval_folds: list[
        tuple[set[int], pd.DataFrame]
    ] = []

    weak_strata = weak[
        "question_id"
    ].astype(str).to_numpy()

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
            ].astype(int).tolist()
        )

        weak_eval_folds.append(
            (heldout, test_df)
        )

    weak_pred, weak_adv = evaluate_rows(
        evaluation_name="WEAK_OOF",
        folds=weak_eval_folds,
        df=df,
        clean_X=clean_X,
        attack_df=attack_df,
        attack_X=attack_X,
        atomic_X=atomic_X,
        rubric_concepts=rubric_concepts,
        embedder=embedder,
        nli=nli,
        run_attacks=True,
    )

    # ---------------------------------------------------------------
    # Reports
    # ---------------------------------------------------------------
    summary_records: list[
        dict[str, object]
    ] = []

    for eval_name, pred_df, adv_df in [
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
            + f"\n{eval_name}\n"
            + "=" * 72
        )

        for model_name in [
            "V2B",
            "V2D",
        ]:
            part = pred_df[
                pred_df["model"].eq(
                    model_name
                )
            ]

            y_true = part[
                "human_score"
            ].to_numpy(dtype=float)

            y_pred = part[
                "prediction"
            ].to_numpy(dtype=float)

            m = print_metrics(
                f"{eval_name} — {model_name}",
                y_true,
                y_pred,
            )

            a = print_adversarial(
                title=f"{eval_name} ADVERSARIAL",
                adv_df=adv_df,
                model_name=model_name,
            )

            summary_records.append(
                {
                    "evaluation": eval_name,
                    "model": model_name,
                    **m,
                    **a,
                }
            )

    # V2D guard intervention diagnostics.
    print("\n## V2D guard intervention — HUMAN OOF")
    human_v2d = human_pred[
        human_pred["model"].eq("V2D")
    ]

    print(
        f"Coverage cap activated: "
        f"{(human_v2d['coverage_cap_reduction'] > 1e-6).mean() * 100:.1f}%"
    )
    print(
        f"Mean cap reduction:     "
        f"{human_v2d['coverage_cap_reduction'].mean():.3f}"
    )
    print(
        f"Contradiction penalty activated: "
        f"{(human_v2d['contradiction_penalty'] > 1e-6).mean() * 100:.1f}%"
    )

    print("\n## V2D guard intervention — WEAK OOF")
    weak_v2d = weak_pred[
        weak_pred["model"].eq("V2D")
    ]

    print(
        f"Coverage cap activated: "
        f"{(weak_v2d['coverage_cap_reduction'] > 1e-6).mean() * 100:.1f}%"
    )
    print(
        f"Mean cap reduction:     "
        f"{weak_v2d['coverage_cap_reduction'].mean():.3f}"
    )
    print(
        f"Contradiction penalty activated: "
        f"{(weak_v2d['contradiction_penalty'] > 1e-6).mean() * 100:.1f}%"
    )

    # ---------------------------------------------------------------
    # Train final V2D artifact on all data
    # ---------------------------------------------------------------
    print(
        "\nTraining final V2D artifact on all repaired data "
        "+ V2B invariance attacks..."
    )

    all_ids: set[int] = set()

    final_base_model, final_clean = train_v2b_fold(
        clean_df=df,
        clean_X=clean_X,
        attack_df=attack_df,
        attack_X=attack_X,
        heldout_ids=all_ids,
    )

    final_guard = fit_coverage_guard(
        train_df=final_clean,
        atomic_X=atomic_X,
    )

    artifact = {
        "version": "research_grader_v2d",
        "architecture": (
            "V2B GradientBoosting base + atomic-rubric isotonic "
            "score ceiling + NLI contradiction guard"
        ),
        "base_model": final_base_model,
        "base_feature_names": v2b.FEATURE_NAMES,
        "embedding_model_name": v2b.ENCODER_NAME,
        "nli_model_name": v2c.NLI_MODEL_NAME,
        "rubric_concepts": rubric_concepts,
        "coverage_guard": final_guard,
        "guard_parameters": {
            "semantic_low": SEMANTIC_LOW,
            "semantic_high": SEMANTIC_HIGH,
            "hard_support": HARD_SUPPORT,
            "coverage_cap_slack": COVERAGE_CAP_SLACK,
            "min_coverage_cap": MIN_COVERAGE_CAP,
            "contradiction_start": CONTRADICTION_START,
            "max_contradiction_penalty": (
                MAX_CONTRADICTION_PENALTY
            ),
        },
        "training_rows": int(len(df)),
        "human_rows": int(len(human)),
        "weak_rows_le_4": int(len(weak)),
        "design_properties": {
            "guard_can_raise_base_score": False,
            "raw_length_added_by_v2d": False,
            "nli_used_as_direct_scorer": False,
            "rubric_guard_is_monotonic": True,
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

    combined_pred = pd.concat(
        [
            human_pred,
            weak_pred,
        ],
        ignore_index=True,
    )

    combined_adv = pd.concat(
        [
            human_adv,
            weak_adv,
        ],
        ignore_index=True,
    )

    summary_df = pd.DataFrame(
        summary_records
    )

    human_pred.to_csv(
        HUMAN_OOF_PATH,
        index=False,
    )

    weak_pred.to_csv(
        WEAK_OOF_PATH,
        index=False,
    )

    combined_adv.to_csv(
        ADVERSARIAL_PATH,
        index=False,
    )

    summary_df.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print("\n## Side-by-side summary")
    display = summary_df[
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
    ].copy()

    print(
        display.round(3).to_string(
            index=False
        )
    )

    print("\n## Saved")
    print(f"Artifact:     {ARTIFACT_PATH}")
    print(f"Human OOF:    {HUMAN_OOF_PATH}")
    print(f"Weak OOF:     {WEAK_OOF_PATH}")
    print(f"Adversarial:  {ADVERSARIAL_PATH}")
    print(f"Summary:      {SUMMARY_PATH}")

    print(
        "\nV1, V2A, V2B, V2C, and the original dataset "
        "were not modified."
    )


if __name__ == "__main__":
    main()
