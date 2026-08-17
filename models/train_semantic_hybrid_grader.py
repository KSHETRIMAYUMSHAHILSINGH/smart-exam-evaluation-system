"""
Train the project's semantic-hybrid grading model.

This version:

1. Cleans conflicting exact duplicate answers.
2. Gives greater weight to real human-scored answers.
3. Uses leave-one-question-out cross-validation.
4. Prevents same-question template leakage.
5. Uses semantic, rubric, lexical, and adequacy features.
6. Replaces raw answer-length reward with 70% length adequacy.
7. Compares multiple regressors before selecting the production model.

Run:

    python models/train_semantic_hybrid_grader.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    cohen_kappa_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# =========================================================
# PATHS
# =========================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT
    / "data"
    / "training_dataset.csv"
)

CLEANED_DATA_PATH = (
    ROOT
    / "data"
    / "training_dataset_cleaned.csv"
)

ARTIFACT_DIR = (
    ROOT
    / "models"
    / "artifacts"
)

MODEL_PATH = (
    ARTIFACT_DIR
    / "semantic_hybrid_grader.joblib"
)

METRICS_PATH = (
    ARTIFACT_DIR
    / "semantic_hybrid_metrics.json"
)

PREDICTIONS_PATH = (
    ROOT
    / "data"
    / "semantic_hybrid_oof_predictions.csv"
)


# =========================================================
# MODEL SETTINGS
# =========================================================

ENCODER_NAME = "all-MiniLM-L6-v2"

HUMAN_SAMPLE_WEIGHT = 5.0
SYNTHETIC_SAMPLE_WEIGHT = 1.0

# Once an answer reaches 70% of the reference-answer
# word length, it receives full length adequacy.
#
# Additional verbosity gives no extra reward.
LENGTH_ADEQUACY_THRESHOLD = 0.70


# =========================================================
# FEATURES
# =========================================================

FEATURE_NAMES = [
    "reference_semantic_similarity",
    "question_semantic_similarity",
    "rubric_semantic_similarity",
    "rubric_semantic_coverage",
    "keyword_coverage",
    "reference_token_recall",
    "student_token_precision",
    "length_adequacy_70",
]


# =========================================================
# TEXT HELPERS
# =========================================================

def normalize_tokens(
    text: str,
) -> list[str]:

    return re.findall(
        r"[a-z0-9]+",
        str(text).lower(),
    )


def normalize_answer(
    text: str,
) -> str:

    return " ".join(
        normalize_tokens(text)
    )


def lexical_keyword_coverage(
    answer: str,
    rubric: str,
) -> float:

    answer_tokens = set(
        normalize_tokens(answer)
    )

    keywords = [
        keyword.strip()
        for keyword
        in str(
            rubric or ""
        ).split(";")
        if keyword.strip()
    ]

    if not keywords:
        return 0.0

    hits = 0

    for keyword in keywords:

        keyword_tokens = normalize_tokens(
            keyword
        )

        if (
            keyword_tokens
            and all(
                token in answer_tokens
                for token
                in keyword_tokens
            )
        ):
            hits += 1

    return (
        hits
        / len(keywords)
    )


def cosine(
    a: np.ndarray,
    b: np.ndarray,
) -> float:

    denominator = float(
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if not denominator:
        return 0.0

    return float(
        np.dot(a, b)
        / denominator
    )


def length_adequacy(
    answer: str,
    reference: str,
) -> float:
    """
    Measure whether an answer is sufficiently developed
    without rewarding unnecessary verbosity.

    Example:

        required length = 70% of reference length

        below threshold:
            partial score from 0..1

        at or above threshold:
            1.0

    An answer twice as long as the reference still gets
    only 1.0.
    """

    answer_length = len(
        normalize_tokens(answer)
    )

    reference_length = max(
        1,
        len(
            normalize_tokens(
                reference
            )
        ),
    )

    required_length = max(
        1.0,
        reference_length
        * LENGTH_ADEQUACY_THRESHOLD,
    )

    return min(
        answer_length
        / required_length,
        1.0,
    )


# =========================================================
# FEATURE EXTRACTION
# =========================================================

def features_for_row(
    row: pd.Series,
    encoder: SentenceTransformer,
) -> list[float]:

    question = str(
        row["question_text"]
    )

    reference = str(
        row["model_answer"]
    )

    answer = str(
        row["student_answer"]
    )

    rubric = str(
        row.get(
            "rubric_keywords",
            "",
        )
    )

    keywords = [
        keyword.strip()
        for keyword
        in rubric.split(";")
        if keyword.strip()
    ]


    texts = [
        answer,
        reference,
        question,
    ] + keywords


    embeddings = encoder.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


    answer_embedding = embeddings[0]
    reference_embedding = embeddings[1]
    question_embedding = embeddings[2]

    keyword_embeddings = (
        embeddings[3:]
        if len(embeddings) > 3
        else []
    )


    # -----------------------------------------------------
    # Reference semantic similarity
    # -----------------------------------------------------

    reference_similarity = max(
        0.0,
        cosine(
            answer_embedding,
            reference_embedding,
        ),
    )


    # -----------------------------------------------------
    # Question semantic similarity
    # -----------------------------------------------------

    question_similarity = max(
        0.0,
        cosine(
            answer_embedding,
            question_embedding,
        ),
    )


    # -----------------------------------------------------
    # Rubric semantic features
    # -----------------------------------------------------

    if len(keyword_embeddings):

        keyword_similarities = [
            max(
                0.0,
                cosine(
                    answer_embedding,
                    embedding,
                ),
            )
            for embedding
            in keyword_embeddings
        ]


        rubric_semantic_similarity = float(
            np.mean(
                keyword_similarities
            )
        )


        rubric_semantic_coverage = float(
            np.mean(
                [
                    similarity >= 0.30
                    for similarity
                    in keyword_similarities
                ]
            )
        )

    else:

        rubric_semantic_similarity = 0.0
        rubric_semantic_coverage = 0.0


    # -----------------------------------------------------
    # Lexical overlap
    # -----------------------------------------------------

    reference_tokens = set(
        normalize_tokens(
            reference
        )
    )

    answer_tokens = set(
        normalize_tokens(
            answer
        )
    )

    overlap = (
        reference_tokens
        & answer_tokens
    )


    reference_token_recall = (
        len(overlap)
        / max(
            1,
            len(reference_tokens),
        )
    )


    student_token_precision = (
        len(overlap)
        / max(
            1,
            len(answer_tokens),
        )
    )


    # -----------------------------------------------------
    # Length adequacy
    # -----------------------------------------------------

    adequacy = length_adequacy(
        answer,
        reference,
    )


    return [
        reference_similarity,
        question_similarity,
        rubric_semantic_similarity,
        rubric_semantic_coverage,
        lexical_keyword_coverage(
            answer,
            rubric,
        ),
        reference_token_recall,
        student_token_precision,
        adequacy,
    ]


# =========================================================
# CLEAN TRAINING DATA
# =========================================================

def clean_training_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:

    df = (
        df
        .dropna(
            subset=[
                "human_score"
            ]
        )
        .copy()
    )


    df[
        "_normalized_answer"
    ] = df[
        "student_answer"
    ].map(
        normalize_answer
    )


    cleaned_rows = []

    duplicate_groups = 0
    conflicting_groups = 0
    rows_removed = 0


    group_columns = [
        "question_id",
        "_normalized_answer",
    ]


    for _, group in df.groupby(
        group_columns,
        sort=False,
        dropna=False,
    ):

        group = group.copy()


        if len(group) > 1:

            duplicate_groups += 1

            if (
                group[
                    "human_score"
                ].nunique()
                > 1
            ):
                conflicting_groups += 1

            rows_removed += (
                len(group) - 1
            )


        human_rows = group[
            group["source"]
            == "human_group"
        ]


        if len(human_rows):

            base_row = (
                human_rows.iloc[0]
                .copy()
            )

            final_score = float(
                human_rows[
                    "human_score"
                ].mean()
            )

            final_source = (
                "human_group"
            )

        else:

            base_row = (
                group.iloc[0]
                .copy()
            )

            final_score = float(
                group[
                    "human_score"
                ].mean()
            )

            final_source = (
                "synthetic_augmented"
            )


        base_row[
            "human_score"
        ] = final_score

        base_row[
            "source"
        ] = final_source

        base_row[
            "duplicate_group_size"
        ] = len(group)


        cleaned_rows.append(
            base_row
        )


    cleaned = pd.DataFrame(
        cleaned_rows
    )


    cleaned = (
        cleaned
        .drop(
            columns=[
                "_normalized_answer"
            ],
            errors="ignore",
        )
        .reset_index(
            drop=True
        )
    )


    summary = {
        "original_rows":
            int(
                len(df)
            ),

        "cleaned_rows":
            int(
                len(cleaned)
            ),

        "duplicate_groups":
            int(
                duplicate_groups
            ),

        "conflicting_duplicate_groups":
            int(
                conflicting_groups
            ),

        "rows_removed":
            int(
                rows_removed
            ),
    }


    return (
        cleaned,
        summary,
    )


# =========================================================
# METRICS
# =========================================================

def safe_pearson(
    y_true,
    y_pred,
) -> float:

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )


    if len(y_true) < 2:
        return 0.0


    if (
        np.std(y_true) == 0
        or np.std(y_pred) == 0
    ):
        return 0.0


    return float(
        np.corrcoef(
            y_true,
            y_pred,
        )[0, 1]
    )


def metric_dict(
    y_true,
    y_pred,
) -> dict[str, float]:

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.clip(
        np.asarray(
            y_pred,
            dtype=float,
        ),
        0,
        10,
    )


    rounded_true = np.clip(
        np.rint(
            y_true
        ),
        0,
        10,
    ).astype(int)


    rounded_pred = np.clip(
        np.rint(
            y_pred
        ),
        0,
        10,
    ).astype(int)


    try:

        qwk = float(
            cohen_kappa_score(
                rounded_true,
                rounded_pred,
                weights="quadratic",
            )
        )

        if np.isnan(qwk):
            qwk = 0.0

    except Exception:

        qwk = 0.0


    try:

        r2 = float(
            r2_score(
                y_true,
                y_pred,
            )
        )

    except Exception:

        r2 = 0.0


    return {
        "mae": round(
            float(
                mean_absolute_error(
                    y_true,
                    y_pred,
                )
            ),
            4,
        ),

        "rmse": round(
            float(
                mean_squared_error(
                    y_true,
                    y_pred,
                )
                ** 0.5
            ),
            4,
        ),

        "r2": round(
            r2,
            4,
        ),

        "qwk": round(
            qwk,
            4,
        ),

        "pearson": round(
            safe_pearson(
                y_true,
                y_pred,
            ),
            4,
        ),
    }


def weighted_mae(
    y_true,
    y_pred,
    sample_weights,
) -> float:

    errors = np.abs(
        np.asarray(
            y_true,
            dtype=float,
        )
        -
        np.asarray(
            y_pred,
            dtype=float,
        )
    )


    return float(
        np.average(
            errors,
            weights=sample_weights,
        )
    )


# =========================================================
# SAMPLE WEIGHTS
# =========================================================

def sample_weights_for(
    frame: pd.DataFrame,
) -> np.ndarray:

    return np.where(
        frame[
            "source"
        ].eq(
            "human_group"
        ),
        HUMAN_SAMPLE_WEIGHT,
        SYNTHETIC_SAMPLE_WEIGHT,
    ).astype(float)


# =========================================================
# MAIN
# =========================================================

def main() -> None:

    # -----------------------------------------------------
    # Load + clean
    # -----------------------------------------------------

    raw_df = pd.read_csv(
        DATA_PATH
    )


    (
        df,
        cleaning_summary,
    ) = clean_training_data(
        raw_df
    )


    df.to_csv(
        CLEANED_DATA_PATH,
        index=False,
    )


    print(
        "\n=== DATA CLEANING ==="
    )

    print(
        json.dumps(
            cleaning_summary,
            indent=2,
        )
    )


    print(
        "\nRows by source:"
    )

    print(
        df[
            "source"
        ].value_counts()
    )


    print(
        "\nRows by question:"
    )

    print(
        df.groupby(
            "question_id"
        ).size()
    )


    # -----------------------------------------------------
    # Sentence-BERT
    # -----------------------------------------------------

    print(
        "\nLoading Sentence-BERT..."
    )

    encoder = SentenceTransformer(
        ENCODER_NAME
    )


    print(
        f"Encoding {len(df)} cleaned answers..."
    )


    X_all = np.asarray(
        [
            features_for_row(
                row,
                encoder,
            )

            for _, row
            in df.iterrows()
        ],
        dtype=float,
    )


    y_all = (
        df[
            "human_score"
        ].to_numpy(
            dtype=float
        )
    )


    # -----------------------------------------------------
    # Candidate models
    # -----------------------------------------------------

    candidates = {

        "Ridge":
            Ridge(
                alpha=0.7
            ),

        "RandomForest":
            RandomForestRegressor(
                n_estimators=500,
                max_depth=6,
                min_samples_leaf=2,
                max_features=None,
                random_state=42,
                n_jobs=-1,
            ),

        "ExtraTrees":
            ExtraTreesRegressor(
                n_estimators=500,
                max_depth=7,
                min_samples_leaf=2,
                max_features=None,
                random_state=42,
                n_jobs=-1,
            ),

        "GradientBoosting":
            GradientBoostingRegressor(
                n_estimators=220,
                learning_rate=0.03,
                max_depth=2,
                min_samples_leaf=2,
                loss="huber",
                random_state=42,
            ),
    }


    question_ids = sorted(
        df[
            "question_id"
        ].unique()
    )


    print(
        "\n=== LEAVE-ONE-QUESTION-OUT "
        "CROSS-VALIDATION ==="
    )


    model_results = {}
    model_predictions = {}


    # -----------------------------------------------------
    # Cross-validation
    # -----------------------------------------------------

    for (
        model_name,
        model_template,
    ) in candidates.items():

        print(
            f"\nTesting {model_name}..."
        )


        oof_predictions = np.full(
            len(df),
            np.nan,
            dtype=float,
        )


        fold_results = {}


        for held_out_question in question_ids:

            validation_mask = (
                df[
                    "question_id"
                ]
                .eq(
                    held_out_question
                )
                .to_numpy()
            )


            training_mask = (
                ~validation_mask
            )


            model = clone(
                model_template
            )


            training_frame = (
                df.loc[
                    training_mask
                ]
            )


            training_weights = (
                sample_weights_for(
                    training_frame
                )
            )


            model.fit(
                X_all[
                    training_mask
                ],
                y_all[
                    training_mask
                ],
                sample_weight=(
                    training_weights
                ),
            )


            fold_prediction = (
                model.predict(
                    X_all[
                        validation_mask
                    ]
                )
            )


            fold_prediction = np.clip(
                fold_prediction,
                0,
                10,
            )


            oof_predictions[
                validation_mask
            ] = fold_prediction


            fold_metrics = metric_dict(
                y_all[
                    validation_mask
                ],
                fold_prediction,
            )


            fold_results[
                held_out_question
            ] = fold_metrics


            print(
                f"  Hold out "
                f"{held_out_question}: "
                f"MAE="
                f"{fold_metrics['mae']:.3f}"
            )


        # -------------------------------------------------
        # Aggregate metrics
        # -------------------------------------------------

        overall_metrics = metric_dict(
            y_all,
            oof_predictions,
        )


        human_mask = (
            df[
                "source"
            ]
            .eq(
                "human_group"
            )
            .to_numpy()
        )


        synthetic_mask = (
            ~human_mask
        )


        human_metrics = metric_dict(
            y_all[
                human_mask
            ],
            oof_predictions[
                human_mask
            ],
        )


        synthetic_metrics = metric_dict(
            y_all[
                synthetic_mask
            ],
            oof_predictions[
                synthetic_mask
            ],
        )


        evaluation_weights = (
            sample_weights_for(
                df
            )
        )


        weighted_mae_value = (
            weighted_mae(
                y_all,
                oof_predictions,
                evaluation_weights,
            )
        )


        model_results[
            model_name
        ] = {
            "overall":
                overall_metrics,

            "human_only":
                human_metrics,

            "synthetic_only":
                synthetic_metrics,

            "human_weighted_mae":
                round(
                    weighted_mae_value,
                    4,
                ),

            "folds":
                fold_results,
        }


        model_predictions[
            model_name
        ] = oof_predictions


    # -----------------------------------------------------
    # Select best model
    # -----------------------------------------------------

    best_name = min(
        model_results,

        key=lambda name:
            model_results[
                name
            ][
                "human_weighted_mae"
            ],
    )


    print(
        "\n=== MODEL COMPARISON ==="
    )


    for (
        model_name,
        result,
    ) in model_results.items():

        print(
            f"\n{model_name}"
        )

        print(
            f"  Weighted MAE: "
            f"{result['human_weighted_mae']}"
        )

        print(
            f"  Human MAE: "
            f"{result['human_only']['mae']}"
        )

        print(
            f"  Overall MAE: "
            f"{result['overall']['mae']}"
        )

        print(
            f"  QWK: "
            f"{result['overall']['qwk']}"
        )


    print(
        f"\nSelected model: "
        f"{best_name}"
    )


    # -----------------------------------------------------
    # Train final production model
    # -----------------------------------------------------

    final_model = clone(
        candidates[
            best_name
        ]
    )


    final_weights = (
        sample_weights_for(
            df
        )
    )


    final_model.fit(
        X_all,
        y_all,
        sample_weight=(
            final_weights
        ),
    )


    # -----------------------------------------------------
    # Save artifact
    # -----------------------------------------------------

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    joblib.dump(
        {
            "model":
                final_model,

            "model_name":
                best_name,

            "encoder_name":
                ENCODER_NAME,

            "feature_names":
                FEATURE_NAMES,

            "score_min":
                0.0,

            "score_max":
                10.0,

            "training_method":
                "leave_one_question_out",

            "length_feature":
                "length_adequacy_70",

            "length_adequacy_threshold":
                LENGTH_ADEQUACY_THRESHOLD,

            "human_sample_weight":
                HUMAN_SAMPLE_WEIGHT,

            "synthetic_sample_weight":
                SYNTHETIC_SAMPLE_WEIGHT,

            "cleaning_summary":
                cleaning_summary,
        },
        MODEL_PATH,
    )


    # -----------------------------------------------------
    # Save OOF predictions
    # -----------------------------------------------------

    prediction_df = df[
        [
            "question_id",
            "student_answer",
            "human_score",
            "source",
        ]
    ].copy()


    prediction_df[
        "predicted_score"
    ] = np.round(
        model_predictions[
            best_name
        ],
        2,
    )


    prediction_df[
        "absolute_error"
    ] = np.round(
        np.abs(
            prediction_df[
                "predicted_score"
            ]
            -
            prediction_df[
                "human_score"
            ]
        ),
        2,
    )


    prediction_df.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )


    # -----------------------------------------------------
    # Save metrics
    # -----------------------------------------------------

    summary = {

        "dataset": {
            "original_rows":
                cleaning_summary[
                    "original_rows"
                ],

            "cleaned_rows":
                cleaning_summary[
                    "cleaned_rows"
                ],

            "human_rows":
                int(
                    df[
                        "source"
                    ]
                    .eq(
                        "human_group"
                    )
                    .sum()
                ),

            "synthetic_rows":
                int(
                    df[
                        "source"
                    ]
                    .eq(
                        "synthetic_augmented"
                    )
                    .sum()
                ),

            "questions":
                int(
                    df[
                        "question_id"
                    ]
                    .nunique()
                ),
        },

        "cleaning":
            cleaning_summary,

        "encoder":
            ENCODER_NAME,

        "validation_method":
            (
                "Leave-One-Question-Out "
                "cross-validation"
            ),

        "length_feature": {
            "name":
                "length_adequacy_70",

            "threshold":
                LENGTH_ADEQUACY_THRESHOLD,

            "behavior":
                (
                    "Full adequacy once answer "
                    "reaches 70% of reference "
                    "length; no reward for "
                    "additional verbosity."
                ),
        },

        "sample_weights": {
            "human_group":
                HUMAN_SAMPLE_WEIGHT,

            "synthetic_augmented":
                SYNTHETIC_SAMPLE_WEIGHT,
        },

        "candidate_models":
            model_results,

        "selected_model":
            best_name,

        "selected_model_metrics":
            model_results[
                best_name
            ],

        "note":
            (
                "Whole-question hold-out evaluation "
                "is used to reduce synthetic/template "
                "leakage. Exact duplicate conflicts "
                "are cleaned. Human-scored examples "
                "receive greater weight. Raw answer "
                "length has been replaced by a capped "
                "70% adequacy feature so unnecessary "
                "verbosity does not earn additional "
                "credit."
            ),
    }


    METRICS_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )


    print(
        "\n=== COMPLETE ==="
    )

    print(
        f"Saved cleaned dataset:\n"
        f"{CLEANED_DATA_PATH}"
    )

    print(
        f"\nSaved model:\n"
        f"{MODEL_PATH}"
    )

    print(
        f"\nSaved OOF predictions:\n"
        f"{PREDICTIONS_PATH}"
    )

    print(
        f"\nSaved metrics:\n"
        f"{METRICS_PATH}"
    )


if __name__ == "__main__":
    main()
