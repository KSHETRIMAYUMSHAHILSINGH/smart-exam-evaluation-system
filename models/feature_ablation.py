"""
Feature-ablation experiment for the semantic hybrid grader.

Purpose:
Determine whether shortcut features such as answer length and token overlap
are causing the trained grading model to under-score concise but correct answers.

This script DOES NOT overwrite the production model.

Run:
    python models/feature_ablation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import (
    cohen_kappa_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from train_semantic_hybrid_grader import (
    ENCODER_NAME,
    FEATURE_NAMES,
    features_for_row,
)


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT
    / "data"
    / "training_dataset_cleaned.csv"
)

OUTPUT_PATH = (
    ROOT
    / "models"
    / "artifacts"
    / "feature_ablation_results.json"
)


HUMAN_WEIGHT = 5.0
SYNTHETIC_WEIGHT = 1.0


# =========================================================
# FEATURE SETS TO TEST
# =========================================================

FEATURE_SETS = {

    # Current production feature set.
    "A_current_all": [
        "reference_semantic_similarity",
        "question_semantic_similarity",
        "rubric_semantic_similarity",
        "rubric_semantic_coverage",
        "keyword_coverage",
        "reference_token_recall",
        "student_token_precision",
        "length_ratio",
    ],

    # Remove only the biggest shortcut.
    "B_no_length": [
        "reference_semantic_similarity",
        "question_semantic_similarity",
        "rubric_semantic_similarity",
        "rubric_semantic_coverage",
        "keyword_coverage",
        "reference_token_recall",
        "student_token_precision",
    ],

    # Remove length + student token precision.
    "C_no_length_no_precision": [
        "reference_semantic_similarity",
        "question_semantic_similarity",
        "rubric_semantic_similarity",
        "rubric_semantic_coverage",
        "keyword_coverage",
        "reference_token_recall",
    ],

    # Keep meaning + concepts, with only one lexical recall feature.
    "D_balanced_semantic": [
        "reference_semantic_similarity",
        "question_semantic_similarity",
        "rubric_semantic_similarity",
        "rubric_semantic_coverage",
        "keyword_coverage",
        "reference_token_recall",
    ],

    # Purely meaning/concept-focused model.
    "E_meaning_concepts_only": [
        "reference_semantic_similarity",
        "question_semantic_similarity",
        "rubric_semantic_similarity",
        "rubric_semantic_coverage",
        "keyword_coverage",
    ],
}


# =========================================================
# METRICS
# =========================================================

def safe_pearson(
    y_true,
    y_pred,
):

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


def metrics(
    y_true,
    y_pred,
):

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
        np.rint(y_true),
        0,
        10,
    ).astype(int)

    rounded_pred = np.clip(
        np.rint(y_pred),
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
    weights,
):

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
            weights=np.asarray(
                weights,
                dtype=float,
            ),
        )
    )


def sample_weights(
    frame,
):

    return np.where(
        frame[
            "source"
        ].eq(
            "human_group"
        ),
        HUMAN_WEIGHT,
        SYNTHETIC_WEIGHT,
    ).astype(float)


# =========================================================
# MAIN
# =========================================================

def main():

    df = pd.read_csv(
        DATA_PATH
    )

    print(
        f"\nLoaded {len(df)} cleaned answers."
    )

    print(
        f"Questions: "
        f"{df['question_id'].nunique()}"
    )

    print(
        f"Human rows: "
        f"{df['source'].eq('human_group').sum()}"
    )

    print(
        f"Synthetic rows: "
        f"{df['source'].eq('synthetic_augmented').sum()}"
    )


    # -----------------------------------------------------
    # Encode features ONCE
    # -----------------------------------------------------

    print(
        "\nLoading Sentence-BERT..."
    )

    encoder = SentenceTransformer(
        ENCODER_NAME
    )

    print(
        "Calculating full feature matrix..."
    )

    X_full = np.asarray(
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

    y = (
        df[
            "human_score"
        ].to_numpy(float)
    )


    feature_index = {
        name: index
        for index, name
        in enumerate(FEATURE_NAMES)
    }


    question_ids = sorted(
        df[
            "question_id"
        ].unique()
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


    evaluation_weights = (
        sample_weights(
            df
        )
    )


    results = {}


    # -----------------------------------------------------
    # Test each feature set
    # -----------------------------------------------------

    for (
        set_name,
        selected_features,
    ) in FEATURE_SETS.items():

        print(
            "\n"
            "========================================"
        )

        print(
            set_name
        )

        print(
            "========================================"
        )

        print(
            "Features:"
        )

        for feature in selected_features:
            print(
                f"  - {feature}"
            )


        indices = [
            feature_index[
                feature
            ]
            for feature
            in selected_features
        ]


        X = X_full[
            :,
            indices
        ]


        oof_predictions = np.full(
            len(df),
            np.nan,
            dtype=float,
        )


        fold_importances = []


        # -------------------------------------------------
        # Leave-one-question-out validation
        # -------------------------------------------------

        for held_out_question in question_ids:

            test_mask = (
                df[
                    "question_id"
                ]
                .eq(
                    held_out_question
                )
                .to_numpy()
            )

            train_mask = (
                ~test_mask
            )


            model = ExtraTreesRegressor(
                n_estimators=500,
                max_depth=7,
                min_samples_leaf=2,
                max_features=None,
                random_state=42,
                n_jobs=-1,
            )


            train_frame = (
                df.loc[
                    train_mask
                ]
            )


            weights = (
                sample_weights(
                    train_frame
                )
            )


            model.fit(
                X[
                    train_mask
                ],
                y[
                    train_mask
                ],
                sample_weight=weights,
            )


            predictions = (
                model.predict(
                    X[
                        test_mask
                    ]
                )
            )


            predictions = np.clip(
                predictions,
                0,
                10,
            )


            oof_predictions[
                test_mask
            ] = predictions


            fold_importances.append(
                model.feature_importances_
            )


            fold_mae = (
                mean_absolute_error(
                    y[
                        test_mask
                    ],
                    predictions,
                )
            )


            print(
                f"Hold out "
                f"{held_out_question}: "
                f"MAE={fold_mae:.3f}"
            )


        # -------------------------------------------------
        # Aggregate results
        # -------------------------------------------------

        overall = metrics(
            y,
            oof_predictions,
        )


        human = metrics(
            y[
                human_mask
            ],
            oof_predictions[
                human_mask
            ],
        )


        weighted = weighted_mae(
            y,
            oof_predictions,
            evaluation_weights,
        )


        mean_importance = np.mean(
            np.asarray(
                fold_importances
            ),
            axis=0,
        )


        importance_dict = {
            feature: round(
                float(importance),
                4,
            )
            for feature, importance
            in sorted(
                zip(
                    selected_features,
                    mean_importance,
                ),
                key=lambda item:
                    item[1],
                reverse=True,
            )
        }


        results[
            set_name
        ] = {
            "features":
                selected_features,

            "weighted_mae":
                round(
                    weighted,
                    4,
                ),

            "human_mae":
                human[
                    "mae"
                ],

            "overall_mae":
                overall[
                    "mae"
                ],

            "qwk":
                overall[
                    "qwk"
                ],

            "pearson":
                overall[
                    "pearson"
                ],

            "feature_importance":
                importance_dict,
        }


        print(
            "\nRESULT"
        )

        print(
            f"Weighted MAE: "
            f"{weighted:.4f}"
        )

        print(
            f"Human MAE: "
            f"{human['mae']:.4f}"
        )

        print(
            f"Overall MAE: "
            f"{overall['mae']:.4f}"
        )

        print(
            f"QWK: "
            f"{overall['qwk']:.4f}"
        )


        print(
            "\nMean feature importance:"
        )

        for (
            feature,
            importance,
        ) in importance_dict.items():

            print(
                f"{feature:35s} "
                f"{importance:.4f}"
            )


    # -----------------------------------------------------
    # Rank models
    # -----------------------------------------------------

    ranked = sorted(
        results.items(),
        key=lambda item: (
            item[1][
                "weighted_mae"
            ],
            item[1][
                "human_mae"
            ],
        ),
    )


    print(
        "\n\n"
        "========================================"
    )

    print(
        "FINAL ABLATION RANKING"
    )

    print(
        "========================================"
    )


    for (
        rank,
        (
            name,
            result,
        ),
    ) in enumerate(
        ranked,
        start=1,
    ):

        print(
            f"{rank}. {name}"
        )

        print(
            f"   Weighted MAE: "
            f"{result['weighted_mae']}"
        )

        print(
            f"   Human MAE:    "
            f"{result['human_mae']}"
        )

        print(
            f"   Overall MAE:  "
            f"{result['overall_mae']}"
        )

        print(
            f"   QWK:          "
            f"{result['qwk']}"
        )


    winner = ranked[0][0]


    print(
        "\nWINNER:"
    )

    print(
        winner
    )


    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    output = {
        "winner":
            winner,

        "evaluation":
            (
                "Leave-One-Question-Out "
                "cross-validation"
            ),

        "results":
            results,
    }


    OUTPUT_PATH.write_text(
        json.dumps(
            output,
            indent=2,
        ),
        encoding="utf-8",
    )


    print(
        f"\nSaved:\n"
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
