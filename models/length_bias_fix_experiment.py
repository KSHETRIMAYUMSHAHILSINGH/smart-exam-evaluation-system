"""
Experiment with safer answer-length features.

This DOES NOT overwrite the production grader.

We compare:

1. Current length_ratio
2. No length feature
3. Length adequacy capped at 50% of reference length
4. Length adequacy capped at 60% of reference length
5. Length adequacy capped at 70% of reference length

The capped feature means that once an answer is reasonably long,
extra verbosity receives no additional reward.

Run:

    python models/length_bias_fix_experiment.py
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
)

from train_semantic_hybrid_grader import (
    ENCODER_NAME,
    FEATURE_NAMES,
    features_for_row,
    normalize_tokens,
)


# =========================================================
# PATHS
# =========================================================

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
    / "length_bias_experiment.json"
)


# =========================================================
# WEIGHTS
# =========================================================

HUMAN_WEIGHT = 5.0
SYNTHETIC_WEIGHT = 1.0


# =========================================================
# EXPERIMENTS
# =========================================================

EXPERIMENTS = {
    "A_current_length": {
        "type": "current",
    },

    "B_no_length": {
        "type": "none",
    },

    "C_adequacy_50": {
        "type": "adequacy",
        "threshold": 0.50,
    },

    "D_adequacy_60": {
        "type": "adequacy",
        "threshold": 0.60,
    },

    "E_adequacy_70": {
        "type": "adequacy",
        "threshold": 0.70,
    },
}


# =========================================================
# HELPERS
# =========================================================

def sample_weights(frame):

    return np.where(
        frame["source"].eq(
            "human_group"
        ),
        HUMAN_WEIGHT,
        SYNTHETIC_WEIGHT,
    ).astype(float)


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


def qwk_score(
    y_true,
    y_pred,
):

    true_rounded = np.clip(
        np.rint(y_true),
        0,
        10,
    ).astype(int)

    pred_rounded = np.clip(
        np.rint(y_pred),
        0,
        10,
    ).astype(int)

    score = cohen_kappa_score(
        true_rounded,
        pred_rounded,
        weights="quadratic",
    )

    if np.isnan(score):
        return 0.0

    return float(score)


def length_adequacy(
    answer,
    reference,
    threshold,
):
    """
    Return 0..1.

    Once the answer reaches the required percentage
    of the reference length, it gets 1.0.

    Going beyond that length provides no extra reward.
    """

    answer_length = len(
        normalize_tokens(
            answer
        )
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
        * threshold,
    )

    return min(
        answer_length
        / required_length,
        1.0,
    )


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
        "Loading Sentence-BERT..."
    )

    encoder = SentenceTransformer(
        ENCODER_NAME
    )


    # -----------------------------------------------------
    # Existing features
    # -----------------------------------------------------

    print(
        "Calculating base features..."
    )

    X_base = np.asarray(
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
        ]
        .to_numpy(float)
    )


    feature_index = {
        name: index
        for index, name
        in enumerate(
            FEATURE_NAMES
        )
    }


    length_index = feature_index[
        "length_ratio"
    ]


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


    eval_weights = sample_weights(
        df
    )


    results = {}


    # -----------------------------------------------------
    # Experiments
    # -----------------------------------------------------

    for (
        experiment_name,
        config,
    ) in EXPERIMENTS.items():

        print(
            "\n"
            "======================================"
        )

        print(
            experiment_name
        )

        print(
            "======================================"
        )


        X = X_base.copy()

        selected_feature_names = list(
            FEATURE_NAMES
        )


        # -------------------------------------------------
        # Remove length
        # -------------------------------------------------

        if config["type"] == "none":

            X = np.delete(
                X,
                length_index,
                axis=1,
            )

            selected_feature_names.remove(
                "length_ratio"
            )


        # -------------------------------------------------
        # Replace length with capped adequacy
        # -------------------------------------------------

        elif config["type"] == "adequacy":

            threshold = config[
                "threshold"
            ]

            adequacy_values = np.asarray(
                [
                    length_adequacy(
                        row[
                            "student_answer"
                        ],
                        row[
                            "model_answer"
                        ],
                        threshold,
                    )

                    for _, row
                    in df.iterrows()
                ],
                dtype=float,
            )


            X[
                :,
                length_index
            ] = adequacy_values


            selected_feature_names[
                length_index
            ] = (
                f"length_adequacy_"
                f"{int(threshold * 100)}"
            )


        # -------------------------------------------------
        # Leave-one-question-out predictions
        # -------------------------------------------------

        predictions = np.full(
            len(df),
            np.nan,
            dtype=float,
        )


        importances = []


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


            model.fit(
                X[
                    train_mask
                ],
                y[
                    train_mask
                ],
                sample_weight=sample_weights(
                    df.loc[
                        train_mask
                    ]
                ),
            )


            fold_prediction = (
                model.predict(
                    X[
                        test_mask
                    ]
                )
            )


            predictions[
                test_mask
            ] = np.clip(
                fold_prediction,
                0,
                10,
            )


            importances.append(
                model.feature_importances_
            )


        # -------------------------------------------------
        # Metrics
        # -------------------------------------------------

        overall_mae = float(
            mean_absolute_error(
                y,
                predictions,
            )
        )


        human_mae = float(
            mean_absolute_error(
                y[
                    human_mask
                ],
                predictions[
                    human_mask
                ],
            )
        )


        weighted = weighted_mae(
            y,
            predictions,
            eval_weights,
        )


        qwk = qwk_score(
            y,
            predictions,
        )


        mean_importance = np.mean(
            np.asarray(
                importances
            ),
            axis=0,
        )


        importance_pairs = sorted(
            zip(
                selected_feature_names,
                mean_importance,
            ),
            key=lambda pair:
                pair[1],
            reverse=True,
        )


        results[
            experiment_name
        ] = {
            "weighted_mae":
                round(
                    weighted,
                    4,
                ),

            "human_mae":
                round(
                    human_mae,
                    4,
                ),

            "overall_mae":
                round(
                    overall_mae,
                    4,
                ),

            "qwk":
                round(
                    qwk,
                    4,
                ),

            "feature_importance": {
                name: round(
                    float(value),
                    4,
                )

                for name, value
                in importance_pairs
            },
        }


        print(
            f"Weighted MAE: "
            f"{weighted:.4f}"
        )

        print(
            f"Human MAE: "
            f"{human_mae:.4f}"
        )

        print(
            f"Overall MAE: "
            f"{overall_mae:.4f}"
        )

        print(
            f"QWK: "
            f"{qwk:.4f}"
        )


        print(
            "\nFeature importance:"
        )


        for (
            name,
            importance,
        ) in importance_pairs:

            print(
                f"{name:35s} "
                f"{importance:.4f}"
            )


    # -----------------------------------------------------
    # Ranking
    # -----------------------------------------------------

    ranking = sorted(
        results.items(),

        key=lambda item: (
            item[1][
                "human_mae"
            ],
            item[1][
                "weighted_mae"
            ],
        ),
    )


    print(
        "\n\n"
        "======================================"
    )

    print(
        "FINAL LENGTH-BIAS RANKING"
    )

    print(
        "======================================"
    )


    for (
        rank,
        (
            name,
            result,
        ),
    ) in enumerate(
        ranking,
        start=1,
    ):

        print(
            f"{rank}. {name}"
        )

        print(
            f"   Human MAE:    "
            f"{result['human_mae']}"
        )

        print(
            f"   Weighted MAE: "
            f"{result['weighted_mae']}"
        )

        print(
            f"   Overall MAE:  "
            f"{result['overall_mae']}"
        )

        print(
            f"   QWK:          "
            f"{result['qwk']}"
        )


    print(
        "\nWINNER:"
    )

    print(
        ranking[0][0]
    )


    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    OUTPUT_PATH.write_text(
        json.dumps(
            {
                "winner":
                    ranking[0][0],

                "results":
                    results,
            },
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
