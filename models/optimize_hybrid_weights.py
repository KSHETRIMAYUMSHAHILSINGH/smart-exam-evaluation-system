from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_scoring import keyword_match_score
from similarity_scoring import semantic_score


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT
    / "data"
    / "training_dataset_cleaned.csv"
)

PREDICTIONS_PATH = (
    ROOT
    / "data"
    / "semantic_hybrid_oof_predictions.csv"
)

OUTPUT_PATH = (
    ROOT
    / "models"
    / "artifacts"
    / "optimized_hybrid_weights.json"
)


HUMAN_WEIGHT = 5.0
SYNTHETIC_WEIGHT = 1.0


def mae(
    y_true,
    y_pred,
):
    return float(
        np.mean(
            np.abs(
                np.asarray(y_true, float)
                - np.asarray(y_pred, float)
            )
        )
    )


def weighted_mae(
    y_true,
    y_pred,
    weights,
):
    errors = np.abs(
        np.asarray(y_true, float)
        - np.asarray(y_pred, float)
    )

    return float(
        np.average(
            errors,
            weights=np.asarray(
                weights,
                float,
            ),
        )
    )


def calculate_baselines(df):
    keyword_scores = []
    semantic_scores = []

    print(
        f"Calculating baseline scores "
        f"for {len(df)} answers..."
    )

    for index, row in df.iterrows():

        keywords = [
            keyword.strip()
            for keyword
            in str(
                row.get(
                    "rubric_keywords",
                    "",
                )
            ).split(";")
            if keyword.strip()
        ]

        keyword_result = (
            keyword_match_score(
                row["student_answer"],
                keywords,
            )
        )

        semantic_result = (
            semantic_score(
                row["student_answer"],
                row["model_answer"],
            )
        )

        keyword_scores.append(
            float(
                keyword_result["score"]
            )
        )

        semantic_scores.append(
            float(
                semantic_result["score"]
            )
        )

        if (
            (index + 1) % 20 == 0
            or index + 1 == len(df)
        ):
            print(
                f"Processed "
                f"{index + 1}/{len(df)}"
            )

    df["keyword_score"] = (
        keyword_scores
    )

    df["semantic_score"] = (
        semantic_scores
    )

    return df


def main():

    df = pd.read_csv(
        DATA_PATH
    )

    predictions = pd.read_csv(
        PREDICTIONS_PATH
    )

    if len(df) != len(predictions):
        raise RuntimeError(
            "Cleaned dataset and OOF prediction "
            "file have different row counts."
        )

    # The OOF file was produced directly from
    # the cleaned dataset in the same order.
    for column in [
        "question_id",
        "student_answer",
        "source",
    ]:

        if not (
            df[column].astype(str).to_numpy()
            ==
            predictions[column]
            .astype(str)
            .to_numpy()
        ).all():

            raise RuntimeError(
                f"Dataset and predictions are "
                f"misaligned on: {column}"
            )

    df["trained_score"] = (
        predictions[
            "predicted_score"
        ].astype(float)
    )

    df = calculate_baselines(
        df
    )

    y_true = (
        df[
            "human_score"
        ].to_numpy(float)
    )

    sample_weights = np.where(
        df["source"].eq(
            "human_group"
        ),
        HUMAN_WEIGHT,
        SYNTHETIC_WEIGHT,
    )

    human_mask = (
        df["source"]
        .eq("human_group")
        .to_numpy()
    )

    synthetic_mask = (
        ~human_mask
    )

    trained = (
        df[
            "trained_score"
        ].to_numpy(float)
    )

    keyword = (
        df[
            "keyword_score"
        ].to_numpy(float)
    )

    semantic = (
        df[
            "semantic_score"
        ].to_numpy(float)
    )

    # -----------------------------------------------------
    # Existing 50 / 30 / 20 system
    # -----------------------------------------------------

    old_prediction = (
        0.50 * trained
        + 0.30 * keyword
        + 0.20 * semantic
    )

    old_result = {
        "trained_weight": 0.50,
        "keyword_weight": 0.30,
        "semantic_weight": 0.20,

        "weighted_mae": round(
            weighted_mae(
                y_true,
                old_prediction,
                sample_weights,
            ),
            4,
        ),

        "overall_mae": round(
            mae(
                y_true,
                old_prediction,
            ),
            4,
        ),

        "human_mae": round(
            mae(
                y_true[
                    human_mask
                ],
                old_prediction[
                    human_mask
                ],
            ),
            4,
        ),

        "synthetic_mae": round(
            mae(
                y_true[
                    synthetic_mask
                ],
                old_prediction[
                    synthetic_mask
                ],
            ),
            4,
        ),
    }

    # -----------------------------------------------------
    # Search every 1% combination
    #
    # trained + keyword + semantic = 100%
    # -----------------------------------------------------

    results = []

    for trained_percent in range(
        0,
        101,
    ):

        for keyword_percent in range(
            0,
            101 - trained_percent,
        ):

            semantic_percent = (
                100
                - trained_percent
                - keyword_percent
            )

            trained_weight = (
                trained_percent
                / 100.0
            )

            keyword_weight = (
                keyword_percent
                / 100.0
            )

            semantic_weight = (
                semantic_percent
                / 100.0
            )

            prediction = (
                trained_weight
                * trained
                +
                keyword_weight
                * keyword
                +
                semantic_weight
                * semantic
            )

            result = {
                "trained_weight":
                    trained_weight,

                "keyword_weight":
                    keyword_weight,

                "semantic_weight":
                    semantic_weight,

                "weighted_mae":
                    weighted_mae(
                        y_true,
                        prediction,
                        sample_weights,
                    ),

                "overall_mae":
                    mae(
                        y_true,
                        prediction,
                    ),

                "human_mae":
                    mae(
                        y_true[
                            human_mask
                        ],
                        prediction[
                            human_mask
                        ],
                    ),

                "synthetic_mae":
                    mae(
                        y_true[
                            synthetic_mask
                        ],
                        prediction[
                            synthetic_mask
                        ],
                    ),
            }

            results.append(
                result
            )

    results.sort(
        key=lambda item: (
            item[
                "weighted_mae"
            ],
            item[
                "human_mae"
            ],
            item[
                "overall_mae"
            ],
        )
    )

    best = results[0]

    top_10 = results[:10]

    # -----------------------------------------------------
    # Individual systems for comparison
    # -----------------------------------------------------

    individual = {
        "trained_only": {
            "weighted_mae": round(
                weighted_mae(
                    y_true,
                    trained,
                    sample_weights,
                ),
                4,
            ),

            "human_mae": round(
                mae(
                    y_true[
                        human_mask
                    ],
                    trained[
                        human_mask
                    ],
                ),
                4,
            ),
        },

        "keyword_only": {
            "weighted_mae": round(
                weighted_mae(
                    y_true,
                    keyword,
                    sample_weights,
                ),
                4,
            ),

            "human_mae": round(
                mae(
                    y_true[
                        human_mask
                    ],
                    keyword[
                        human_mask
                    ],
                ),
                4,
            ),
        },

        "semantic_only": {
            "weighted_mae": round(
                weighted_mae(
                    y_true,
                    semantic,
                    sample_weights,
                ),
                4,
            ),

            "human_mae": round(
                mae(
                    y_true[
                        human_mask
                    ],
                    semantic[
                        human_mask
                    ],
                ),
                4,
            ),
        },
    }

    output = {
        "rows": int(
            len(df)
        ),

        "human_rows": int(
            human_mask.sum()
        ),

        "synthetic_rows": int(
            synthetic_mask.sum()
        ),

        "selection_metric":
            "human-weighted MAE",

        "sample_weights": {
            "human_group":
                HUMAN_WEIGHT,

            "synthetic_augmented":
                SYNTHETIC_WEIGHT,
        },

        "individual_systems":
            individual,

        "old_weights":
            old_result,

        "optimized_weights": {
            key: round(
                float(value),
                4,
            )

            for key, value
            in best.items()
        },

        "top_10":
            [
                {
                    key: round(
                        float(value),
                        4,
                    )

                    for key, value
                    in result.items()
                }

                for result
                in top_10
            ],
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            output,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n"
        "===================================="
    )

    print(
        "CURRENT WEIGHTS"
    )

    print(
        "===================================="
    )

    print(
        "Trained:  50%"
    )

    print(
        "Keyword:  30%"
    )

    print(
        "Semantic: 20%"
    )

    print(
        f"Weighted MAE: "
        f"{old_result['weighted_mae']}"
    )

    print(
        f"Human MAE: "
        f"{old_result['human_mae']}"
    )

    print(
        "\n"
        "===================================="
    )

    print(
        "OPTIMIZED WEIGHTS"
    )

    print(
        "===================================="
    )

    print(
        f"Trained:  "
        f"{best['trained_weight'] * 100:.0f}%"
    )

    print(
        f"Keyword:  "
        f"{best['keyword_weight'] * 100:.0f}%"
    )

    print(
        f"Semantic: "
        f"{best['semantic_weight'] * 100:.0f}%"
    )

    print(
        f"Weighted MAE: "
        f"{best['weighted_mae']:.4f}"
    )

    print(
        f"Human MAE: "
        f"{best['human_mae']:.4f}"
    )

    print(
        f"Overall MAE: "
        f"{best['overall_mae']:.4f}"
    )

    print(
        "\n"
        "===================================="
    )

    print(
        "INDIVIDUAL SYSTEMS"
    )

    print(
        "===================================="
    )

    for name, values in individual.items():

        print(
            f"{name}: "
            f"weighted MAE="
            f"{values['weighted_mae']}, "
            f"human MAE="
            f"{values['human_mae']}"
        )

    print(
        f"\nSaved results:\n"
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
