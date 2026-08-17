"""
Experiment with sentence-level rubric concept matching.

This DOES NOT overwrite the production grader.

Goal:
Improve grading of:
- short but correct answers
- paraphrased concepts
- long but weak answers
- irrelevant padding

Run:
    python models/concept_feature_experiment.py
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    cohen_kappa_score,
    mean_absolute_error,
)

from train_semantic_hybrid_grader import (
    ENCODER_NAME,
    FEATURE_NAMES,
    features_for_row,
)


# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------

DATA_PATH = "data/training_dataset_cleaned.csv"

HUMAN_WEIGHT = 5.0
SYNTHETIC_WEIGHT = 1.0

SEMANTIC_CONCEPT_THRESHOLD = 0.38


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def split_sentences(text: str) -> list[str]:

    text = str(text).strip()

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    if not sentences:
        return [text]

    return sentences


def sample_weights(frame):

    return np.where(
        frame["source"].eq("human_group"),
        HUMAN_WEIGHT,
        SYNTHETIC_WEIGHT,
    ).astype(float)


def weighted_mae(
    true,
    predicted,
    weights,
):

    errors = np.abs(
        np.asarray(true, float)
        - np.asarray(predicted, float)
    )

    return float(
        np.average(
            errors,
            weights=np.asarray(weights, float),
        )
    )


def qwk(
    true,
    predicted,
):

    true = np.clip(
        np.rint(true),
        0,
        10,
    ).astype(int)

    predicted = np.clip(
        np.rint(predicted),
        0,
        10,
    ).astype(int)

    result = cohen_kappa_score(
        true,
        predicted,
        weights="quadratic",
    )

    if np.isnan(result):
        return 0.0

    return float(result)


# ---------------------------------------------------------
# New concept features
# ---------------------------------------------------------

def concept_features(
    answer: str,
    rubric: str,
    encoder,
):
    """
    Compare every rubric concept against each sentence
    in the student answer.

    For each concept we keep its BEST sentence match.
    """

    keywords = [
        keyword.strip()
        for keyword
        in str(rubric or "").split(";")
        if keyword.strip()
    ]

    if not keywords:
        return [
            0.0,
            0.0,
            0.0,
        ]

    sentences = split_sentences(
        answer
    )

    texts = (
        sentences
        + keywords
    )

    embeddings = encoder.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    sentence_embeddings = embeddings[
        :len(sentences)
    ]

    keyword_embeddings = embeddings[
        len(sentences):
    ]

    # cosine similarity because embeddings
    # are already normalized
    similarity_matrix = (
        keyword_embeddings
        @ sentence_embeddings.T
    )

    # Best sentence evidence for every concept.
    best_matches = np.max(
        similarity_matrix,
        axis=1,
    )

    best_matches = np.clip(
        best_matches,
        0.0,
        1.0,
    )

    concept_mean = float(
        np.mean(best_matches)
    )

    concept_coverage = float(
        np.mean(
            best_matches
            >= SEMANTIC_CONCEPT_THRESHOLD
        )
    )

    concept_minimum = float(
        np.min(best_matches)
    )

    return [
        concept_mean,
        concept_coverage,
        concept_minimum,
    ]


# ---------------------------------------------------------
# Feature sets
# ---------------------------------------------------------

def build_features(
    row,
    encoder,
    mode,
):

    base = features_for_row(
        row,
        encoder,
    )

    concepts = concept_features(
        row["student_answer"],
        row.get(
            "rubric_keywords",
            "",
        ),
        encoder,
    )

    base_map = dict(
        zip(
            FEATURE_NAMES,
            base,
        )
    )

    if mode == "current":

        return base


    if mode == "concept_plus_current":

        return (
            base
            + concepts
        )


    if mode == "concept_replacement":

        # Remove the old whole-answer rubric semantic
        # features and replace them with sentence-level
        # concept evidence.

        return [
            base_map[
                "reference_semantic_similarity"
            ],

            base_map[
                "question_semantic_similarity"
            ],

            base_map[
                "keyword_coverage"
            ],

            base_map[
                "reference_token_recall"
            ],

            base_map[
                "student_token_precision"
            ],

            base_map[
                "length_adequacy_70"
            ],

            *concepts,
        ]


    raise ValueError(
        f"Unknown feature mode: {mode}"
    )


# ---------------------------------------------------------
# Evaluate one feature design
# ---------------------------------------------------------

def evaluate_mode(
    df,
    encoder,
    mode,
):

    print(
        f"\nBuilding features for: {mode}"
    )

    X = np.asarray(
        [
            build_features(
                row,
                encoder,
                mode,
            )

            for _, row
            in df.iterrows()
        ],
        dtype=float,
    )

    y = df[
        "human_score"
    ].to_numpy(float)

    questions = sorted(
        df[
            "question_id"
        ].unique()
    )

    predictions = np.full(
        len(df),
        np.nan,
    )

    for held_out_question in questions:

        test_mask = (
            df["question_id"]
            .eq(held_out_question)
            .to_numpy()
        )

        train_mask = ~test_mask

        model = GradientBoostingRegressor(
            n_estimators=220,
            learning_rate=0.03,
            max_depth=2,
            min_samples_leaf=2,
            loss="huber",
            random_state=42,
        )

        model.fit(
            X[train_mask],
            y[train_mask],
            sample_weight=sample_weights(
                df.loc[train_mask]
            ),
        )

        predictions[test_mask] = np.clip(
            model.predict(
                X[test_mask]
            ),
            0,
            10,
        )

    human_mask = (
        df["source"]
        .eq("human_group")
        .to_numpy()
    )

    weights = sample_weights(
        df
    )

    return {
        "weighted_mae": weighted_mae(
            y,
            predictions,
            weights,
        ),

        "human_mae": float(
            mean_absolute_error(
                y[human_mask],
                predictions[human_mask],
            )
        ),

        "overall_mae": float(
            mean_absolute_error(
                y,
                predictions,
            )
        ),

        "qwk": qwk(
            y,
            predictions,
        ),
    }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    df = pd.read_csv(
        DATA_PATH
    )

    print(
        f"Loaded {len(df)} cleaned answers."
    )

    print(
        "Loading Sentence-BERT..."
    )

    encoder = SentenceTransformer(
        ENCODER_NAME
    )

    modes = [
        "current",
        "concept_plus_current",
        "concept_replacement",
    ]

    results = {}

    for mode in modes:

        result = evaluate_mode(
            df,
            encoder,
            mode,
        )

        results[mode] = result

        print(
            f"\n{mode}"
        )

        print(
            f"  Human MAE:    "
            f"{result['human_mae']:.4f}"
        )

        print(
            f"  Weighted MAE: "
            f"{result['weighted_mae']:.4f}"
        )

        print(
            f"  Overall MAE:  "
            f"{result['overall_mae']:.4f}"
        )

        print(
            f"  QWK:          "
            f"{result['qwk']:.4f}"
        )


    ranking = sorted(
        results.items(),
        key=lambda item: (
            item[1]["human_mae"],
            item[1]["weighted_mae"],
        ),
    )


    print(
        "\n"
        "======================================"
    )

    print(
        "CONCEPT FEATURE RANKING"
    )

    print(
        "======================================"
    )


    for number, (
        name,
        result,
    ) in enumerate(
        ranking,
        1,
    ):

        print(
            f"{number}. {name}"
        )

        print(
            f"   Human MAE:    "
            f"{result['human_mae']:.4f}"
        )

        print(
            f"   Weighted MAE: "
            f"{result['weighted_mae']:.4f}"
        )

        print(
            f"   Overall MAE:  "
            f"{result['overall_mae']:.4f}"
        )

        print(
            f"   QWK:          "
            f"{result['qwk']:.4f}"
        )


    print(
        "\nWINNER:"
    )

    print(
        ranking[0][0]
    )


    # -----------------------------------------------------
    # Show concept features for our robustness answers
    # -----------------------------------------------------

    q1 = df[
        df["question_id"]
        == "Q1"
    ].iloc[0]


    tests = {

        "A_actual": """
Artificial intelligence allows computers to imitate some human abilities such as learning from data, recognizing patterns, making decisions and solving problems. It is used in chatbots, recommendation systems and image recognition.
""",

        "B_short_correct": """
Artificial intelligence enables computers to learn from data, recognize patterns, make decisions, reason, and solve problems.
""",

        "C_strong_concise": """
Artificial intelligence enables computers to perform human-like tasks including learning, reasoning, problem-solving, understanding information, and recognizing patterns. It is used in chatbots, recommendation systems, image recognition, and automated evaluation.
""",

        "D_irrelevant_padding": """
Artificial intelligence allows computers to imitate some human abilities such as learning from data, recognizing patterns, making decisions and solving problems. It is used in chatbots, recommendation systems and image recognition. The weather can change during the day. Students use books and notebooks in classrooms. Cars travel on roads and people use phones for communication.
""",

        "E_long_weak": """
Artificial intelligence is a computer technology used in many places. Computers are used in schools, offices, hospitals, shops, homes, banks, airports, factories, and many other organizations. Modern computers can store large amounts of information and run many different applications. Technology is important in everyday life and continues to develop rapidly.
""",
    }


    print(
        "\n"
        "======================================"
    )

    print(
        "NEW CONCEPT FEATURES"
    )

    print(
        "======================================"
    )


    for name, answer in tests.items():

        values = concept_features(
            answer.strip(),
            q1["rubric_keywords"],
            encoder,
        )

        print(
            f"\n{name}"
        )

        print(
            f"concept_mean:     "
            f"{values[0]:.4f}"
        )

        print(
            f"concept_coverage: "
            f"{values[1]:.4f}"
        )

        print(
            f"concept_minimum:  "
            f"{values[2]:.4f}"
        )


if __name__ == "__main__":
    main()
