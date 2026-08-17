"""
Build the balanced training dataset for Smart Exam Evaluation.

Dataset design
--------------
20 real human-group answers
40 mid-range paraphrased synthetic answers
40 robustness/boundary synthetic answers

Total: 100 rows

The robustness examples deliberately break the old shortcut:

    longer answer = higher score

They include:
- long irrelevant answers
- long misconceptions
- long but weak answers
- short but strong answers
- concise high-quality paraphrases

Run:
    python scripts/build_training_dataset.py
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from expand_midrange_dataset import EXTRA as MIDRANGE_EXAMPLES


ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "data" / "manifest.csv"

OUTPUT = ROOT / "data" / "training_dataset.csv"


# =========================================================
# ROBUSTNESS / BOUNDARY EXAMPLES
# =========================================================

ROBUST_EXAMPLES = {

    # -----------------------------------------------------
    # Q1 — Artificial Intelligence
    # -----------------------------------------------------

    "Q1": [

        (
            0,
            "Cloud storage keeps files on remote servers and allows "
            "people to access them through the internet. Databases "
            "organize records into structured collections, while "
            "computer networks connect devices so information can "
            "be transferred between different locations."
        ),

        (
            1,
            "Artificial intelligence mainly means storing very large "
            "amounts of information inside a computer. An AI system "
            "works by keeping records in databases and retrieving "
            "those records quickly whenever a user requests them. "
            "It does not need to learn, reason, or recognize patterns."
        ),

        (
            2,
            "Artificial intelligence is computer technology used to "
            "automate some tasks and make software behave intelligently."
        ),

        (
            3,
            "Artificial intelligence is a modern computer technology "
            "used in schools, offices, hospitals, phones, websites, "
            "and many other places. It can make computer applications "
            "more useful and is becoming increasingly common in "
            "everyday life as technology continues to improve."
        ),

        (
            8,
            "Artificial intelligence enables machines to learn from "
            "data, recognize patterns, make decisions, and solve problems."
        ),

        (
            8,
            "AI creates computer systems that imitate several human "
            "abilities. They can adapt from examples, identify useful "
            "patterns, choose suitable actions, and solve unfamiliar "
            "problems without relying only on fixed instructions."
        ),

        (
            9,
            "Artificial intelligence enables machines to perform "
            "human-like cognitive tasks such as learning from data, "
            "reasoning, solving problems, understanding language, "
            "recognizing patterns, and making decisions."
        ),

        (
            10,
            "Artificial Intelligence is the field of creating computer "
            "systems capable of tasks associated with human intelligence. "
            "AI systems can learn from examples, reason, solve problems, "
            "make decisions, understand language, and recognize patterns "
            "in data. Applications include chatbots, recommendation "
            "systems, image recognition, and automated evaluation."
        ),
    ],


    # -----------------------------------------------------
    # Q2 — Machine Learning
    # -----------------------------------------------------

    "Q2": [

        (
            0,
            "A database management system stores records in tables, "
            "controls access to information, supports queries, and "
            "helps applications retrieve saved data. Cloud systems "
            "can host these databases on remote servers accessed "
            "through the internet."
        ),

        (
            1,
            "Machine learning means programmers manually write every "
            "possible decision rule before the program runs. The "
            "computer always follows those fixed instructions and "
            "does not change its behaviour based on examples or data."
        ),

        (
            2,
            "Machine learning allows a computer to improve at a task "
            "after seeing examples."
        ),

        (
            3,
            "Machine learning is a modern computer technology used "
            "in websites, mobile applications, businesses, hospitals, "
            "schools, and many other systems. It is popular because "
            "computers can process large amounts of information quickly "
            "and technology is becoming more important every year."
        ),

        (
            8,
            "Machine learning lets models learn patterns from training "
            "data and use those patterns to make predictions or "
            "decisions on new inputs."
        ),

        (
            8,
            "ML trains a model using examples instead of programming "
            "a separate rule for every situation. The trained model "
            "can then generalize what it learned to unseen data."
        ),

        (
            9,
            "Machine learning is a branch of AI where models learn "
            "patterns from examples, adjust during training, and use "
            "the learned relationships to classify, predict, or make "
            "decisions about new data."
        ),

        (
            10,
            "Machine Learning is a branch of Artificial Intelligence "
            "in which models learn patterns and relationships from "
            "training data. Training adjusts the model so it can "
            "generalize to unseen inputs instead of requiring a "
            "hard-coded rule for every case. Learned models can perform "
            "tasks such as classification, regression, prediction, "
            "and automated decision making."
        ),
    ],


    # -----------------------------------------------------
    # Q3 — Natural Language Processing
    # -----------------------------------------------------

    "Q3": [

        (
            0,
            "Cloud computing provides remote servers, storage, "
            "databases, networking, and software over the internet. "
            "Organizations use these services so they do not need "
            "to maintain every physical server and storage device "
            "inside their own buildings."
        ),

        (
            1,
            "Natural Language Processing is a networking protocol "
            "that transfers files between computers. It controls "
            "connections between devices and determines how packets "
            "travel across a network from one machine to another."
        ),

        (
            2,
            "Natural language processing helps computers work with "
            "text written by people."
        ),

        (
            3,
            "Natural language processing is a technology used in "
            "phones, websites, applications, customer service systems, "
            "and many online platforms. It is useful because people "
            "communicate through computers every day and modern "
            "software needs to handle large amounts of information."
        ),

        (
            8,
            "NLP enables computers to process human text or speech "
            "and perform tasks such as classification, sentiment "
            "analysis, translation, and question answering."
        ),

        (
            8,
            "Natural language processing represents and analyzes "
            "human language so software can interpret meaning, compare "
            "texts, classify messages, and generate useful responses."
        ),

        (
            9,
            "NLP is an area of AI that enables computers to understand "
            "and analyze written or spoken language. It supports "
            "classification, sentiment analysis, translation, question "
            "answering, semantic similarity, and comparison of answers."
        ),

        (
            10,
            "Natural Language Processing is an area of AI concerned "
            "with processing and understanding human language. NLP "
            "systems analyze text or speech and can perform sentiment "
            "analysis, classification, translation, question answering, "
            "and semantic comparison. In automated grading, these "
            "techniques can compare a student's response with a "
            "reference answer based on meaning."
        ),
    ],


    # -----------------------------------------------------
    # Q4 — Cloud Computing
    # -----------------------------------------------------

    "Q4": [

        (
            0,
            "Machine learning models examine training examples to "
            "discover statistical patterns. After training, those "
            "patterns can be used for classification, regression, "
            "prediction, recommendation, and automated decisions "
            "on previously unseen data."
        ),

        (
            1,
            "Cloud computing is mainly a weather forecasting technique. "
            "Computers examine clouds, temperature, rainfall, wind, "
            "and atmospheric measurements so meteorologists can "
            "predict future weather conditions and storms."
        ),

        (
            2,
            "Cloud computing lets people store files and access "
            "computer services through the internet."
        ),

        (
            3,
            "Cloud computing is an internet technology used by "
            "companies, schools, websites, mobile applications, "
            "and many other organizations. It is increasingly popular "
            "because the internet is widely available and modern "
            "businesses use many online services every day."
        ),

        (
            8,
            "Cloud computing provides remote servers, storage, "
            "databases, and software over the internet and allows "
            "resources to scale according to demand."
        ),

        (
            8,
            "Cloud services let users access computing infrastructure "
            "remotely instead of maintaining all hardware locally. "
            "They provide flexible capacity and access from different "
            "locations or devices."
        ),

        (
            9,
            "Cloud computing delivers servers, storage, databases, "
            "networking, and applications through the internet. "
            "Resources can scale with demand, be accessed remotely, "
            "and reduce the amount of local infrastructure that an "
            "organization must maintain."
        ),

        (
            10,
            "Cloud computing provides computing resources and services "
            "over the internet rather than requiring users to own all "
            "infrastructure locally. Services include servers, storage, "
            "databases, networking, and software. Resources can be "
            "scaled on demand, accessed from many locations, and can "
            "reduce hardware and maintenance costs while improving "
            "flexibility."
        ),
    ],


    # -----------------------------------------------------
    # Q5 — Database Management Systems
    # -----------------------------------------------------

    "Q5": [

        (
            0,
            "Natural language processing enables computers to analyze "
            "human text and speech. It can be used for translation, "
            "sentiment analysis, question answering, chatbots, and "
            "semantic comparison of different pieces of text."
        ),

        (
            1,
            "A Database Management System is primarily graphics "
            "software used to create pictures, animations, diagrams, "
            "and visual effects. Designers use it to draw objects "
            "and edit images rather than to store structured records."
        ),

        (
            2,
            "A DBMS is software that stores information and keeps "
            "records organized."
        ),

        (
            3,
            "A database management system is business software used "
            "in offices, hospitals, schools, banks, stores, and many "
            "other organizations. It is useful because companies "
            "handle a lot of information and computers allow employees "
            "to work with that information more quickly."
        ),

        (
            8,
            "A DBMS stores and organizes structured data, supports "
            "queries and retrieval, and helps maintain accurate, "
            "consistent, and secure records."
        ),

        (
            8,
            "Database management software provides organized storage, "
            "controlled access, efficient retrieval, and support for "
            "multiple users while reducing unnecessary duplication."
        ),

        (
            9,
            "A DBMS manages structured data and provides efficient "
            "storage, querying, retrieval, updates, security, integrity, "
            "and controlled access for multiple users and applications."
        ),

        (
            10,
            "A Database Management System is software used to store, "
            "organize, manage, and retrieve structured data efficiently. "
            "It maintains accuracy, integrity, consistency, and security, "
            "controls user access, supports multiple users, and reduces "
            "unnecessary duplication. DBMSs are widely used in banking, "
            "hospitals, education systems, and e-commerce."
        ),
    ],
}


# =========================================================
# HELPERS
# =========================================================

def normalize_answer(text: str) -> str:
    return " ".join(
        re.findall(
            r"[a-z0-9]+",
            str(text).lower(),
        )
    )


def word_count(text: str) -> int:
    return len(
        re.findall(
            r"[A-Za-z0-9]+",
            str(text),
        )
    )


def safe_corr(frame: pd.DataFrame) -> float:
    if len(frame) < 2:
        return 0.0

    if (
        frame["human_score"].std() == 0
        or frame["word_count"].std() == 0
    ):
        return 0.0

    return float(
        frame[
            [
                "human_score",
                "word_count",
            ]
        ]
        .corr()
        .iloc[0, 1]
    )


# =========================================================
# MAIN
# =========================================================

def main() -> None:

    # -----------------------------------------------------
    # Load human-labelled data
    # -----------------------------------------------------

    base = pd.read_csv(
        SOURCE
    )

    required_columns = [
        "question_id",
        "question_text",
        "model_answer",
        "student_answer",
        "human_score",
        "rubric_keywords",
    ]

    missing = [
        column
        for column in required_columns
        if column not in base.columns
    ]

    if missing:
        raise ValueError(
            "manifest.csv is missing columns: "
            + ", ".join(missing)
        )

    human = base[
        required_columns
    ].copy()

    human["source"] = "human_group"
    human["augmentation_batch"] = "human"


    # -----------------------------------------------------
    # Build synthetic rows
    # -----------------------------------------------------

    rows = []

    for qid in sorted(
        human["question_id"].unique()
    ):

        question_rows = human[
            human["question_id"] == qid
        ]

        if question_rows.empty:
            raise ValueError(
                f"No human question data found for {qid}"
            )

        if qid not in MIDRANGE_EXAMPLES:
            raise ValueError(
                f"No mid-range examples found for {qid}"
            )

        if qid not in ROBUST_EXAMPLES:
            raise ValueError(
                f"No robustness examples found for {qid}"
            )

        first = question_rows.iloc[0]


        # -------------------------------------------------
        # Existing useful 4–7 paraphrases
        # -------------------------------------------------

        for score, answer in MIDRANGE_EXAMPLES[qid]:

            rows.append(
                {
                    "question_id":
                        qid,

                    "question_text":
                        first["question_text"],

                    "model_answer":
                        first["model_answer"],

                    "student_answer":
                        answer,

                    "human_score":
                        float(score),

                    "rubric_keywords":
                        first["rubric_keywords"],

                    "source":
                        "synthetic_augmented",

                    "augmentation_batch":
                        "midrange_balanced_v3",
                }
            )


        # -------------------------------------------------
        # New robustness cases
        # -------------------------------------------------

        for score, answer in ROBUST_EXAMPLES[qid]:

            rows.append(
                {
                    "question_id":
                        qid,

                    "question_text":
                        first["question_text"],

                    "model_answer":
                        first["model_answer"],

                    "student_answer":
                        answer,

                    "human_score":
                        float(score),

                    "rubric_keywords":
                        first["rubric_keywords"],

                    "source":
                        "synthetic_augmented",

                    "augmentation_batch":
                        "robustness_v3",
                }
            )


    synthetic = pd.DataFrame(
        rows
    )


    # -----------------------------------------------------
    # Combine
    # -----------------------------------------------------

    out = pd.concat(
        [
            human,
            synthetic,
        ],
        ignore_index=True,
    )


    # -----------------------------------------------------
    # Validate duplicate labels
    # -----------------------------------------------------

    out["_normalized_answer"] = (
        out["student_answer"]
        .map(
            normalize_answer
        )
    )


    conflicts = (
        out.groupby(
            [
                "question_id",
                "_normalized_answer",
            ]
        )["human_score"]
        .nunique()
    )


    conflicts = conflicts[
        conflicts > 1
    ]


    if len(conflicts):

        print(
            "\nERROR: conflicting duplicate labels detected:"
        )

        print(
            conflicts
        )

        raise ValueError(
            "Dataset contains identical answers "
            "with different scores."
        )


    duplicate_count = int(
        out.duplicated(
            subset=[
                "question_id",
                "_normalized_answer",
            ]
        ).sum()
    )


    out = out.drop(
        columns=[
            "_normalized_answer"
        ]
    )


    # -----------------------------------------------------
    # Stable order
    # -----------------------------------------------------

    out = (
        out
        .sort_values(
            [
                "question_id",
                "source",
                "human_score",
                "augmentation_batch",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    out.to_csv(
        OUTPUT,
        index=False,
    )


    # -----------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------

    diagnostic = out.copy()

    diagnostic["word_count"] = (
        diagnostic[
            "student_answer"
        ]
        .map(
            word_count
        )
    )


    human_diag = diagnostic[
        diagnostic["source"]
        == "human_group"
    ]

    synthetic_diag = diagnostic[
        diagnostic["source"]
        == "synthetic_augmented"
    ]


    print(
        "\n======================================"
    )

    print(
        "BALANCED DATASET BUILT"
    )

    print(
        "======================================"
    )

    print(
        f"\nSaved to:\n{OUTPUT}"
    )

    print(
        f"\nTotal rows:     {len(out)}"
    )

    print(
        f"Human rows:     {len(human_diag)}"
    )

    print(
        f"Synthetic rows: {len(synthetic_diag)}"
    )

    print(
        f"Duplicate rows: {duplicate_count}"
    )


    print(
        "\n======================================"
    )

    print(
        "ROWS BY AUGMENTATION BATCH"
    )

    print(
        "======================================"
    )

    print(
        out[
            "augmentation_batch"
        ]
        .value_counts()
    )


    print(
        "\n======================================"
    )

    print(
        "ROWS PER QUESTION"
    )

    print(
        "======================================"
    )

    print(
        out.groupby(
            "question_id"
        ).size()
    )


    print(
        "\n======================================"
    )

    print(
        "SCORE DISTRIBUTION"
    )

    print(
        "======================================"
    )

    print(
        out[
            "human_score"
        ]
        .value_counts()
        .sort_index()
    )


    print(
        "\n======================================"
    )

    print(
        "WORD COUNT / SCORE CORRELATION"
    )

    print(
        "======================================"
    )

    print(
        f"ALL:            "
        f"{safe_corr(diagnostic):.6f}"
    )

    print(
        f"HUMAN ONLY:     "
        f"{safe_corr(human_diag):.6f}"
    )

    print(
        f"SYNTHETIC ONLY: "
        f"{safe_corr(synthetic_diag):.6f}"
    )


    print(
        "\n======================================"
    )

    print(
        "MEAN WORD COUNT BY SCORE"
    )

    print(
        "======================================"
    )

    print(
        diagnostic.groupby(
            "human_score"
        )[
            "word_count"
        ]
        .agg(
            [
                "mean",
                "min",
                "max",
            ]
        )
        .round(2)
    )


if __name__ == "__main__":
    main()
