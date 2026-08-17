"""
Add targeted length-invariance examples to the existing training dataset.

Starting dataset:
    140 rows
    - 20 human_group
    - 120 synthetic_augmented

Adds:
    20 counterfactual length-invariance examples
    - 4 per question

Final:
    160 rows

Purpose:
Teach the grader that changing answer length alone should not
change the mark.

Important:
The original dataset already contains a small number of known
conflicting duplicate answers. Those are handled later by the
training-data cleaning pipeline.

This script therefore allows PRE-EXISTING conflicts, but refuses:
    - any NEW conflicting duplicate caused by v4
    - any v4 answer that exactly duplicates an existing answer

Run:
    python scripts/add_length_invariance_v4.py
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT
    / "data"
    / "training_dataset.csv"
)

BACKUP_PATH = (
    ROOT
    / "data"
    / "training_dataset_before_length_invariance_v4.csv"
)

BATCH_NAME = "length_invariance_v4"


# =========================================================
# COUNTERFACTUAL EXAMPLES
# =========================================================

EXAMPLES = {

    # =====================================================
    # Q1 — Artificial Intelligence
    # =====================================================

    "Q1": [

        (
            3,
            "Artificial intelligence is computer technology that "
            "helps automate tasks and makes software act smarter."
        ),

        (
            3,
            "Artificial intelligence is computer technology that "
            "helps automate tasks and makes software act smarter. "
            "Computers are also used in schools, offices, banks, "
            "hospitals, shops, and homes. Modern devices can store "
            "large amounts of information, connect to the internet, "
            "and run many different applications."
        ),

        (
            8,
            "Artificial intelligence enables computers to learn from "
            "data, recognize patterns, make decisions, reason, and "
            "solve problems."
        ),

        (
            8,
            "Artificial intelligence enables computers to learn from "
            "data, recognize patterns, make decisions, reason, and "
            "solve problems. Computers are also widely used for "
            "documents, communication, file storage, websites, and "
            "general office work. These unrelated uses do not change "
            "the main capabilities that define artificial intelligence."
        ),
    ],


    # =====================================================
    # Q2 — Machine Learning
    # =====================================================

    "Q2": [

        (
            3,
            "Machine learning is technology that helps computer "
            "programs improve after receiving data."
        ),

        (
            3,
            "Machine learning is technology that helps computer "
            "programs improve after receiving data. Computers are "
            "used in businesses, schools, hospitals, websites, mobile "
            "phones, and many other places. They can store files, "
            "perform calculations, and connect users through networks."
        ),

        (
            8,
            "Machine learning trains models on examples so they can "
            "discover patterns and make predictions or decisions on "
            "new data without a separate hard-coded rule for every case."
        ),

        (
            8,
            "Machine learning trains models on examples so they can "
            "discover patterns and make predictions or decisions on "
            "new data without a separate hard-coded rule for every case. "
            "Computers can also create documents, display websites, "
            "store files, and communicate through networks. Those "
            "general computer functions are separate from the learning "
            "process itself."
        ),
    ],


    # =====================================================
    # Q3 — Natural Language Processing
    # =====================================================

    "Q3": [

        (
            3,
            "Natural language processing is technology that helps "
            "computers work with words and text."
        ),

        (
            3,
            "Natural language processing is technology that helps "
            "computers work with words and text. Computers are also "
            "used to store files, browse websites, run applications, "
            "play media, and connect to online services. These are "
            "common functions of modern computer systems."
        ),

        (
            8,
            "Natural language processing enables computers to analyze "
            "human text or speech for tasks such as classification, "
            "sentiment analysis, translation, question answering, and "
            "semantic comparison."
        ),

        (
            8,
            "Natural language processing enables computers to analyze "
            "human text or speech for tasks such as classification, "
            "sentiment analysis, translation, question answering, and "
            "semantic comparison. Computers also perform many unrelated "
            "tasks such as file storage, networking, calculations, and "
            "displaying multimedia content."
        ),
    ],


    # =====================================================
    # Q4 — Cloud Computing
    # =====================================================

    "Q4": [

        (
            3,
            "Cloud computing means using computer services through "
            "the internet instead of only using one local computer."
        ),

        (
            3,
            "Cloud computing means using computer services through "
            "the internet instead of only using one local computer. "
            "Modern computers are found in schools, homes, offices, "
            "shops, hospitals, and many other places. They can process "
            "documents, display media, and run different applications."
        ),

        (
            8,
            "Cloud computing provides servers, storage, databases, "
            "networking, and software over the internet while allowing "
            "resources to scale according to demand."
        ),

        (
            8,
            "Cloud computing provides servers, storage, databases, "
            "networking, and software over the internet while allowing "
            "resources to scale according to demand. Computers are "
            "also used for ordinary local tasks such as editing files, "
            "viewing media, and running desktop applications, which "
            "are not what defines cloud computing."
        ),
    ],


    # =====================================================
    # Q5 — Database Management Systems
    # =====================================================

    "Q5": [

        (
            3,
            "A DBMS is software that stores information and helps "
            "users organize records."
        ),

        (
            3,
            "A DBMS is software that stores information and helps "
            "users organize records. Computers are also used for "
            "email, websites, presentations, calculations, multimedia, "
            "and communication. Organizations use many different "
            "software programs for their daily activities."
        ),

        (
            8,
            "A DBMS stores and organizes structured data, supports "
            "queries and retrieval, controls access, and helps maintain "
            "accurate and consistent records."
        ),

        (
            8,
            "A DBMS stores and organizes structured data, supports "
            "queries and retrieval, controls access, and helps maintain "
            "accurate and consistent records. Organizations also use "
            "word processors, browsers, presentation software, and "
            "communication tools, but those applications are separate "
            "from the database management functions."
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


def safe_corr(
    frame: pd.DataFrame,
) -> float:

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


def conflict_keys(
    frame: pd.DataFrame,
) -> set[tuple]:

    temp = frame.copy()

    temp["_normalized_answer"] = (
        temp[
            "student_answer"
        ]
        .map(
            normalize_answer
        )
    )

    conflicts = (
        temp.groupby(
            [
                "question_id",
                "_normalized_answer",
            ],
            dropna=False,
        )[
            "human_score"
        ]
        .nunique()
    )

    conflicts = conflicts[
        conflicts > 1
    ]

    return set(
        conflicts.index.tolist()
    )


# =========================================================
# MAIN
# =========================================================

def main() -> None:

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Missing dataset: {DATA_PATH}"
        )


    # -----------------------------------------------------
    # Load current dataset
    # -----------------------------------------------------

    df = pd.read_csv(
        DATA_PATH
    )


    print(
        "\nStarting rows:",
        len(df),
    )


    # -----------------------------------------------------
    # Create backup if one does not already exist
    # -----------------------------------------------------

    if not BACKUP_PATH.exists():

        df.to_csv(
            BACKUP_PATH,
            index=False,
        )

        print(
            "Created backup:"
        )

        print(
            BACKUP_PATH
        )

    else:

        print(
            "Backup already exists:"
        )

        print(
            BACKUP_PATH
        )


    # -----------------------------------------------------
    # Ensure augmentation column exists
    # -----------------------------------------------------

    if (
        "augmentation_batch"
        not in df.columns
    ):

        df[
            "augmentation_batch"
        ] = ""


    # -----------------------------------------------------
    # Remove old v4 rows if script is rerun
    # -----------------------------------------------------

    old_v4_mask = (
        df[
            "augmentation_batch"
        ]
        .fillna("")
        .eq(
            BATCH_NAME
        )
    )


    old_v4_count = int(
        old_v4_mask.sum()
    )


    if old_v4_count:

        df = (
            df[
                ~old_v4_mask
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        print(
            f"Removed {old_v4_count} "
            f"previous {BATCH_NAME} rows."
        )


    # -----------------------------------------------------
    # Record PRE-EXISTING conflicts
    # -----------------------------------------------------

    baseline_conflicts = (
        conflict_keys(
            df
        )
    )


    print(
        "\nPre-existing conflicting "
        f"duplicate groups: "
        f"{len(baseline_conflicts)}"
    )


    # -----------------------------------------------------
    # Diagnostics before augmentation
    # -----------------------------------------------------

    before = df.copy()

    before[
        "word_count"
    ] = (
        before[
            "student_answer"
        ]
        .map(
            word_count
        )
    )


    before_synthetic = (
        before[
            before["source"]
            == "synthetic_augmented"
        ]
    )


    before_corr = safe_corr(
        before_synthetic
    )


    # -----------------------------------------------------
    # Build new v4 rows
    # -----------------------------------------------------

    rows = []


    for (
        qid,
        examples,
    ) in EXAMPLES.items():

        qdf = df[
            df["question_id"]
            == qid
        ]


        if qdf.empty:

            raise ValueError(
                f"Question {qid} "
                f"not found in dataset."
            )


        first = qdf.iloc[0]


        for (
            score,
            answer,
        ) in examples:

            rows.append(
                {
                    "question_id":
                        qid,

                    "question_text":
                        first[
                            "question_text"
                        ],

                    "model_answer":
                        first[
                            "model_answer"
                        ],

                    "student_answer":
                        answer,

                    "human_score":
                        float(score),

                    "rubric_keywords":
                        first[
                            "rubric_keywords"
                        ],

                    "source":
                        "synthetic_augmented",

                    "augmentation_batch":
                        BATCH_NAME,
                }
            )


    extra = pd.DataFrame(
        rows
    )


    # -----------------------------------------------------
    # Check v4 for internal duplicate answers
    # -----------------------------------------------------

    extra[
        "_normalized_answer"
    ] = (
        extra[
            "student_answer"
        ]
        .map(
            normalize_answer
        )
    )


    v4_internal_duplicates = (
        extra.duplicated(
            subset=[
                "question_id",
                "_normalized_answer",
            ],
            keep=False,
        )
    )


    if v4_internal_duplicates.any():

        print(
            "\nERROR:"
        )

        print(
            "Duplicate answers exist "
            "inside v4:"
        )

        print(
            extra.loc[
                v4_internal_duplicates,
                [
                    "question_id",
                    "human_score",
                    "student_answer",
                ],
            ]
        )

        raise ValueError(
            "Duplicate v4 examples detected."
        )


    # -----------------------------------------------------
    # Ensure v4 does not duplicate existing rows
    # -----------------------------------------------------

    existing = df.copy()

    existing[
        "_normalized_answer"
    ] = (
        existing[
            "student_answer"
        ]
        .map(
            normalize_answer
        )
    )


    existing_keys = set(
        zip(
            existing[
                "question_id"
            ],
            existing[
                "_normalized_answer"
            ],
        )
    )


    v4_keys = set(
        zip(
            extra[
                "question_id"
            ],
            extra[
                "_normalized_answer"
            ],
        )
    )


    overlap = (
        existing_keys
        & v4_keys
    )


    if overlap:

        print(
            "\nERROR:"
        )

        print(
            "Some v4 answers already exist "
            "in the original dataset:"
        )

        for key in sorted(
            overlap
        ):

            print(
                key
            )

        raise ValueError(
            "v4 contains exact duplicates "
            "of existing training answers."
        )


    extra = extra.drop(
        columns=[
            "_normalized_answer"
        ]
    )


    # -----------------------------------------------------
    # Combine
    # -----------------------------------------------------

    out = pd.concat(
        [
            df,
            extra,
        ],
        ignore_index=True,
    )


    # -----------------------------------------------------
    # Detect only NEW conflicts
    # -----------------------------------------------------

    final_conflicts = (
        conflict_keys(
            out
        )
    )


    new_conflicts = (
        final_conflicts
        - baseline_conflicts
    )


    if new_conflicts:

        print(
            "\nERROR:"
        )

        print(
            "v4 introduced NEW conflicting "
            "duplicate labels:"
        )

        for key in sorted(
            new_conflicts
        ):

            print(
                key
            )

        raise ValueError(
            "Length-invariance v4 introduced "
            "new conflicting duplicates."
        )


    # -----------------------------------------------------
    # Save only after validation succeeds
    # -----------------------------------------------------

    out.to_csv(
        DATA_PATH,
        index=False,
    )


    # -----------------------------------------------------
    # Diagnostics after augmentation
    # -----------------------------------------------------

    diagnostic = (
        out.copy()
    )


    diagnostic[
        "word_count"
    ] = (
        diagnostic[
            "student_answer"
        ]
        .map(
            word_count
        )
    )


    human = (
        diagnostic[
            diagnostic["source"]
            == "human_group"
        ]
    )


    synthetic = (
        diagnostic[
            diagnostic["source"]
            == "synthetic_augmented"
        ]
    )


    v4 = (
        diagnostic[
            diagnostic[
                "augmentation_batch"
            ]
            .fillna("")
            .eq(
                BATCH_NAME
            )
        ]
    )


    print(
        "\n"
        "======================================"
    )

    print(
        "LENGTH-INVARIANCE V4 COMPLETE"
    )

    print(
        "======================================"
    )


    print(
        f"\nAdded rows:     "
        f"{len(extra)}"
    )

    print(
        f"Total rows:     "
        f"{len(out)}"
    )

    print(
        f"Human rows:     "
        f"{len(human)}"
    )

    print(
        f"Synthetic rows: "
        f"{len(synthetic)}"
    )

    print(
        f"V4 rows:        "
        f"{len(v4)}"
    )

    print(
        f"Pre-existing conflicting groups: "
        f"{len(baseline_conflicts)}"
    )

    print(
        f"New conflicting groups: "
        f"{len(new_conflicts)}"
    )


    print(
        "\n"
        "======================================"
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
        "\n"
        "======================================"
    )

    print(
        "LENGTH / SCORE CORRELATION"
    )

    print(
        "======================================"
    )


    print(
        f"Original synthetic: "
        f"{before_corr:.6f}"
    )

    print(
        f"New synthetic:      "
        f"{safe_corr(synthetic):.6f}"
    )

    print(
        f"Human only:         "
        f"{safe_corr(human):.6f}"
    )

    print(
        f"All rows:           "
        f"{safe_corr(diagnostic):.6f}"
    )


    print(
        "\n"
        "======================================"
    )

    print(
        "V4 PAIRS"
    )

    print(
        "======================================"
    )


    print(
        v4[
            [
                "question_id",
                "human_score",
                "word_count",
                "student_answer",
            ]
        ]
        .to_string(
            index=False
        )
    )


    print(
        "\nSaved:"
    )

    print(
        DATA_PATH
    )


if __name__ == "__main__":
    main()
