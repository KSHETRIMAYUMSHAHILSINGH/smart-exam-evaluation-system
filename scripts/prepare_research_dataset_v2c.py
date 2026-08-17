"""Prepare the cleaned Research Grader V2C dataset.

Why this exists
---------------
The dataset integrity audit found five exact/normalized duplicate-answer
conflicts. In every case, the same synthetic answer was labelled both 7 and 8.

Inspection of scripts/build_training_dataset.py shows why:
    score 7: concepts [0, 1, 2, 3]
    score 8: concepts [0, 1, 2, 3]
for one variant, so identical text can receive two labels.

The old generator also appends the same generic synthesis sentence to score >= 9:
    "Together, these capabilities make it useful in real-world information
     systems and intelligent applications."

That generic sentence is not unique rubric evidence and can teach a grader that
extra prose deserves extra marks. V2C removes it from synthetic training answers.

This script DOES NOT overwrite data/training_dataset.csv.

Outputs
-------
data/training_dataset_v2c.csv
data/v2c_synthetic_train.csv
data/v2c_human_gold.csv
data/v2c_dataset_repair_log.csv
data/rubric_concepts_v2c.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "data" / "training_dataset.csv"
OUTPUT = ROOT / "data" / "training_dataset_v2c.csv"
SYNTHETIC_OUTPUT = ROOT / "data" / "v2c_synthetic_train.csv"
HUMAN_OUTPUT = ROOT / "data" / "v2c_human_gold.csv"
REPAIR_LOG = ROOT / "data" / "v2c_dataset_repair_log.csv"
RUBRIC_OUTPUT = ROOT / "data" / "rubric_concepts_v2c.json"

GENERIC_SYNTHESIS = (
    "Together, these capabilities make it useful in real-world "
    "information systems and intelligent applications."
)


# Atomic rubric concepts are taken from the topic definitions already used by
# the project's synthetic-data generator. V2C will score these individually
# instead of treating the entire reference answer as one similarity target.
RUBRIC_CONCEPTS = {
    "Q1": {
        "name": "Artificial Intelligence",
        "criteria": [
            "Artificial Intelligence builds computer systems that perform tasks associated with human intelligence.",
            "AI systems can learn from examples and improve their behaviour from data.",
            "AI systems can reason and support decision making.",
            "AI systems can solve problems instead of following only one fixed sequence of instructions.",
            "AI systems can understand language and recognize patterns in text, images, or other data.",
            "AI applications include systems such as chatbots, recommendation systems, image recognition, or automated evaluation.",
        ],
    },
    "Q2": {
        "name": "Machine Learning",
        "criteria": [
            "Machine Learning is a branch of AI in which models learn from data or examples.",
            "A machine learning model identifies useful patterns or relationships in training data.",
            "Learned patterns are used to make predictions or decisions on new or unseen inputs.",
            "Machine learning does not require a separate hard-coded rule for every individual case.",
            "Training examples are used to adjust the model so its performance can improve.",
            "Common machine learning tasks include classification, regression, or prediction.",
        ],
    },
    "Q3": {
        "name": "Natural Language Processing",
        "criteria": [
            "Natural Language Processing is an area of AI that enables computers to work with human language.",
            "NLP processes or analyzes human-written or spoken language.",
            "NLP can perform tasks such as text classification or sentiment analysis.",
            "NLP can be used for machine translation or question answering.",
            "Semantic representations can be used to measure text similarity and meaning.",
            "NLP can support automated grading by comparing a student's response with a reference answer.",
        ],
    },
    "Q4": {
        "name": "Cloud Computing",
        "criteria": [
            "Cloud computing delivers computing resources or services over the internet.",
            "Cloud services can include storage, databases, servers, networking, or software.",
            "Users can access cloud resources without maintaining all physical hardware themselves.",
            "Cloud resources can be scaled up or down according to demand.",
            "Cloud computing improves flexibility or accessibility from different locations or devices.",
            "Cloud computing can reduce infrastructure or maintenance costs for organizations.",
        ],
    },
    "Q5": {
        "name": "Database Management Systems",
        "criteria": [
            "A Database Management System is software for storing, organizing, managing, and retrieving data.",
            "A DBMS supports efficient storage, organization, querying, or retrieval of records.",
            "A DBMS helps maintain data accuracy, integrity, or consistency.",
            "A DBMS provides security or controlled access to information.",
            "A DBMS can support multiple users or applications while reducing unnecessary duplication of data.",
            "DBMSs are used in applications such as banking, hospitals, education systems, or e-commerce.",
        ],
    },
}


def clean_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_answer(text: str) -> str:
    text = clean_text(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def remove_generic_synthesis(text: str) -> tuple[str, bool]:
    text = clean_text(text)

    # Exact normal form first.
    if GENERIC_SYNTHESIS in text:
        repaired = text.replace(GENERIC_SYNTHESIS, " ")
        repaired = re.sub(r"\s+", " ", repaired).strip()
        return repaired, True

    # Tolerate spacing/punctuation changes.
    pattern = re.compile(
        r"\s*Together,\s+these\s+capabilities\s+make\s+it\s+useful\s+"
        r"in\s+real-world\s+information\s+systems\s+and\s+intelligent\s+"
        r"applications\.?\s*$",
        flags=re.IGNORECASE,
    )
    repaired, count = pattern.subn("", text)
    repaired = re.sub(r"\s+", " ", repaired).strip()

    return repaired, count > 0


def find_conflicting_groups(df: pd.DataFrame) -> list[pd.DataFrame]:
    work = df.copy()
    work["_normalized_answer"] = work["student_answer"].map(normalize_answer)

    conflicts: list[pd.DataFrame] = []

    for _, group in work.groupby(
        ["question_id", "_normalized_answer"],
        sort=False,
        dropna=False,
    ):
        if len(group) < 2:
            continue

        scores = group["human_score"].astype(float).unique()
        if len(scores) > 1:
            conflicts.append(group.copy())

    return conflicts


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing dataset: {SOURCE}")

    df = pd.read_csv(SOURCE)

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
            f"training_dataset.csv is missing required columns: {sorted(missing)}"
        )

    df = df.copy()
    df["student_answer"] = df["student_answer"].map(clean_text)
    df["human_score"] = pd.to_numeric(
        df["human_score"],
        errors="raise",
    ).astype(float)

    print("=" * 76)
    print("RESEARCH GRADER V2C — DATASET PREPARATION")
    print("=" * 76)
    print(f"Source rows: {len(df)}")

    repair_records: list[dict[str, object]] = []

    # ------------------------------------------------------------------
    # 1. Remove generic score-9 filler from synthetic answers only.
    # ------------------------------------------------------------------
    synthesis_removed = 0

    for idx in df.index:
        if str(df.at[idx, "source"]) != "synthetic_augmented":
            continue

        old_answer = clean_text(df.at[idx, "student_answer"])
        new_answer, changed = remove_generic_synthesis(old_answer)

        if not changed:
            continue

        df.at[idx, "student_answer"] = new_answer
        synthesis_removed += 1

        repair_records.append(
            {
                "action": "remove_generic_synthesis",
                "question_id": df.at[idx, "question_id"],
                "source": df.at[idx, "source"],
                "old_score": float(df.at[idx, "human_score"]),
                "new_score": float(df.at[idx, "human_score"]),
                "old_answer": old_answer,
                "new_answer": new_answer,
                "reason": (
                    "Removed generic non-rubric filler that should not "
                    "increase evidence quality."
                ),
            }
        )

    print(f"Generic synthesis sentences removed: {synthesis_removed}")

    # ------------------------------------------------------------------
    # 2. Resolve exact synthetic label conflicts conservatively.
    #
    # The audit showed identical synthetic content labelled 7 and 8.
    # Because the higher label adds no evidence, keep the LOWER label and
    # drop the duplicate higher-labelled row.
    # ------------------------------------------------------------------
    conflicts = find_conflicting_groups(df)
    print(f"Conflicting duplicate groups before repair: {len(conflicts)}")

    rows_to_drop: set[int] = set()

    for group in conflicts:
        sources = set(group["source"].astype(str))

        if sources != {"synthetic_augmented"}:
            raise RuntimeError(
                "Found a conflicting duplicate group containing non-synthetic "
                "data. Refusing automatic repair; inspect manually."
            )

        scores = group["human_score"].astype(float)
        keep_score = float(scores.min())

        # Keep the first row at the conservative/lower score.
        keep_candidates = group[scores == keep_score]
        keep_idx = int(keep_candidates.index[0])

        for idx in group.index:
            idx = int(idx)
            if idx == keep_idx:
                continue

            rows_to_drop.add(idx)

            repair_records.append(
                {
                    "action": "drop_conflicting_duplicate",
                    "question_id": group.at[idx, "question_id"],
                    "source": group.at[idx, "source"],
                    "old_score": float(group.at[idx, "human_score"]),
                    "new_score": keep_score,
                    "old_answer": clean_text(group.at[idx, "student_answer"]),
                    "new_answer": clean_text(group.at[keep_idx, "student_answer"]),
                    "reason": (
                        "Identical synthetic answer had multiple labels; "
                        "kept the conservative lower score because the higher "
                        "score contained no additional rubric evidence."
                    ),
                }
            )

    if rows_to_drop:
        df = df.drop(index=sorted(rows_to_drop)).reset_index(drop=True)

    print(f"Conflicting duplicate rows dropped: {len(rows_to_drop)}")

    # ------------------------------------------------------------------
    # 3. Validate repaired dataset.
    # ------------------------------------------------------------------
    remaining_conflicts = find_conflicting_groups(df)

    if remaining_conflicts:
        raise RuntimeError(
            f"Repair failed: {len(remaining_conflicts)} conflicting duplicate "
            "groups remain."
        )

    if any(
        GENERIC_SYNTHESIS.lower()
        in clean_text(answer).lower()
        for answer in df.loc[
            df["source"].astype(str).eq("synthetic_augmented"),
            "student_answer",
        ]
    ):
        raise RuntimeError(
            "Repair failed: generic synthesis sentence still exists "
            "in synthetic answers."
        )

    unknown_questions = sorted(
        set(df["question_id"].astype(str)) - set(RUBRIC_CONCEPTS)
    )
    if unknown_questions:
        raise RuntimeError(
            f"Missing V2C rubric concepts for questions: {unknown_questions}"
        )

    # Stable output ordering.
    sort_columns = ["question_id", "source", "human_score"]
    df = df.sort_values(sort_columns).reset_index(drop=True)

    synthetic = df[
        df["source"].astype(str).eq("synthetic_augmented")
    ].copy()

    human = df[
        df["source"].astype(str).eq("human_group")
    ].copy()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT, index=False)
    synthetic.to_csv(SYNTHETIC_OUTPUT, index=False)
    human.to_csv(HUMAN_OUTPUT, index=False)

    repair_df = pd.DataFrame(repair_records)
    repair_df.to_csv(REPAIR_LOG, index=False)

    with RUBRIC_OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(RUBRIC_CONCEPTS, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # 4. Report.
    # ------------------------------------------------------------------
    print("\n## Repaired dataset")
    print(f"Rows:       {len(df)}")
    print(f"Synthetic:  {len(synthetic)}")
    print(f"Human gold: {len(human)}")
    print(f"Questions:  {df['question_id'].nunique()}")

    print("\nRows per question:")
    print(df.groupby("question_id").size().to_string())

    print("\nScore distribution:")
    print(df["human_score"].value_counts().sort_index().to_string())

    print("\nHuman-gold rows per question:")
    print(human.groupby("question_id").size().to_string())

    print("\n## Integrity checks")
    print("Conflicting exact/normalized duplicates: 0")
    print("Generic score-9 synthesis filler:        0")
    print(f"Atomic rubric criteria per question:     6")

    print("\n## Saved")
    print(f"Clean V2C dataset: {OUTPUT}")
    print(f"Synthetic train:   {SYNTHETIC_OUTPUT}")
    print(f"Human gold:        {HUMAN_OUTPUT}")
    print(f"Repair log:        {REPAIR_LOG}")
    print(f"Rubric concepts:   {RUBRIC_OUTPUT}")

    print(
        "\nOriginal data/training_dataset.csv was NOT modified."
    )


if __name__ == "__main__":
    main()
