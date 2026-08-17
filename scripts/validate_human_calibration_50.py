"""Validate the 50-row human calibration dataset.

This script only READS:
    data/human_calibration_50.csv

It does NOT modify or merge any dataset.

Checks
------
- exactly 50 rows
- 10 rows per question
- student_answer filled
- human_score numeric and between 0 and 10
- duplicate answers
- score distribution overall and per question
- coverage of low/mid scores 0-6
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "data" / "human_calibration_50.csv"


def main():
    if not PATH.exists():
        raise FileNotFoundError(PATH)

    df = pd.read_csv(PATH)

    print("=" * 72)
    print("HUMAN CALIBRATION VALIDATION")
    print("=" * 72)

    print(f"\nRows: {len(df)}")

    required = [
        "calibration_id",
        "question_id",
        "student_answer",
        "human_score",
        "source",
    ]

    missing_cols = [c for c in required if c not in df.columns]

    if missing_cols:
        print(f"\nFAIL: missing columns: {missing_cols}")
        return

    # Answers
    answers = df["student_answer"].fillna("").astype(str).str.strip()
    blank_answers = int((answers == "").sum())

    # Scores
    scores = pd.to_numeric(
        df["human_score"],
        errors="coerce",
    )

    missing_scores = int(scores.isna().sum())
    invalid_scores = int(
        ((scores < 0) | (scores > 10)).fillna(False).sum()
    )

    print(f"Blank answers: {blank_answers}")
    print(f"Missing/non-numeric scores: {missing_scores}")
    print(f"Scores outside 0-10: {invalid_scores}")

    print("\nRows per question:")
    print(
        df["question_id"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    if missing_scores == 0:
        df["human_score_numeric"] = scores

        print("\nOverall score distribution:")
        print(
            df["human_score_numeric"]
            .value_counts()
            .sort_index()
            .to_string()
        )

        print("\nScore-range coverage:")
        ranges = pd.cut(
            df["human_score_numeric"],
            bins=[-0.01, 2, 4, 6, 10],
            labels=["0-2", "3-4", "5-6", "7-10"],
        )

        print(
            ranges.value_counts()
            .reindex(["0-2", "3-4", "5-6", "7-10"])
            .fillna(0)
            .astype(int)
            .to_string()
        )

        print("\nPer-question score-range coverage:")
        temp = df.copy()
        temp["range"] = ranges

        table = (
            temp.groupby(["question_id", "range"], observed=False)
            .size()
            .unstack(fill_value=0)
            .reindex(columns=["0-2", "3-4", "5-6", "7-10"])
        )

        print(table.to_string())

    normalized = (
        answers
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )

    duplicate_mask = (
        (normalized != "")
        & normalized.duplicated(keep=False)
    )

    duplicate_count = int(duplicate_mask.sum())

    print(f"\nDuplicate-answer rows: {duplicate_count}")

    if duplicate_count:
        print(
            df.loc[
                duplicate_mask,
                [
                    "calibration_id",
                    "question_id",
                    "student_answer",
                    "human_score",
                ],
            ]
            .sort_values("student_answer")
            .to_string(index=False)
        )

    rows_ok = len(df) == 50
    per_q_ok = (
        df["question_id"]
        .value_counts()
        .eq(10)
        .all()
        and df["question_id"].nunique() == 5
    )

    complete = (
        rows_ok
        and per_q_ok
        and blank_answers == 0
        and missing_scores == 0
        and invalid_scores == 0
    )

    print("\n" + "=" * 72)

    if complete:
        print("VALIDATION RESULT: READY")
        print(
            "The file is structurally complete and ready for the next review."
        )
    else:
        print("VALIDATION RESULT: NOT READY")
        print(
            "Finish the missing/invalid entries before merging or training."
        )

    print("=" * 72)


if __name__ == "__main__":
    main()
