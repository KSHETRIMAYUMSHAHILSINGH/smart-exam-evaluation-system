"""Audit the ML Assisted Grading training dataset for label conflicts and near-duplicates.

This script DOES NOT modify training_dataset.csv.

Checks
------
1. Exact duplicate student answers within the same question.
2. Normalized duplicates (case/spacing/punctuation normalized).
3. Conflicting labels for duplicate answers.
4. Near-duplicate answers with large score gaps using sentence embeddings.
5. Per-question/source/score distributions.

Outputs
-------
data/dataset_audit_exact_duplicates.csv
data/dataset_audit_conflicts.csv
data/dataset_audit_near_duplicates.csv
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "training_dataset.csv"

EXACT_OUT = ROOT / "data" / "dataset_audit_exact_duplicates.csv"
CONFLICT_OUT = ROOT / "data" / "dataset_audit_conflicts.csv"
NEAR_OUT = ROOT / "data" / "dataset_audit_near_duplicates.csv"

ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
NEAR_DUPLICATE_THRESHOLD = 0.92
MIN_SCORE_GAP = 1.0


def clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_answer(text: str) -> str:
    text = clean_text(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_duplicate_report(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    work["_normalized_answer"] = work["student_answer"].map(normalize_answer)

    rows = []
    conflict_rows = []

    grouped = work.groupby(
        ["question_id", "_normalized_answer"],
        dropna=False,
        sort=False,
    )

    for (qid, normalized), group in grouped:
        if not normalized or len(group) < 2:
            continue

        scores = sorted(group["human_score"].astype(float).unique().tolist())
        score_min = float(min(scores))
        score_max = float(max(scores))
        score_gap = score_max - score_min

        record = {
            "question_id": qid,
            "count": int(len(group)),
            "scores": " | ".join(f"{s:g}" for s in scores),
            "score_min": score_min,
            "score_max": score_max,
            "score_gap": score_gap,
            "sources": " | ".join(
                sorted(
                    group.get(
                        "source",
                        pd.Series(["unknown"] * len(group)),
                    )
                    .fillna("unknown")
                    .astype(str)
                    .unique()
                    .tolist()
                )
            ),
            "student_answer": clean_text(group.iloc[0]["student_answer"]),
        }
        rows.append(record)

        if score_gap > 0:
            conflict_rows.append(record)

    duplicates = pd.DataFrame(rows)
    conflicts = pd.DataFrame(conflict_rows)

    if not duplicates.empty:
        duplicates = duplicates.sort_values(
            ["score_gap", "count"],
            ascending=[False, False],
        ).reset_index(drop=True)

    if not conflicts.empty:
        conflicts = conflicts.sort_values(
            ["score_gap", "count"],
            ascending=[False, False],
        ).reset_index(drop=True)

    return duplicates, conflicts


def near_duplicate_report(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\nLoading encoder for near-duplicate audit: {ENCODER_NAME}")
    model = SentenceTransformer(ENCODER_NAME)

    rows = []

    for qid, group in df.groupby("question_id", sort=False):
        group = group.reset_index(drop=True)
        answers = group["student_answer"].map(clean_text).tolist()

        if len(answers) < 2:
            continue

        vectors = model.encode(
            answers,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        similarity = vectors @ vectors.T

        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                sim = float(similarity[i, j])
                if sim < NEAR_DUPLICATE_THRESHOLD:
                    continue

                score_a = float(group.loc[i, "human_score"])
                score_b = float(group.loc[j, "human_score"])
                gap = abs(score_a - score_b)

                if gap < MIN_SCORE_GAP:
                    continue

                answer_a = answers[i]
                answer_b = answers[j]

                # Exact normalized conflicts are already reported separately.
                if normalize_answer(answer_a) == normalize_answer(answer_b):
                    continue

                rows.append(
                    {
                        "question_id": qid,
                        "similarity": sim,
                        "score_a": score_a,
                        "score_b": score_b,
                        "score_gap": gap,
                        "source_a": str(group.loc[i].get("source", "unknown")),
                        "source_b": str(group.loc[j].get("source", "unknown")),
                        "answer_a": answer_a,
                        "answer_b": answer_b,
                    }
                )

    out = pd.DataFrame(rows)

    if not out.empty:
        out = out.sort_values(
            ["score_gap", "similarity"],
            ascending=[False, False],
        ).reset_index(drop=True)

    return out


def print_conflicts(conflicts: pd.DataFrame, limit: int = 20) -> None:
    print("\n## Conflicting exact/normalized duplicate labels")

    if conflicts.empty:
        print("None found.")
        return

    print(f"Found {len(conflicts)} conflicting duplicate answer groups.")

    for row in conflicts.head(limit).itertuples(index=False):
        print("\n" + "-" * 72)
        print(f"Question: {row.question_id}")
        print(f"Occurrences: {row.count}")
        print(f"Scores: {row.scores}")
        print(f"Score gap: {row.score_gap:.1f}")
        print(f"Sources: {row.sources}")
        print(f"Answer: {row.student_answer}")


def print_near_duplicates(near: pd.DataFrame, limit: int = 20) -> None:
    print("\n## Near-duplicate answers with score gaps")

    if near.empty:
        print(
            f"None found at similarity >= {NEAR_DUPLICATE_THRESHOLD:.2f} "
            f"and score gap >= {MIN_SCORE_GAP:.1f}."
        )
        return

    print(
        f"Found {len(near)} pairs at similarity >= "
        f"{NEAR_DUPLICATE_THRESHOLD:.2f} with score gap >= {MIN_SCORE_GAP:.1f}."
    )

    for row in near.head(limit).itertuples(index=False):
        print("\n" + "-" * 72)
        print(f"Question: {row.question_id}")
        print(f"Similarity: {row.similarity:.3f}")
        print(f"Scores: {row.score_a:g} vs {row.score_b:g}")
        print(f"Gap: {row.score_gap:g}")
        print(f"A: {row.answer_a}")
        print(f"B: {row.answer_b}")


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required = {
        "question_id",
        "student_answer",
        "human_score",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"training_dataset.csv is missing columns: {sorted(missing)}"
        )

    df = df.dropna(subset=["student_answer", "human_score"]).copy()
    df["student_answer"] = df["student_answer"].map(clean_text)
    df["human_score"] = pd.to_numeric(
        df["human_score"],
        errors="raise",
    ).astype(float)

    print("=" * 76)
    print("ML ASSISTED GRADING — DATASET INTEGRITY AUDIT")
    print("=" * 76)
    print(f"Dataset:   {DATA_PATH}")
    print(f"Rows:      {len(df)}")
    print(f"Questions: {df['question_id'].nunique()}")

    if "source" in df.columns:
        print("\n## Sources")
        print(df["source"].fillna("unknown").value_counts().to_string())

    print("\n## Score distribution")
    print(df["human_score"].value_counts().sort_index().to_string())

    print("\n## Rows per question")
    print(df.groupby("question_id").size().to_string())

    duplicates, conflicts = exact_duplicate_report(df)

    EXACT_OUT.parent.mkdir(parents=True, exist_ok=True)
    duplicates.to_csv(EXACT_OUT, index=False)
    conflicts.to_csv(CONFLICT_OUT, index=False)

    print("\n## Exact/normalized duplicate summary")
    print(f"Duplicate answer groups:             {len(duplicates)}")
    print(f"Conflicting duplicate label groups: {len(conflicts)}")

    if not conflicts.empty:
        print(
            "Largest duplicate-label score gap: "
            f"{conflicts['score_gap'].max():.1f}"
        )

    print_conflicts(conflicts)

    near = near_duplicate_report(df)
    near.to_csv(NEAR_OUT, index=False)
    print_near_duplicates(near)

    print("\n## Saved")
    print(f"Exact duplicates: {EXACT_OUT}")
    print(f"Conflicts:        {CONFLICT_OUT}")
    print(f"Near duplicates:  {NEAR_OUT}")

    print(
        "\nIMPORTANT: This audit does not modify the training dataset."
    )


if __name__ == "__main__":
    main()
