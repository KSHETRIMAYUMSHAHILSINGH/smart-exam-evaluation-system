"""Prepare a 50-row human calibration template for the grading project.

Creates 10 blank human-response slots per question (Q1-Q5).

IMPORTANT
---------
- Does NOT modify training_dataset_v2c.csv.
- Does NOT train any model.
- Does NOT generate synthetic student answers.
- Student answers and human scores must be filled manually.
- Intended for real low/mid-score human calibration data.

Inputs
------
data/training_dataset_v2c.csv
data/rubric_concepts_v2c.json

Outputs
-------
data/human_calibration_50.csv
data/human_calibration_instructions.txt
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "training_dataset_v2c.csv"
RUBRIC_PATH = ROOT / "data" / "rubric_concepts_v2c.json"

OUT_CSV = ROOT / "data" / "human_calibration_50.csv"
OUT_TXT = ROOT / "data" / "human_calibration_instructions.txt"

ROWS_PER_QUESTION = 10


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    if not RUBRIC_PATH.exists():
        raise FileNotFoundError(RUBRIC_PATH)

    df = pd.read_csv(DATA_PATH)

    with RUBRIC_PATH.open("r", encoding="utf-8") as f:
        rubrics = json.load(f)

    rows = []

    for qid in sorted(df["question_id"].astype(str).unique()):
        qdf = df[df["question_id"].astype(str) == qid]

        question_text = str(qdf.iloc[0]["question_text"])
        model_answer = str(qdf.iloc[0]["model_answer"])

        criteria = rubrics[qid]["criteria"]

        rubric_text = " | ".join(
            f"C{i+1}: {criterion}"
            for i, criterion in enumerate(criteria)
        )

        for i in range(1, ROWS_PER_QUESTION + 1):
            rows.append({
                "calibration_id": f"{qid}_H{i:02d}",
                "question_id": qid,
                "question_text": question_text,
                "model_answer": model_answer,
                "rubric_criteria": rubric_text,
                "student_answer": "",
                "human_score": "",
                "grader_notes": "",
                "source": "human_calibration",
            })

    out = pd.DataFrame(rows)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    instructions = """HUMAN CALIBRATION DATA — COLLECTION GUIDE

Goal
----
Collect 50 genuinely human-written answers:
10 answers for each of Q1-Q5.

Important
---------
Do NOT generate these answers with ChatGPT or another LLM.
We need real human wording, mistakes, omissions, and partial understanding.

Collection
----------
1. Give the question to a person.
2. Let them answer naturally without showing the model answer or rubric.
3. Paste their answer into student_answer.
4. Grade it manually from 0 to 10 using the rubric.
5. Put the score in human_score.
6. Optionally explain the score briefly in grader_notes.

Desired final coverage
----------------------
The priority is real answers scoring 0-6.

For EACH question, try to obtain a spread roughly like:
- 0-2 marks: at least 3 answers
- 3-4 marks: at least 3 answers
- 5-6 marks: at least 3 answers
- any remaining slot: natural answer of any score

Do not force a person to deliberately write a particular score.
Collect naturally first, then recruit more responses for score ranges
that are missing.

Scoring consistency
-------------------
Use the same interpretation of the six atomic rubric criteria for all answers.
Do not change a score because the answer is long, confident, or well-written
unless that writing actually demonstrates rubric-relevant knowledge.

Do not overwrite:
- training_dataset.csv
- training_dataset_v2c.csv
- any V1/V2/V3 model artifact

Only edit:
data/human_calibration_50.csv
"""

    OUT_TXT.write_text(instructions, encoding="utf-8")

    print("=" * 72)
    print("HUMAN CALIBRATION TEMPLATE CREATED")
    print("=" * 72)
    print(f"Rows:              {len(out)}")
    print(f"Questions:         {out['question_id'].nunique()}")
    print(f"Rows/question:     {ROWS_PER_QUESTION}")
    print(f"\nCSV:               {OUT_CSV}")
    print(f"Instructions:      {OUT_TXT}")
    print("\nNo existing dataset or model was modified.")


if __name__ == "__main__":
    main()
