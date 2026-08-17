"""Build a 100-answer prototype dataset for the Smart Exam Evaluation project.

The existing 20 group-graded answers are preserved as `source=human_group`.
This script adds 80 clearly labelled *synthetic augmentation* examples so the
prototype grader sees the full 0-10 score range. Synthetic rows are not claimed
as real student data; they are for rapid prototype training only.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "manifest.csv"
OUTPUT = ROOT / "data" / "training_dataset.csv"

TOPICS = {
    "Q1": {
        "name": "Artificial Intelligence",
        "intro": [
            "Artificial Intelligence is the field of building computer systems that perform tasks associated with human intelligence.",
            "AI enables machines to carry out tasks that normally need human-like intelligence.",
        ],
        "concepts": [
            "It can learn from examples and improve its behaviour from data.",
            "AI systems can reason and support decision making.",
            "They can solve problems instead of following only one fixed sequence of instructions.",
            "AI can understand language and recognize patterns in text, images, or other data.",
            "Applications include chatbots, recommendation systems, image recognition, and automated evaluation.",
        ],
        "vague": "It is used to make computers act smart and automate difficult tasks.",
        "misconception": "AI simply means storing a very large amount of information in a computer database.",
        "irrelevant": "Cloud storage keeps files on remote servers and lets users access them through the internet.",
    },
    "Q2": {
        "name": "Machine Learning",
        "intro": [
            "Machine Learning is a branch of AI in which models learn from data.",
            "ML is an AI approach that lets software learn patterns from examples.",
        ],
        "concepts": [
            "The model identifies useful patterns or relationships in training data.",
            "Those learned patterns are used to make predictions or decisions on new inputs.",
            "The system is not explicitly programmed with a separate hard-coded rule for every case.",
            "Training examples are used to adjust the model so its performance can improve.",
            "Common ML tasks include classification, regression, and prediction.",
        ],
        "vague": "It helps a computer get better at a task after seeing examples.",
        "misconception": "Machine learning is a database technique that only stores and retrieves records faster.",
        "irrelevant": "A DBMS organizes tables, controls access, and supports queries on stored records.",
    },
    "Q3": {
        "name": "Natural Language Processing",
        "intro": [
            "Natural Language Processing is an area of AI that enables computers to work with human language.",
            "NLP combines AI techniques with language processing so machines can interpret text or speech.",
        ],
        "concepts": [
            "It processes and analyzes human-written or spoken language.",
            "NLP can perform text classification and sentiment analysis.",
            "It is used for machine translation and question answering.",
            "Semantic representations can be used to measure text similarity and meaning.",
            "In automated grading, NLP can compare a student's response with a reference answer.",
        ],
        "vague": "It is the technology behind many systems that read or respond to text.",
        "misconception": "NLP is a networking protocol used to transfer files between computers.",
        "irrelevant": "Cloud computing provides servers, storage, and databases over the internet.",
    },
    "Q4": {
        "name": "Cloud Computing",
        "intro": [
            "Cloud computing delivers computing resources and services over the internet.",
            "Cloud computing lets users access computing services remotely instead of owning all infrastructure locally.",
        ],
        "concepts": [
            "Services can include storage, databases, servers, networking, and software.",
            "Users access these resources through the internet without maintaining all physical hardware themselves.",
            "Resources can be scaled up or down according to demand.",
            "It improves flexibility and accessibility from different locations or devices.",
            "It can reduce infrastructure and maintenance costs for organizations.",
        ],
        "vague": "It means using online computing resources instead of keeping everything on one local computer.",
        "misconception": "Cloud computing is only a weather forecasting technique that predicts rainfall using computers.",
        "irrelevant": "Machine learning models learn statistical patterns from training examples to make predictions.",
    },
    "Q5": {
        "name": "Database Management Systems",
        "intro": [
            "A Database Management System is software for storing, organizing, managing, and retrieving data.",
            "A DBMS provides a structured way to manage and access large collections of data.",
        ],
        "concepts": [
            "It supports efficient storage, organization, querying, and retrieval of records.",
            "A DBMS helps maintain data accuracy, integrity, and consistency.",
            "It provides security and controlled access to sensitive information.",
            "It can support many users and applications while reducing unnecessary duplication of data.",
            "DBMSs are important in banking, hospitals, education systems, and e-commerce applications.",
        ],
        "vague": "It keeps information arranged so applications can find and update it more easily.",
        "misconception": "A DBMS is mainly a graphics program used to design images and animations.",
        "irrelevant": "Natural language processing helps computers interpret human language and perform translation.",
    },
}

# 16 synthetic rows + 4 existing human-group rows = 20 per question.
SYNTHETIC_SPECS = [
    (0, "irrelevant", []),
    (1, "misconception", []),
    (2, "partial", [0]),
    (3, "partial_vague", [1]),
    (3, "partial", [3]),
    (4, "partial", [0, 1]),
    (5, "partial_vague", [0, 2]),
    (5, "partial", [1, 3, 4]),
    (6, "partial", [0, 1, 2]),
    (6, "partial_vague", [0, 2, 3]),
    (7, "partial", [0, 1, 2, 3]),
    (7, "partial", [0, 2, 3, 4]),
    (8, "partial", [0, 1, 2, 3]),
    (8, "partial", [0, 1, 3, 4]),
    (9, "partial", [0, 1, 2, 3, 4]),
    (9, "partial", [0, 1, 2, 3, 4]),
]


def build_synthetic_answer(topic: dict, score: int, mode: str, idxs: list[int], variant: int) -> str:
    if mode == "irrelevant":
        return topic["irrelevant"]
    if mode == "misconception":
        return topic["misconception"]

    parts = []
    # Include a proper definition more often as scores increase, while still varying wording.
    if score >= 5 or (score >= 3 and variant % 2 == 0):
        parts.append(topic["intro"][variant % len(topic["intro"])])
    elif score <= 2:
        parts.append(f"{topic['name']} is a computer-related technology.")

    parts.extend(topic["concepts"][i] for i in idxs)
    if mode == "partial_vague":
        parts.append(topic["vague"])

    # High-quality variants get a concise synthesis sentence rather than copying the reference verbatim.
    if score >= 9:
        parts.append("Together, these capabilities make it useful in real-world information systems and intelligent applications.")
    return " ".join(parts)


def main() -> None:
    base = pd.read_csv(SOURCE)
    base["source"] = "human_group"

    rows = []
    for qid, qdf in base.groupby("question_id", sort=False):
        first = qdf.iloc[0]
        topic = TOPICS[qid]
        for variant, (score, mode, idxs) in enumerate(SYNTHETIC_SPECS):
            rows.append({
                "question_id": qid,
                "question_text": first["question_text"],
                "model_answer": first["model_answer"],
                "student_answer": build_synthetic_answer(topic, score, mode, idxs, variant),
                "human_score": float(score),
                "rubric_keywords": first["rubric_keywords"],
                "source": "synthetic_augmented",
            })

    synthetic = pd.DataFrame(rows)
    out = pd.concat([base, synthetic], ignore_index=True)
    # Stable ordering: 20 examples per question.
    out = out.sort_values(["question_id", "source", "human_score"], ascending=[True, True, True]).reset_index(drop=True)
    out.to_csv(OUTPUT, index=False)

    print(f"Wrote {len(out)} rows to {OUTPUT}")
    print(out.groupby(["question_id", "source"]).size())
    print("\nScore distribution:")
    print(out["human_score"].value_counts().sort_index())


if __name__ == "__main__":
    main()
