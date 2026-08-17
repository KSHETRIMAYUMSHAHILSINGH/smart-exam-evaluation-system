"""Prepare V3 Difference-Aware Contrastive Grading data.

Research basis
--------------
Adapted from DiffCL (Chen et al., ACL Findings 2026):
- five heuristic difference markers [DIFF0]...[DIFF4]
- same-score positive pairs for InfoNCE
- hierarchical pairs where score gap <= 1 is "close" and > 1 is "far"

Important adaptation for this project
-------------------------------------
Our labels are 0-10 marks rather than the paper's five categorical labels.
We therefore:
1. preserve the original 0-10 human score,
2. add a 5-level auxiliary score bucket,
3. build within-question score-gap pairs,
4. keep pair row IDs so held-out examples can be removed BEFORE training.

NO model is trained by this script.

Inputs
------
data/training_dataset_v2c.csv
data/rubric_concepts_v2c.json

Outputs
-------
data/v3_contrastive_examples.csv
data/v3_infonce_pairs.csv
data/v3_hierarchical_pairs.csv
data/v3_pair_manifest.json
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "training_dataset_v2c.csv"
RUBRIC_PATH = ROOT / "data" / "rubric_concepts_v2c.json"

EXAMPLES_PATH = ROOT / "data" / "v3_contrastive_examples.csv"
INFONCE_PATH = ROOT / "data" / "v3_infonce_pairs.csv"
HIER_PATH = ROOT / "data" / "v3_hierarchical_pairs.csv"
MANIFEST_PATH = ROOT / "data" / "v3_pair_manifest.json"

NEGATIONS = {
    "not", "no", "never", "none", "nothing", "neither",
    "cannot", "can't", "isn't", "aren't", "doesn't",
    "don't", "didn't", "won't", "without",
}

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.I)


def clean_text(value) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(clean_text(text).lower())


def jaccard_similarity(a: str, b: str) -> float:
    a_set = set(tokens(a))
    b_set = set(tokens(b))

    if not a_set and not b_set:
        return 1.0

    if not a_set or not b_set:
        return 0.0

    return len(a_set & b_set) / len(a_set | b_set)


def levenshtein_distance(a_tokens: list[str], b_tokens: list[str]) -> int:
    """Memory-efficient word-level Levenshtein distance."""
    if len(a_tokens) < len(b_tokens):
        a_tokens, b_tokens = b_tokens, a_tokens

    previous = list(range(len(b_tokens) + 1))

    for i, a_tok in enumerate(a_tokens, start=1):
        current = [i]

        for j, b_tok in enumerate(b_tokens, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            substitute_cost = previous[j - 1] + (
                0 if a_tok == b_tok else 1
            )

            current.append(
                min(insert_cost, delete_cost, substitute_cost)
            )

        previous = current

    return previous[-1]


def edit_similarity(reference: str, student: str) -> float:
    r = tokens(reference)
    s = tokens(student)

    denom = max(len(r), len(s), 1)
    distance = levenshtein_distance(r, s)

    return float(np.clip(1.0 - distance / denom, 0.0, 1.0))


def has_negation(text: str) -> bool:
    return any(tok in NEGATIONS for tok in tokens(text))


def diffcl_similarity(reference: str, student: str) -> tuple[float, float, float]:
    jac = jaccard_similarity(reference, student)
    edit = edit_similarity(reference, student)

    # Equation used by DiffCL: mean of Jaccard + edit similarity.
    combined = (jac + edit) / 2.0

    return jac, edit, combined


def difference_marker(reference: str, student: str) -> tuple[str, int, bool]:
    _, _, similarity = diffcl_similarity(reference, student)

    negation_mismatch = has_negation(reference) != has_negation(student)

    # DiffCL explicitly downgrades negation mismatch to DIFF4.
    if negation_mismatch:
        return "DIFF4", 4, True

    if similarity >= 0.80:
        level = 0
    elif similarity >= 0.60:
        level = 1
    elif similarity >= 0.35:
        level = 2
    elif similarity >= 0.20:
        level = 3
    else:
        level = 4

    return f"DIFF{level}", level, False


def score_level_5(score: float) -> int:
    """Auxiliary 5-level score target while preserving the raw 0-10 score."""
    score = float(np.clip(score, 0.0, 10.0))

    if score <= 2:
        return 0
    if score <= 4:
        return 1
    if score <= 6:
        return 2
    if score <= 8:
        return 3
    return 4


def structured_input(row: pd.Series, rubric_concepts: dict) -> str:
    qid = str(row["question_id"])

    criteria = rubric_concepts[qid]["criteria"]
    rubric = " | ".join(
        f"C{i + 1}: {clean_text(c)}"
        for i, c in enumerate(criteria)
    )

    return (
        f"Question: {clean_text(row['question_text'])}\n"
        f"Reference: {clean_text(row['model_answer'])}\n"
        f"Rubric: {rubric}\n"
        f"Difference: [{row['difference_marker']}]\n"
        f"Student: {clean_text(row['student_answer'])}"
    )


def build_examples(df: pd.DataFrame, rubric_concepts: dict) -> pd.DataFrame:
    records = []

    for idx, row in df.iterrows():
        reference = clean_text(row["model_answer"])
        student = clean_text(row["student_answer"])

        jac, edit, sim = diffcl_similarity(reference, student)
        marker, diff_level, neg_mismatch = difference_marker(
            reference,
            student,
        )

        record = row.to_dict()

        record["_row_id"] = int(idx)
        record["jaccard_similarity"] = jac
        record["edit_similarity"] = edit
        record["diffcl_similarity"] = sim
        record["difference_marker"] = marker
        record["difference_level"] = diff_level
        record["negation_mismatch"] = int(neg_mismatch)
        record["score_level_5"] = score_level_5(
            float(row["human_score"])
        )

        records.append(record)

    out = pd.DataFrame(records)

    out["structured_input"] = [
        structured_input(row, rubric_concepts)
        for _, row in out.iterrows()
    ]

    return out


def build_infonce_pairs(examples: pd.DataFrame) -> pd.DataFrame:
    """Same-score positives.

    Primary positives are within the same question.
    Cross-question same-score positives are included at lower weight to help
    learn a common scoring geometry without dominating topic semantics.
    """
    records = []

    # High-confidence positives: same question + exact same raw score.
    for (qid, score), group in examples.groupby(
        ["question_id", "human_score"],
        sort=True,
    ):
        ids = group["_row_id"].astype(int).tolist()

        for a, b in combinations(ids, 2):
            records.append({
                "anchor_row_id": a,
                "positive_row_id": b,
                "question_id_anchor": qid,
                "question_id_positive": qid,
                "human_score": float(score),
                "pair_scope": "same_question",
                "pair_weight": 1.0,
            })

            # Symmetric ordered pair for anchor-based InfoNCE.
            records.append({
                "anchor_row_id": b,
                "positive_row_id": a,
                "question_id_anchor": qid,
                "question_id_positive": qid,
                "human_score": float(score),
                "pair_scope": "same_question",
                "pair_weight": 1.0,
            })

    # Lower-weight cross-question positives.
    # Limit each anchor to at most 3 deterministic positives.
    for score, group in examples.groupby("human_score", sort=True):
        rows = group[
            ["_row_id", "question_id"]
        ].sort_values("_row_id")

        rows = list(rows.itertuples(index=False, name=None))

        for anchor_id, anchor_qid in rows:
            candidates = [
                (rid, qid)
                for rid, qid in rows
                if qid != anchor_qid
            ]

            for positive_id, positive_qid in candidates[:3]:
                records.append({
                    "anchor_row_id": int(anchor_id),
                    "positive_row_id": int(positive_id),
                    "question_id_anchor": anchor_qid,
                    "question_id_positive": positive_qid,
                    "human_score": float(score),
                    "pair_scope": "cross_question",
                    "pair_weight": 0.35,
                })

    pairs = pd.DataFrame(records)

    if pairs.empty:
        raise RuntimeError("No InfoNCE positive pairs were generated.")

    pairs = pairs.drop_duplicates(
        subset=[
            "anchor_row_id",
            "positive_row_id",
            "pair_scope",
        ]
    ).reset_index(drop=True)

    return pairs


def build_hierarchical_pairs(examples: pd.DataFrame) -> pd.DataFrame:
    """Within-question score-gap pairs.

    Mirrors DiffCL:
      abs(score gap) <= 1  -> close pair
      abs(score gap) > 1   -> margin-separated pair

    We also store direction for a later ordinal/ranking auxiliary loss.
    """
    records = []

    for qid, group in examples.groupby("question_id", sort=True):
        rows = list(
            group[
                ["_row_id", "human_score", "source"]
            ].itertuples(index=False, name=None)
        )

        for left, right in combinations(rows, 2):
            a_id, a_score, a_source = left
            b_id, b_score, b_source = right

            a_score = float(a_score)
            b_score = float(b_score)

            gap = abs(a_score - b_score)

            if gap <= 1.0:
                relation = "close"
                margin_target = 1.0
            else:
                relation = "far"
                # DiffCL's far-pair margin operates at cosine 0.5.
                margin_target = 0.5

            if a_score > b_score:
                higher_id, lower_id = int(a_id), int(b_id)
            elif b_score > a_score:
                higher_id, lower_id = int(b_id), int(a_id)
            else:
                higher_id, lower_id = -1, -1

            # Human-involving pairs are useful but should not silently dominate.
            contains_human = int(
                str(a_source) == "human_group"
                or str(b_source) == "human_group"
            )

            records.append({
                "row_id_a": int(a_id),
                "row_id_b": int(b_id),
                "question_id": qid,
                "score_a": a_score,
                "score_b": b_score,
                "score_gap": gap,
                "relation": relation,
                "margin_target": margin_target,
                "higher_row_id": higher_id,
                "lower_row_id": lower_id,
                "contains_human": contains_human,
            })

    pairs = pd.DataFrame(records)

    if pairs.empty:
        raise RuntimeError("No hierarchical pairs were generated.")

    return pairs


def integrity_checks(
    examples: pd.DataFrame,
    infonce: pd.DataFrame,
    hierarchical: pd.DataFrame,
):
    ids = set(examples["_row_id"].astype(int))

    assert examples["_row_id"].is_unique
    assert examples["human_score"].between(0, 10).all()
    assert examples["difference_level"].between(0, 4).all()
    assert examples["score_level_5"].between(0, 4).all()

    assert set(infonce["anchor_row_id"]).issubset(ids)
    assert set(infonce["positive_row_id"]).issubset(ids)

    assert set(hierarchical["row_id_a"]).issubset(ids)
    assert set(hierarchical["row_id_b"]).issubset(ids)

    assert (
        infonce["anchor_row_id"]
        != infonce["positive_row_id"]
    ).all()

    assert (
        hierarchical["row_id_a"]
        != hierarchical["row_id_b"]
    ).all()

    # Same-score InfoNCE pairs must truly have equal scores.
    score_lookup = (
        examples.set_index("_row_id")["human_score"].to_dict()
    )

    for row in infonce.itertuples(index=False):
        a = float(score_lookup[int(row.anchor_row_id)])
        b = float(score_lookup[int(row.positive_row_id)])

        if not math.isclose(a, b):
            raise AssertionError(
                f"Invalid InfoNCE pair: {row.anchor_row_id}, "
                f"{row.positive_row_id}: {a} != {b}"
            )


def main():
    print("=" * 76)
    print("V3 DIFFERENCE-AWARE CONTRASTIVE DATA PREPARATION")
    print("=" * 76)

    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    if not RUBRIC_PATH.exists():
        raise FileNotFoundError(RUBRIC_PATH)

    df = pd.read_csv(DATA_PATH).reset_index(drop=True)

    with RUBRIC_PATH.open("r", encoding="utf-8") as f:
        rubric_concepts = json.load(f)

    df["human_score"] = pd.to_numeric(
        df["human_score"],
        errors="raise",
    ).astype(float)

    examples = build_examples(
        df,
        rubric_concepts,
    )

    infonce = build_infonce_pairs(
        examples
    )

    hierarchical = build_hierarchical_pairs(
        examples
    )

    integrity_checks(
        examples,
        infonce,
        hierarchical,
    )

    print(f"\nExamples:               {len(examples)}")
    print(
        f"Human examples:         "
        f"{(examples['source'] == 'human_group').sum()}"
    )
    print(
        f"Synthetic examples:     "
        f"{(examples['source'] != 'human_group').sum()}"
    )

    print("\nDifference markers:")
    print(
        examples["difference_marker"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nAuxiliary score levels:")
    print(
        examples["score_level_5"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(f"\nInfoNCE positive pairs: {len(infonce)}")
    print(
        infonce["pair_scope"]
        .value_counts()
        .to_string()
    )

    print(
        f"\nHierarchical pairs:     "
        f"{len(hierarchical)}"
    )
    print(
        hierarchical["relation"]
        .value_counts()
        .to_string()
    )

    print("\nScore-gap distribution:")
    gap_counts = (
        hierarchical["score_gap"]
        .value_counts()
        .sort_index()
    )
    print(gap_counts.to_string())

    # Save.
    EXAMPLES_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    examples.to_csv(
        EXAMPLES_PATH,
        index=False,
    )

    infonce.to_csv(
        INFONCE_PATH,
        index=False,
    )

    hierarchical.to_csv(
        HIER_PATH,
        index=False,
    )

    manifest = {
        "version": "v3_contrastive_data_v1",
        "source_dataset": str(DATA_PATH),
        "rubric_file": str(RUBRIC_PATH),
        "example_count": int(len(examples)),
        "infonce_pair_count": int(len(infonce)),
        "hierarchical_pair_count": int(len(hierarchical)),
        "difference_marker": {
            "method": (
                "DiffCL lexical heuristic: mean(Jaccard, "
                "word-level edit similarity)"
            ),
            "thresholds": {
                "DIFF0": ">=0.80",
                "DIFF1": "0.60-0.80",
                "DIFF2": "0.35-0.60",
                "DIFF3": "0.20-0.35",
                "DIFF4": "<=0.20 or negation mismatch",
            },
        },
        "score_adaptation": {
            "raw_target": "human_score 0-10",
            "auxiliary_5_level": {
                "0": "0-2",
                "1": "3-4",
                "2": "5-6",
                "3": "7-8",
                "4": "9-10",
            },
        },
        "contrastive_design": {
            "infonce": (
                "exact same-score positives; same-question weight=1.0; "
                "cross-question weight=0.35"
            ),
            "hierarchical": (
                "within-question; score gap <=1 close, >1 far"
            ),
        },
        "leakage_rule": (
            "V3 trainer MUST split rows first and remove every contrastive "
            "pair containing a held-out row before fitting each fold."
        ),
    }

    with MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    print("\n## Saved")
    print(f"Examples:     {EXAMPLES_PATH}")
    print(f"InfoNCE:      {INFONCE_PATH}")
    print(f"Hierarchical: {HIER_PATH}")
    print(f"Manifest:     {MANIFEST_PATH}")

    print(
        "\nImportant: no model was trained and no previous "
        "dataset/artifact was modified."
    )


if __name__ == "__main__":
    main()
