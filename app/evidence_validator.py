from __future__ import annotations

import re
from typing import Any


# Each rubric criterion has one or more concept groups.
# At least one token/phrase from EACH group must occur in the evidence span.
# The groups are intentionally small and interpretable.
#
# This is NOT a second grader. It only checks whether a span that Qwen already
# cited contains enough explicit lexical evidence to justify that criterion.
CRITERION_GATES: dict[str, dict[int, list[set[str]]]] = {
    "Q1": {
        1: [
            {"artificial intelligence", "ai"},
            {"computer", "computers", "machine", "machines", "system", "systems"},
            {"human intelligence", "intelligence", "human-like"},
        ],
        2: [
            {"learn", "learns", "learning", "learned", "training"},
        ],
        3: [
            {"reason", "reasoning", "decision", "decisions", "decision-making"},
        ],
        4: [
            {"solve", "solves", "problem", "problems", "problem-solving"},
        ],
        5: [
            {"language", "text", "speech", "pattern", "patterns", "image", "images"},
        ],
        6: [
            {
                "chatbot",
                "chatbots",
                "recommendation",
                "recommendations",
                "image recognition",
                "automated evaluation",
                "application",
                "applications",
            },
        ],
    },
    "Q2": {
        1: [
            {"machine learning", "ml"},
            {"data", "example", "examples", "training"},
        ],
        2: [
            {"pattern", "patterns", "relationship", "relationships"},
            {"data", "training", "examples"},
        ],
        3: [
            {"predict", "predicts", "prediction", "predictions", "decision", "decisions"},
            {"new", "unseen", "future"},
        ],
        4: [
            {"hard-coded", "hardcoded", "explicitly programmed", "rule", "rules"},
        ],
        5: [
            {"train", "training", "learn", "learning", "adjust", "update", "fit"},
            {"model", "models", "parameter", "parameters", "performance"},
        ],
        6: [
            {
                "classification",
                "regression",
                "prediction",
                "predictions",
                "classify",
                "predict",
            },
        ],
    },
    "Q3": {
        1: [
            {"natural language processing", "nlp"},
            {"language", "text", "speech"},
        ],
        2: [
            {"process", "processing", "analyze", "analysis", "interpret"},
            {"language", "text", "speech", "written", "spoken"},
        ],
        3: [
            {"classification", "sentiment", "text classification", "sentiment analysis"},
        ],
        4: [
            {"translation", "translate", "question answering", "qa"},
        ],
        5: [
            {"semantic", "meaning", "similarity"},
        ],
        6: [
            {"grading", "evaluation", "student", "learner"},
            {"reference", "answer", "response", "compare", "comparison"},
        ],
    },
    "Q4": {
        1: [
            {"cloud", "cloud computing"},
            {"internet", "online", "remote", "network"},
        ],
        2: [
            {
                "server",
                "servers",
                "storage",
                "database",
                "databases",
                "networking",
                "software",
                "resource",
                "resources",
            },
        ],
        3: [
            {"hardware", "infrastructure", "physical"},
            {"maintain", "maintenance", "own", "ownership", "local"},
        ],
        4: [
            {"scale", "scalable", "scalability", "elastic", "elasticity"},
            {"demand", "capacity", "need", "usage"},
        ],
        5: [
            {"flexibility", "accessible", "access", "remote", "location", "device"},
        ],
        6: [
            {"cost", "costs", "expense", "expenses", "saving", "savings", "cheaper"},
        ],
    },
    "Q5": {
        1: [
            {"database management system", "dbms"},
            {"store", "storage", "organize", "organizing", "manage", "retrieve"},
        ],
        2: [
            {"query", "querying", "retrieve", "retrieval", "record", "records", "storage"},
        ],
        3: [
            {"accuracy", "integrity", "consistency", "accurate", "correctness"},
        ],
        4: [
            {"security", "secure", "access", "permission", "permissions", "controlled"},
        ],
        5: [
            {
                "multiple users",
                "multi-user",
                "concurrent",
                "users",
                "applications",
                "duplication",
                "duplicate",
                "redundancy",
            },
        ],
        6: [
            {
                "banking",
                "hospital",
                "hospitals",
                "education",
                "e-commerce",
                "ecommerce",
                "shopping",
                "application",
                "applications",
            },
        ],
    },
}


def _normalize(text: Any) -> str:
    text = str(text or "").lower()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"[^a-z0-9+\- ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _contains_term(text: str, term: str) -> bool:
    term = _normalize(term)

    if " " in term or "-" in term:
        return term in text

    return re.search(
        rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])",
        text,
    ) is not None


def criterion_explicitly_supported(
    question_id: str,
    criterion_id: int,
    evidence: str,
) -> bool:
    qid = str(question_id).strip().upper()
    gates = CRITERION_GATES.get(qid, {}).get(int(criterion_id))

    # Unknown question/criterion: do not silently override the frozen engine.
    if not gates:
        return True

    normalized = _normalize(evidence)
    if not normalized:
        return False

    for group in gates:
        if not any(_contains_term(normalized, term) for term in group):
            return False

    return True


def validate_rubric_results(
    question_id: str,
    rubric_results: list[dict],
    student_answer: str = "",
) -> tuple[list[dict], list[int]]:
    """Validate only rubric claims already made by frozen Qwen.

    V2 first checks Qwen's cited evidence span. If that span is too short or
    poorly aligned, it falls back to the ORIGINAL student answer.

    The guard still cannot add support:
      - Qwen must already have claimed the criterion.
      - The explicit concept must occur in cited evidence OR the student's text.
    """
    validated: list[dict] = []
    rejected: list[int] = []

    for item in rubric_results:
        current = dict(item)
        criterion_id = int(current.get("criterion_id", 0) or 0)
        claimed = bool(current.get("supported", False))
        evidence = str(current.get("evidence", "") or "")

        evidence_explicit = (
            claimed
            and criterion_explicitly_supported(
                question_id,
                criterion_id,
                evidence,
            )
        )

        answer_explicit = (
            claimed
            and criterion_explicitly_supported(
                question_id,
                criterion_id,
                student_answer,
            )
        )

        explicit = bool(
            claimed
            and (
                evidence_explicit
                or answer_explicit
            )
        )

        current["qwen_claimed_supported"] = claimed
        current["supported"] = explicit
        current["guard_rejected"] = bool(claimed and not explicit)

        if current["guard_rejected"]:
            rejected.append(criterion_id)

        validated.append(current)

    return validated, rejected


def recompute_score_from_validated_support(
    core: int,
    supported_count: int,
    contradiction_penalty: float,
    wrong_topic: bool,
    non_answer: bool,
) -> float:
    """Recompute with the frozen V3.3.7 score equation.

    The engine's core and contradiction judgment are untouched. Only rubric
    support count is replaced by evidence-validated support.
    """
    core = max(0, min(5, int(core)))
    supported_count = max(0, min(6, int(supported_count)))
    penalty = max(0.0, min(5.0, float(contradiction_penalty)))

    if wrong_topic or non_answer:
        return 0.0

    # Same coherence floor used by V3.3.7 calculate_score().
    if supported_count >= 5 and core < 4:
        core = 4
    elif supported_count >= 3 and core < 3:
        core = 3

    base_map = {
        0: 0.0,
        1: 1.0,
        2: 3.0,
        3: 5.0,
        4: 6.0,
        5: 7.0,
    }
    base = base_map[core]

    coverage = supported_count / 6.0
    support_bonus = 2.0 * coverage

    if coverage >= 5 / 6:
        completeness_bonus = 1.0
    elif coverage >= 3 / 6:
        completeness_bonus = 0.5
    else:
        completeness_bonus = 0.0

    score = base + support_bonus + completeness_bonus - penalty

    if core == 0:
        score = 0.0
    elif core == 1:
        score = min(score, 2.0)
    elif core == 2:
        score = min(score, 4.0)
    elif core == 3:
        score = min(score, 6.5)

    if supported_count == 0 and core <= 2:
        score = min(score, 1.0)

    return round(max(0.0, min(10.0, float(score))), 3)


def apply_evidence_guard(
    question_id: str,
    raw_qwen_score: float,
    core: int,
    contradiction_penalty: float,
    wrong_topic: bool,
    non_answer: bool,
    rubric_results: list[dict],
    student_answer: str = "",
) -> dict:
    validated, rejected = validate_rubric_results(
        question_id,
        rubric_results,
        student_answer=student_answer,
    )

    supported_ids = [
        int(item["criterion_id"])
        for item in validated
        if item.get("supported")
    ]
    supported_count = len(supported_ids)

    guarded_score = recompute_score_from_validated_support(
        core=core,
        supported_count=supported_count,
        contradiction_penalty=contradiction_penalty,
        wrong_topic=wrong_topic,
        non_answer=non_answer,
    )

    return {
        "final_score": guarded_score,
        "raw_qwen_score": round(float(raw_qwen_score), 3),
        "rubric_results": validated,
        "supported_ids": supported_ids,
        "supported_count": supported_count,
        "rubric_coverage": round(supported_count / 6.0, 4),
        "evidence_guard_applied": bool(rejected),
        "rejected_criterion_ids": rejected,
    }
