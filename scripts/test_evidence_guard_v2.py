#!/usr/bin/env python3
from __future__ import annotations

import hashlib
from pathlib import Path

from app.evidence_validator import apply_evidence_guard

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ENGINE_SHA = "3b8b189c93a11c49ad1a8ed3bc3c72296eb1e278d6de45bcdf950b60535b10fb"


def all_claimed_with_bad_evidence():
    return [
        {
            "criterion_id": i,
            "criterion": f"C{i}",
            "supported": True,
            "evidence": "Artificial Intelligence is a field of computing.",
        }
        for i in range(1, 7)
    ]


def main():
    # Frozen engine must still be exact.
    engine = ROOT / "scripts" / "qwen_v337_engine.py"
    actual_sha = hashlib.sha256(engine.read_bytes()).hexdigest()
    assert actual_sha == EXPECTED_ENGINE_SHA, actual_sha

    # Earlier verified typed answer must retain the same application result:
    # C1,C2,C3,C5 supported; C4,C6 rejected.
    typed = (
        "Artificial intelligence enables computers to perform tasks that "
        "normally require human intelligence, such as learning, reasoning, "
        "pattern recognition, language understanding, and decision making."
    )
    r1 = apply_evidence_guard(
        question_id="Q1",
        raw_qwen_score=10.0,
        core=5,
        contradiction_penalty=0.0,
        wrong_topic=False,
        non_answer=False,
        rubric_results=all_claimed_with_bad_evidence(),
        student_answer=typed,
    )
    assert r1["supported_ids"] == [1, 2, 3, 5], r1
    assert r1["rejected_criterion_ids"] == [4, 6], r1
    assert r1["final_score"] == 8.833, r1

    # Current difficult-handwriting OCR transcription.
    # Explicitly contains reasoning, solving problems, decisions and language.
    ocr_answer = (
        "Artificial Intelligence (AI) refers to the simulation of human "
        "intelligence by machines, which are programmed to think and learn "
        "like humans. It encompasses systems that can reason, solve problems, "
        "make decisions, understand language, and perceive their environment."
    )
    r2 = apply_evidence_guard(
        question_id="Q1",
        raw_qwen_score=9.0,
        core=4,
        contradiction_penalty=0.0,
        wrong_topic=False,
        non_answer=False,
        rubric_results=all_claimed_with_bad_evidence(),
        student_answer=ocr_answer,
    )
    assert r2["supported_ids"] == [1, 2, 3, 4, 5], r2
    assert r2["rejected_criterion_ids"] == [6], r2
    assert r2["final_score"] == 8.667, r2

    # Guard still cannot invent application examples.
    assert 6 not in r2["supported_ids"]

    print("EVIDENCE GUARD V2 REGRESSION CHECKS: PASS")
    print("- frozen V3.3.7 engine SHA unchanged: PASS")
    print("- earlier typed-answer behavior unchanged: 8.833/10")
    print("- OCR answer keeps explicit C1-C5 support: PASS")
    print("- unsupported C6 still rejected: PASS")
    print("- expected OCR-answer guarded score: 8.667/10")


if __name__ == "__main__":
    main()
