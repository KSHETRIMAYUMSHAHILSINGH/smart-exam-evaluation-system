#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import py_compile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ENGINE_SHA = "3b8b189c93a11c49ad1a8ed3bc3c72296eb1e278d6de45bcdf950b60535b10fb"


def main():
    for path in [
        ROOT / "app" / "evidence_validator.py",
        ROOT / "app" / "qwen_runtime.py",
        ROOT / "app" / "schemas.py",
        ROOT / "app" / "main.py",
        ROOT / "streamlit_app.py",
    ]:
        py_compile.compile(str(path), doraise=True)

    engine = ROOT / "scripts" / "qwen_v337_engine.py"
    sha = hashlib.sha256(engine.read_bytes()).hexdigest()
    assert sha == EXPECTED_ENGINE_SHA

    from app.evidence_validator import apply_evidence_guard

    answer = (
        "Artificial intelligence enables computers to perform tasks that normally "
        "require human intelligence, such as learning, reasoning, pattern recognition, "
        "language understanding, and decision making."
    )

    raw_results = [
        {
            "criterion_id": i,
            "criterion": f"C{i}",
            "supported": True,
            "evidence": answer,
        }
        for i in range(1, 7)
    ]

    guarded = apply_evidence_guard(
        question_id="Q1",
        raw_qwen_score=10.0,
        core=5,
        contradiction_penalty=0.0,
        wrong_topic=False,
        non_answer=False,
        rubric_results=raw_results,
    )

    assert guarded["supported_ids"] == [1, 2, 3, 5], guarded
    assert guarded["rejected_criterion_ids"] == [4, 6], guarded
    assert guarded["supported_count"] == 4, guarded
    assert guarded["final_score"] == 8.833, guarded

    print("EVIDENCE-GUARD STATIC CHECKS: PASS")
    print("- frozen V3.3.7 engine SHA unchanged: PASS")
    print("- screenshot answer keeps C1,C2,C3,C5: PASS")
    print("- screenshot answer rejects C4,C6: PASS")
    print("- guarded final score: 8.833/10")
    print("- raw frozen-Qwen score remains available: 10.0/10")


if __name__ == "__main__":
    main()
