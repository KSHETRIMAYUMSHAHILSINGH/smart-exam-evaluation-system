#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import py_compile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ENGINE = "3b8b189c93a11c49ad1a8ed3bc3c72296eb1e278d6de45bcdf950b60535b10fb"
EXPECTED_BASE = "f2ae96b94c0bb062be7ba85530ff59d642ac61c141a1054e8cc105b9293cca2e"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    files = [
        ROOT / "app" / "main.py",
        ROOT / "app" / "schemas.py",
        ROOT / "app" / "qwen_runtime.py",
        ROOT / "app" / "qwen_rubrics.py",
        ROOT / "streamlit_app.py",
        ROOT / "scripts" / "qwen_v337_engine.py",
        ROOT / "scripts" / "robustness_test_qwen_v33.py",
    ]

    for path in files:
        if not path.exists():
            raise FileNotFoundError(path)
        py_compile.compile(str(path), doraise=True)

    assert sha(ROOT / "scripts" / "qwen_v337_engine.py") == EXPECTED_ENGINE
    assert sha(ROOT / "scripts" / "robustness_test_qwen_v33.py") == EXPECTED_BASE

    from app.qwen_rubrics import RUBRICS
    assert set(RUBRICS) == {"Q1", "Q2", "Q3", "Q4", "Q5"}
    assert all(len(v) == 6 for v in RUBRICS.values())

    print("FINAL INTEGRATION STATIC CHECKS: PASS")
    print("- app files compile: PASS")
    print("- V3.3.7 engine SHA: PASS")
    print("- frozen base SHA: PASS")
    print("- Q1-Q5 six-criterion rubrics: PASS")


if __name__ == "__main__":
    main()
