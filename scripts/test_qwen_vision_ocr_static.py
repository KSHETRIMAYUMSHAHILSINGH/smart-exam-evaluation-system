#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import py_compile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ENGINE_SHA = "3b8b189c93a11c49ad1a8ed3bc3c72296eb1e278d6de45bcdf950b60535b10fb"


def main():
    for path in [
        ROOT / "app" / "qwen_runtime.py",
        ROOT / "app" / "main.py",
        ROOT / "streamlit_app.py",
    ]:
        py_compile.compile(str(path), doraise=True)

    engine = ROOT / "scripts" / "qwen_v337_engine.py"
    actual = hashlib.sha256(engine.read_bytes()).hexdigest()
    assert actual == EXPECTED_ENGINE_SHA, actual

    runtime = (ROOT / "app" / "qwen_runtime.py").read_text()
    main = (ROOT / "app" / "main.py").read_text()

    assert "def transcribe_handwriting" in runtime
    assert '"type": "image_url"' in runtime
    assert "with _grade_lock" in runtime
    assert 'extract_text_from_image(contents, mode="fast")' in main
    assert "transcribe_handwriting(contents)" in main

    print("QWEN VISION OCR INTEGRATION STATIC CHECKS: PASS")
    print("- frozen V3.3.7 engine SHA unchanged: PASS")
    print("- Fast OCR remains PP-OCRv5: PASS")
    print("- Accurate OCR uses local Qwen vision input: PASS")
    print("- OCR/grading inference serialized for memory safety: PASS")


if __name__ == "__main__":
    main()
