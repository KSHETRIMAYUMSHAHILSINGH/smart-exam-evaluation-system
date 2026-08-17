from __future__ import annotations

import base64
import hashlib
import io
import json
import importlib
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from app.qwen_rubrics import get_qwen_rubric
from app.evidence_validator import apply_evidence_guard

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
DATA_DIR = ROOT / "data"

EXPECTED_ENGINE_SHA256 = "3b8b189c93a11c49ad1a8ed3bc3c72296eb1e278d6de45bcdf950b60535b10fb"
EXPECTED_BASE_SHA256 = "f2ae96b94c0bb062be7ba85530ff59d642ac61c141a1054e8cc105b9293cca2e"

SOURCE_ROOT = Path(
    os.getenv(
        "QWEN_SOURCE_ROOT",
        str(Path.home() / "smart-exam-eval-pretrained"),
    )
).expanduser().resolve()

SERVER_EXECUTABLE = Path(
    os.getenv(
        "QWEN_SERVER_EXECUTABLE",
        str(SOURCE_ROOT / "venv" / "bin" / "mlx_vlm.server"),
    )
).expanduser()

MODEL_RELATIVE_PATH = Path("models/Qwen3.5-4B-MLX-4bit")
LOCAL_MODEL_PATH = ROOT / MODEL_RELATIVE_PATH
SERVER_LOG = DATA_DIR / "qwen_v337_server.log"

_process: subprocess.Popen | None = None
_log_file = None
_port: int | None = None
_engine = None
_startup_error: str | None = None
_grade_lock = threading.Lock()
_state_lock = threading.Lock()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_frozen_files() -> None:
    engine_path = SCRIPTS_DIR / "qwen_v337_engine.py"
    base_path = SCRIPTS_DIR / "robustness_test_qwen_v33.py"

    if not engine_path.exists():
        raise FileNotFoundError(
            "Missing scripts/qwen_v337_engine.py. Run the installer first."
        )
    if not base_path.exists():
        raise FileNotFoundError(
            "Missing scripts/robustness_test_qwen_v33.py. Run the installer first."
        )

    engine_sha = _sha256(engine_path)
    base_sha = _sha256(base_path)

    if engine_sha != EXPECTED_ENGINE_SHA256:
        raise RuntimeError(
            "Frozen Qwen V3.3.7 engine SHA mismatch. "
            f"Expected {EXPECTED_ENGINE_SHA256}, got {engine_sha}."
        )
    if base_sha != EXPECTED_BASE_SHA256:
        raise RuntimeError(
            "Frozen Qwen base dependency SHA mismatch. "
            f"Expected {EXPECTED_BASE_SHA256}, got {base_sha}."
        )


def _load_engine():
    global _engine
    if _engine is not None:
        return _engine

    _verify_frozen_files()
    scripts_text = str(SCRIPTS_DIR)
    if scripts_text not in sys.path:
        sys.path.insert(0, scripts_text)

    module = importlib.import_module("qwen_v337_engine")
    if getattr(module, "INFERENCE_SEED", None) != 42:
        raise RuntimeError("Qwen V3.3.7 inference seed is not 42.")

    _engine = module
    return module


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


def _wait_for_server(port: int, process: subprocess.Popen, timeout: int = 90) -> None:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout

    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                "MLX Qwen server exited during startup. "
                f"Check {SERVER_LOG}."
            )
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            pass
        time.sleep(0.5)

    raise TimeoutError(
        "MLX Qwen server did not become ready within 90 seconds. "
        f"Check {SERVER_LOG}."
    )


def _cleanup_locked() -> None:
    global _process, _log_file, _port

    if _process is not None:
        try:
            if _process.poll() is None:
                _process.terminate()
                try:
                    _process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    _process.kill()
                    _process.wait(timeout=5)
        except Exception:
            pass

    if _log_file is not None:
        try:
            _log_file.close()
        except Exception:
            pass

    _process = None
    _log_file = None
    _port = None


def start_qwen() -> dict[str, Any]:
    global _process, _log_file, _port, _startup_error

    with _state_lock:
        if _process is not None and _process.poll() is None and _port is not None:
            return status()

        try:
            _load_engine()

            if not SERVER_EXECUTABLE.exists():
                raise FileNotFoundError(
                    f"Validated MLX server executable not found: {SERVER_EXECUTABLE}"
                )
            if not LOCAL_MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"Qwen model link/path not found: {LOCAL_MODEL_PATH}. "
                    "Run scripts/install_qwen_v337_runtime.py first."
                )

            DATA_DIR.mkdir(parents=True, exist_ok=True)
            _port = _free_port()
            _log_file = SERVER_LOG.open("w", encoding="utf-8")

            command = [
                str(SERVER_EXECUTABLE),
                "--model",
                str(MODEL_RELATIVE_PATH),
                "--port",
                str(_port),
                "--log-level",
                "WARNING",
            ]

            _process = subprocess.Popen(
                command,
                cwd=str(ROOT),
                stdout=_log_file,
                stderr=subprocess.STDOUT,
            )
            _wait_for_server(_port, _process)
            _startup_error = None
            return status()

        except Exception as exc:
            _startup_error = str(exc)
            _cleanup_locked()
            raise


def stop_qwen() -> None:
    with _state_lock:
        _cleanup_locked()


def is_ready() -> bool:
    return bool(
        _process is not None
        and _process.poll() is None
        and _port is not None
    )


def status() -> dict[str, Any]:
    return {
        "ready": is_ready(),
        "version": "Qwen V3.3.7",
        "seed": 42,
        "engine_sha256": EXPECTED_ENGINE_SHA256,
        "base_sha256": EXPECTED_BASE_SHA256,
        "port": _port,
        "source_root": str(SOURCE_ROOT),
        "model_path": str(LOCAL_MODEL_PATH),
        "startup_error": _startup_error,
    }



def _prepare_ocr_image(file_bytes: bytes) -> str:
    """Normalize an uploaded page and return one JPEG data URL."""
    if not file_bytes:
        raise ValueError("OCR image is empty.")

    with Image.open(io.BytesIO(file_bytes)) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")

        max_side = max(image.size)
        if max_side > 2600:
            scale = 2600.0 / max_side
            image = image.resize(
                (
                    max(1, int(image.width * scale)),
                    max(1, int(image.height * scale)),
                ),
                Image.Resampling.LANCZOS,
            )

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=94, optimize=True)

    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def transcribe_handwriting(file_bytes: bytes) -> dict[str, Any]:
    """Literal handwriting transcription with the already-loaded Qwen VLM.

    This does not call or modify qwen_v337_engine.py. It only reuses the same
    local vision-language model process for an image transcription request.
    """
    if not is_ready() or _port is None:
        raise RuntimeError(
            "Qwen vision OCR is unavailable because the local Qwen server "
            f"is not ready. Status: {status()}"
        )

    data_url = _prepare_ocr_image(file_bytes)

    prompt = (
        "Transcribe the handwritten student exam answer in this image. "
        "Return ONLY text actually written by the student. Do not answer the "
        "exam question yourself. Do not improve, summarize, complete, or "
        "correct the student's content. Preserve wording, spelling, numbers, "
        "and line order as closely as possible. Ignore ruled-paper lines, "
        "page borders, shadows, and non-text marks. If a word truly cannot be "
        "read, write [?] for that word instead of inventing content. Output "
        "plain text only, with no markdown or commentary."
    )

    payload = {
        "model": str(MODEL_RELATIVE_PATH),
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
        "temperature": 0,
        "seed": 42,
        "max_tokens": 1200,
        "enable_thinking": False,
    }

    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{_port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.perf_counter()
    try:
        with _grade_lock:
            with urllib.request.urlopen(request, timeout=180) as response:
                response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Qwen vision OCR HTTP {exc.code}: {error_body}"
        ) from exc

    elapsed = time.perf_counter() - started
    response_json = json.loads(response_body)
    text = str(response_json["choices"][0]["message"]["content"]).strip()

    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()

    if not text:
        raise RuntimeError("Qwen vision OCR returned empty text.")

    return {
        "text": text,
        "engine": "Qwen3.5-4B-MLX-4bit vision OCR",
        "seconds": round(float(elapsed), 3),
    }

def grade_qwen(
    question_id: str,
    question_text: str,
    model_answer: str,
    student_answer: str,
    criteria: list[str] | None = None,
) -> dict[str, Any]:
    if not is_ready() or _port is None:
        raise RuntimeError(
            "Qwen V3.3.7 server is not ready. "
            f"Status: {status()}"
        )

    engine = _load_engine()
    # CUSTOM-RUBRIC SUPPORT
    # Existing Q1-Q5 calls still use the frozen registered rubric.
    # Custom questions may provide six generated atomic criteria.
    if criteria is None:
        criteria = get_qwen_rubric(question_id)
    else:
        criteria = [
            str(item).strip()
            for item in criteria
            if str(item).strip()
        ]

        if len(criteria) != 6:
            raise ValueError(
                "Frozen Qwen V3.3.7 requires exactly "
                "6 rubric criteria."
            )

    with _grade_lock:
        score, detail, _raw, elapsed = engine.request_grade(
            port=_port,
            question=question_text,
            reference=model_answer,
            criteria=criteria,
            student_answer=student_answer,
        )

    spans = {
        int(item.get("id")): str(item.get("text", ""))
        for item in detail.get("spans", [])
        if item.get("id") is not None
    }
    evidence_span_ids = detail.get("evidence_span_ids", {})
    rubric_results = []

    for index, criterion in enumerate(criteria, start=1):
        span_id = int(evidence_span_ids.get(f"c{index}", 0) or 0)
        rubric_results.append(
            {
                "criterion_id": index,
                "criterion": criterion,
                "supported": span_id > 0,
                "evidence": spans.get(span_id) if span_id > 0 else None,
            }
        )

    supported_ids = [int(x) for x in detail.get("supported_ids", [])]

    guard = apply_evidence_guard(
        question_id=question_id,
        raw_qwen_score=float(score),
        core=int(detail.get("core", 0)),
        contradiction_penalty=float(detail.get("penalty", 0.0)),
        wrong_topic=bool(detail.get("wrong_topic", False)),
        non_answer=bool(detail.get("non_answer", False)),
        rubric_results=rubric_results,
        student_answer=student_answer,
    )

    return {
        "score": guard["final_score"],
        "raw_qwen_score": guard["raw_qwen_score"],
        "grader_name": "Qwen3.5-4B-MLX-4bit",
        "grader_version": "V3.3.7 frozen + evidence guard v2",
        "core": int(detail.get("core", 0)),
        "supported_count": guard["supported_count"],
        "rubric_coverage": guard["rubric_coverage"],
        "supported_ids": guard["supported_ids"],
        "contradiction_penalty": float(detail.get("penalty", 0.0)),
        "contradiction_quote": str(detail.get("contradiction_quote", "") or ""),
        "wrong_topic": bool(detail.get("wrong_topic", False)),
        "non_answer": bool(detail.get("non_answer", False)),
        "deterministic_non_answer": bool(
            detail.get("deterministic_non_answer", False)
        ),
        "non_answer_reason": str(detail.get("non_answer_reason", "") or ""),
        "preprocessed": bool(detail.get("preprocessed", False)),
        "duplicate_count": int(detail.get("duplicate_count", 0) or 0),
        "quarantined_count": len(detail.get("quarantined", [])),
        "rubric_results": guard["rubric_results"],
        "evidence_guard_applied": guard["evidence_guard_applied"],
        "rejected_criterion_ids": guard["rejected_criterion_ids"],
        "elapsed_sec": round(float(elapsed), 3),
    }
