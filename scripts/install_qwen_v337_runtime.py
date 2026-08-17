#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(
    os.getenv(
        "QWEN_SOURCE_ROOT",
        str(Path.home() / "smart-exam-eval-pretrained"),
    )
).expanduser().resolve()

EXPECTED_ENGINE_SHA = "3b8b189c93a11c49ad1a8ed3bc3c72296eb1e278d6de45bcdf950b60535b10fb"
EXPECTED_BASE_SHA = "f2ae96b94c0bb062be7ba85530ff59d642ac61c141a1054e8cc105b9293cca2e"

LOCAL_ENGINE = ROOT / "scripts" / "qwen_v337_engine.py"
SOURCE_BASE = SOURCE / "scripts" / "robustness_test_qwen_v33.py"
LOCAL_BASE = ROOT / "scripts" / "robustness_test_qwen_v33.py"
SOURCE_MODEL = SOURCE / "models" / "Qwen3.5-4B-MLX-4bit"
LOCAL_MODEL = ROOT / "models" / "Qwen3.5-4B-MLX-4bit"
SOURCE_FREEZE = SOURCE / "evaluation" / "qwen_v337_frozen"
LOCAL_FREEZE = ROOT / "evaluation" / "qwen_v337_frozen"
SERVER_EXEC = SOURCE / "venv" / "bin" / "mlx_vlm.server"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 88)
    print("QWEN V3.3.7 FINAL APP INTEGRATION INSTALLER")
    print("=" * 88)
    print(f"Final app:    {ROOT}")
    print(f"Qwen source:  {SOURCE}")

    if not LOCAL_ENGINE.exists():
        raise FileNotFoundError(LOCAL_ENGINE)

    engine_sha = sha256(LOCAL_ENGINE)
    if engine_sha != EXPECTED_ENGINE_SHA:
        raise RuntimeError(
            f"Bundled V3.3.7 engine SHA mismatch. Expected {EXPECTED_ENGINE_SHA}; "
            f"got {engine_sha}."
        )

    if not SOURCE_BASE.exists():
        raise FileNotFoundError(SOURCE_BASE)

    source_base_sha = sha256(SOURCE_BASE)
    if source_base_sha != EXPECTED_BASE_SHA:
        raise RuntimeError(
            f"Source base dependency SHA mismatch. Expected {EXPECTED_BASE_SHA}; "
            f"got {source_base_sha}."
        )

    if not SOURCE_MODEL.exists():
        raise FileNotFoundError(SOURCE_MODEL)
    if not SERVER_EXEC.exists():
        raise FileNotFoundError(SERVER_EXEC)

    LOCAL_BASE.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_BASE, LOCAL_BASE)
    local_base_sha = sha256(LOCAL_BASE)
    if local_base_sha != EXPECTED_BASE_SHA:
        raise RuntimeError("Copied base dependency failed SHA verification.")

    LOCAL_MODEL.parent.mkdir(parents=True, exist_ok=True)
    if LOCAL_MODEL.is_symlink():
        current = LOCAL_MODEL.resolve()
        if current != SOURCE_MODEL.resolve():
            LOCAL_MODEL.unlink()
            LOCAL_MODEL.symlink_to(SOURCE_MODEL, target_is_directory=True)
    elif LOCAL_MODEL.exists():
        print(f"Model path already exists; leaving it untouched: {LOCAL_MODEL}")
    else:
        LOCAL_MODEL.symlink_to(SOURCE_MODEL, target_is_directory=True)

    if SOURCE_FREEZE.exists():
        LOCAL_FREEZE.parent.mkdir(parents=True, exist_ok=True)
        if LOCAL_FREEZE.exists():
            shutil.rmtree(LOCAL_FREEZE)
        shutil.copytree(SOURCE_FREEZE, LOCAL_FREEZE)

    print()
    print("Engine SHA:        ", engine_sha)
    print("Base SHA:          ", local_base_sha)
    print("Model path:        ", LOCAL_MODEL)
    print("Model resolves to: ", LOCAL_MODEL.resolve())
    print("MLX executable:    ", SERVER_EXEC)
    print()
    print("INSTALLATION: PASS")
    print("Frozen V3.3.7 engine was not modified.")
    print("=" * 88)


if __name__ == "__main__":
    main()
