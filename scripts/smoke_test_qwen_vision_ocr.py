#!/usr/bin/env python3
from __future__ import annotations

from io import BytesIO
from PIL import Image, ImageDraw

from app.qwen_runtime import start_qwen, stop_qwen, transcribe_handwriting


def make_test_image() -> bytes:
    image = Image.new("RGB", (1500, 500), "white")
    draw = ImageDraw.Draw(image)
    draw.text(
        (80, 120),
        "Artificial intelligence enables computers to learn and reason.",
        fill="black",
    )
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def main():
    print("Starting existing Qwen3.5 vision-language model...")
    state = start_qwen()
    print(f"Ready on port {state['port']}")

    try:
        result = transcribe_handwriting(make_test_image())
        print("Engine:", result["engine"])
        print("Time:  ", result["seconds"])
        print("Text:  ", result["text"])

        lowered = result["text"].lower()
        if "artificial" not in lowered or "intelligence" not in lowered:
            raise RuntimeError(
                "Vision request worked but transcription did not preserve the test text."
            )

        print("QWEN VISION OCR RUNTIME SMOKE TEST: PASS")
    finally:
        stop_qwen()


if __name__ == "__main__":
    main()
