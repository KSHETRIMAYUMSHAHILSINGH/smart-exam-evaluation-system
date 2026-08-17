#!/usr/bin/env python3
from __future__ import annotations

from app.qwen_runtime import grade_qwen, start_qwen, stop_qwen

QUESTION = "What is Artificial Intelligence?"
REFERENCE = (
    "Artificial Intelligence (AI) is a technology that enables computers to "
    "perform tasks that normally require human intelligence. These tasks include "
    "learning, reasoning, problem-solving, understanding language, and recognizing "
    "patterns. AI is used in applications such as chatbots, recommendation systems, "
    "image recognition, and automated evaluation."
)


def main():
    print("Starting frozen Qwen V3.3.7 once...")
    state = start_qwen()
    print(f"Ready on port {state['port']}")

    try:
        strong = grade_qwen("Q1", QUESTION, REFERENCE, REFERENCE)
        zero = grade_qwen("Q1", QUESTION, REFERENCE, "8")

        print()
        print("Strong reference answer:")
        print(
            f"score={strong['score']} core={strong['core']} "
            f"support={strong['supported_count']}/6"
        )
        print("Numeric non-answer:")
        print(
            f"score={zero['score']} non_answer={zero['non_answer']} "
            f"reason={zero['non_answer_reason']}"
        )

        if strong["score"] < 8.0:
            raise RuntimeError("Strong-answer smoke score unexpectedly low.")
        if zero["score"] != 0.0:
            raise RuntimeError("Numeric non-answer guard failed.")

        print()
        print("QWEN V3.3.7 RUNTIME SMOKE TEST: PASS")
    finally:
        stop_qwen()


if __name__ == "__main__":
    main()
