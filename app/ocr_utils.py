import io
import os
import re

import cv2
import numpy as np
import pytesseract
import requests
from PIL import Image


PADDLE_OCR_URL = os.getenv(
    "PADDLE_OCR_URL",
    "http://127.0.0.1:8001/ocr",
)


def _clean_ocr_text(
    text: str,
) -> str:
    """
    Convert OCR line-by-line output into
    normal readable paragraphs.
    """

    if not text:
        return ""

    # Normalize line endings.
    text = text.replace(
        "\r\n",
        "\n",
    ).replace(
        "\r",
        "\n",
    )

    # Remove spaces around newlines.
    text = re.sub(
        r"[ \t]+\n",
        "\n",
        text,
    )

    text = re.sub(
        r"\n[ \t]+",
        "\n",
        text,
    )

    # Join words split by OCR line-break hyphens.
    #
    # Example:
    #
    # intelli-
    # gence
    #
    # becomes:
    #
    # intelligence
    text = re.sub(
        r"([A-Za-z])-\n([A-Za-z])",
        r"\1\2",
        text,
    )

    # Two or more newlines count as
    # a genuine paragraph break.
    paragraphs = re.split(
        r"\n\s*\n+",
        text,
    )

    cleaned_paragraphs = []

    for paragraph in paragraphs:

        lines = [
            line.strip()
            for line
            in paragraph.split("\n")
            if line.strip()
        ]

        if not lines:
            continue

        # Join physical handwriting lines
        # into normal sentence flow.
        paragraph_text = " ".join(
            lines
        )

        # Collapse repeated spaces.
        paragraph_text = re.sub(
            r"\s+",
            " ",
            paragraph_text,
        )

        # Remove spaces before punctuation.
        paragraph_text = re.sub(
            r"\s+([,.!?;:])",
            r"\1",
            paragraph_text,
        )

        # Clean opening brackets.
        paragraph_text = re.sub(
            r"([\(\[\{])\s+",
            r"\1",
            paragraph_text,
        )

        # Clean closing brackets.
        paragraph_text = re.sub(
            r"\s+([\)\]\}])",
            r"\1",
            paragraph_text,
        )

        cleaned_paragraphs.append(
            paragraph_text.strip()
        )

    return "\n\n".join(
        cleaned_paragraphs
    ).strip()


def _local_tesseract_fallback(
    file_bytes: bytes,
) -> str:
    """
    Emergency OCR fallback if the
    PP-OCRv5 service is unavailable.
    """

    arr = np.frombuffer(
        file_bytes,
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        arr,
        cv2.IMREAD_COLOR,
    )

    if image is None:

        pil = Image.open(
            io.BytesIO(
                file_bytes
            )
        ).convert(
            "RGB"
        )

        image = cv2.cvtColor(
            np.array(pil),
            cv2.COLOR_RGB2BGR,
        )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    gray = cv2.resize(
        gray,
        None,
        fx=1.5,
        fy=1.5,
        interpolation=cv2.INTER_CUBIC,
    )

    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15,
    )

    text = pytesseract.image_to_string(
        binary,
        config="--oem 3 --psm 6",
    )

    return _clean_ocr_text(
        text
    )


def extract_text_from_image(
    file_bytes: bytes,
    mode: str = "fast",
) -> str:
    """
    Extract handwriting using PP-OCRv5.

    mode="fast"
        Fast mobile detector.

    mode="accurate"
        Strong server detector with
        orientation and document correction.

    Existing code calling this function
    without a mode will automatically
    continue using Fast OCR.
    """

    mode = (
        mode
        .strip()
        .lower()
    )

    if mode not in {
        "fast",
        "accurate",
    }:
        mode = "fast"

    try:

        response = requests.post(
            PADDLE_OCR_URL,

            files={
                "file": (
                    "answer.png",
                    file_bytes,
                    "image/png",
                )
            },

            data={
                "mode": mode,
            },

            timeout=180,
        )

        response.raise_for_status()

        data = response.json()

        text = data.get(
            "extracted_text",
            "",
        )

        if text and text.strip():

            return _clean_ocr_text(
                text
            )

    except Exception as exc:

        print(
            "[OCR] PP-OCRv5 service unavailable; "
            f"using Tesseract fallback: {exc}"
        )

    return _local_tesseract_fallback(
        file_bytes
    )
