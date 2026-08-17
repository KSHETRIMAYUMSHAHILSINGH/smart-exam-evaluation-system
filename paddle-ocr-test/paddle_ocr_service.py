from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
)
from paddleocr import PaddleOCR
from PIL import Image, ImageOps, UnidentifiedImageError

import io
import tempfile
import threading
import time
from pathlib import Path


app = FastAPI(
    title="PP-OCRv5 Dual-Mode Handwriting Service"
)


# =========================================================
# SETTINGS
# =========================================================

FAST_MAX_IMAGE_DIMENSION = 2200

# Accurate mode preserves more original handwriting detail.
ACCURATE_MAX_IMAGE_DIMENSION = 3000

FAST_JPEG_QUALITY = 92
ACCURATE_JPEG_QUALITY = 96


# =========================================================
# FAST OCR
# =========================================================

print(
    "[OCR] Loading FAST model...",
    flush=True,
)

load_start = time.perf_counter()

fast_ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="en_PP-OCRv5_mobile_rec",

    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)

print(
    f"[OCR] FAST model loaded in "
    f"{time.perf_counter() - load_start:.2f}s",
    flush=True,
)


# =========================================================
# ACCURATE OCR
# =========================================================
#
# Accurate mode uses:
#
# 1. The English handwriting recognizer.
# 2. The larger PP-OCRv5 server recognizer.
#
# Both use the FAST mobile detector.
#
# This avoids the extremely slow server detector.
#
# The accurate model is lazy-loaded only when needed.
# =========================================================

accurate_ocr = None
accurate_model_lock = threading.Lock()


def get_accurate_ocr():

    global accurate_ocr

    if accurate_ocr is not None:
        return accurate_ocr

    with accurate_model_lock:

        if accurate_ocr is not None:
            return accurate_ocr

        print(
            "",
            flush=True,
        )

        print(
            "[OCR] Loading ACCURATE recognizer...",
            flush=True,
        )

        start = time.perf_counter()

        accurate_ocr = PaddleOCR(
            # Keep the fast detector.
            text_detection_model_name=(
                "PP-OCRv5_mobile_det"
            ),

            # Stronger general PP-OCRv5 recognizer.
            text_recognition_model_name=(
                "PP-OCRv5_server_rec"
            ),

            # Do NOT load the expensive document
            # preprocessing models.
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

        print(
            f"[OCR] ACCURATE recognizer loaded in "
            f"{time.perf_counter() - start:.2f}s",
            flush=True,
        )

    return accurate_ocr


# =========================================================
# IMAGE PREPARATION
# =========================================================

def prepare_image(
    data: bytes,
    mode: str,
):
    """
    Prepare uploaded handwriting for OCR.

    FAST:
        Maximum dimension: 2200 px

    ACCURATE:
        Maximum dimension: 3000 px
        Higher JPEG quality

    Both:
        - Correct EXIF rotation
        - Convert to RGB
        - Preserve aspect ratio
        - Never upscale
    """

    try:

        image = Image.open(
            io.BytesIO(data)
        )

    except UnidentifiedImageError:

        raise HTTPException(
            status_code=400,
            detail=(
                "Uploaded file is not "
                "a valid image."
            ),
        )

    image = ImageOps.exif_transpose(
        image
    )

    if image.mode != "RGB":
        image = image.convert("RGB")

    original_width, original_height = (
        image.size
    )

    if mode == "accurate":

        max_dimension = (
            ACCURATE_MAX_IMAGE_DIMENSION
        )

        jpeg_quality = (
            ACCURATE_JPEG_QUALITY
        )

    else:

        max_dimension = (
            FAST_MAX_IMAGE_DIMENSION
        )

        jpeg_quality = (
            FAST_JPEG_QUALITY
        )

    largest_dimension = max(
        original_width,
        original_height,
    )

    if largest_dimension > max_dimension:

        scale = (
            max_dimension
            / largest_dimension
        )

        new_width = max(
            1,
            int(
                original_width * scale
            ),
        )

        new_height = max(
            1,
            int(
                original_height * scale
            ),
        )

        if hasattr(
            Image,
            "Resampling",
        ):

            resample_filter = (
                Image.Resampling.LANCZOS
            )

        else:

            resample_filter = (
                Image.LANCZOS
            )

        image = image.resize(
            (
                new_width,
                new_height,
            ),
            resample=resample_filter,
        )

    optimized_width, optimized_height = (
        image.size
    )

    output = io.BytesIO()

    image.save(
        output,
        format="JPEG",
        quality=jpeg_quality,
        optimize=True,
    )

    optimized_data = output.getvalue()

    return (
        optimized_data,
        (
            original_width,
            original_height,
        ),
        (
            optimized_width,
            optimized_height,
        ),
    )


# =========================================================
# RESULT READING
# =========================================================

def extract_result_data(
    results,
):
    """
    Read PaddleOCR text and confidence scores.
    """

    lines = []
    scores = []

    for result in results:

        payload = result.json

        if callable(payload):
            payload = payload()

        if not isinstance(
            payload,
            dict,
        ):
            continue

        payload = payload.get(
            "res",
            payload,
        )

        texts = (
            payload.get("rec_texts")
            or payload.get("texts")
            or []
        )

        rec_scores = (
            payload.get("rec_scores")
            or []
        )

        try:
            rec_scores = list(
                rec_scores
            )

        except TypeError:
            rec_scores = []

        for index, text in enumerate(
            texts
        ):

            text = str(
                text
            ).strip()

            if not text:
                continue

            lines.append(
                text
            )

            if index < len(
                rec_scores
            ):

                try:

                    scores.append(
                        float(
                            rec_scores[index]
                        )
                    )

                except (
                    TypeError,
                    ValueError,
                ):
                    pass

    text = "\n".join(
        lines
    ).strip()

    if scores:

        average_confidence = (
            sum(scores)
            / len(scores)
        )

        minimum_confidence = min(
            scores
        )

    else:

        average_confidence = None
        minimum_confidence = None

    return {
        "text": text,
        "lines": lines,
        "average_confidence":
            average_confidence,
        "minimum_confidence":
            minimum_confidence,
    }


# =========================================================
# RUN A MODEL
# =========================================================

def run_model(
    model,
    image_path: str,
):

    start = time.perf_counter()

    results = model.predict(
        image_path
    )

    seconds = (
        time.perf_counter()
        - start
    )

    result = extract_result_data(
        results
    )

    result[
        "seconds"
    ] = seconds

    return result


# =========================================================
# ACCURATE RESULT SELECTION
# =========================================================

def choose_best_result(
    english_result,
    server_result,
):
    """
    Choose the stronger OCR result.

    We consider:

    - Average recognition confidence
    - Lowest confidence
    - Amount of readable text recovered

    A small preference is given to the
    English handwriting model because it
    handles English spacing particularly well.
    """

    candidates = [
        (
            "english_handwriting",
            english_result,
        ),
        (
            "server_recognizer",
            server_result,
        ),
    ]

    max_characters = max(
        len(
            result["text"]
            .replace(" ", "")
            .replace("\n", "")
        )
        for _, result
        in candidates
    )

    max_characters = max(
        max_characters,
        1,
    )

    winner_name = None
    winner_result = None
    winner_score = -1.0

    for name, result in candidates:

        avg_conf = (
            result[
                "average_confidence"
            ]
            if result[
                "average_confidence"
            ] is not None
            else 0.0
        )

        min_conf = (
            result[
                "minimum_confidence"
            ]
            if result[
                "minimum_confidence"
            ] is not None
            else 0.0
        )

        character_count = len(
            result["text"]
            .replace(" ", "")
            .replace("\n", "")
        )

        coverage = (
            character_count
            / max_characters
        )

        quality_score = (
            (avg_conf * 0.75)
            + (min_conf * 0.10)
            + (coverage * 0.15)
        )

        # Tiny preference for the specialized
        # English handwriting recognizer when
        # results are extremely close.
        if name == "english_handwriting":
            quality_score += 0.005

        if quality_score > winner_score:

            winner_score = (
                quality_score
            )

            winner_name = name

            winner_result = result

    return (
        winner_name,
        winner_result,
        winner_score,
    )


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "ok",

        "engine": (
            "PP-OCRv5-dual"
        ),

        "fast": {
            "detector":
                "PP-OCRv5_mobile_det",

            "recognizer":
                "en_PP-OCRv5_mobile_rec",

            "loaded":
                True,
        },

        "accurate": {
            "detector":
                "PP-OCRv5_mobile_det",

            "recognizers": [
                "en_PP-OCRv5_mobile_rec",
                "PP-OCRv5_server_rec",
            ],

            "server_recognizer_loaded":
                accurate_ocr
                is not None,
        },
    }


# =========================================================
# OCR ENDPOINT
# =========================================================

@app.post("/ocr")
async def run_ocr(
    file: UploadFile = File(...),
    mode: str = Form("fast"),
):

    total_start = (
        time.perf_counter()
    )

    mode = (
        mode.strip().lower()
    )

    if mode not in {
        "fast",
        "accurate",
    }:

        raise HTTPException(
            status_code=400,
            detail=(
                "OCR mode must be "
                "'fast' or 'accurate'."
            ),
        )

    data = await file.read()

    if not data:

        raise HTTPException(
            status_code=400,
            detail="empty image",
        )

    original_mb = (
        len(data)
        / (1024 * 1024)
    )


    # -----------------------------------------------------
    # PREPROCESS
    # -----------------------------------------------------

    preprocess_start = (
        time.perf_counter()
    )

    (
        optimized_data,
        original_size,
        optimized_size,
    ) = prepare_image(
        data,
        mode,
    )

    preprocess_seconds = (
        time.perf_counter()
        - preprocess_start
    )

    optimized_mb = (
        len(optimized_data)
        / (1024 * 1024)
    )


    print(
        "",
        flush=True,
    )

    print(
        "[OCR] ==================================",
        flush=True,
    )

    print(
        f"[OCR] MODE: {mode.upper()}",
        flush=True,
    )

    print(
        f"[OCR] Upload size: "
        f"{original_mb:.2f} MB",
        flush=True,
    )

    print(
        f"[OCR] Original resolution: "
        f"{original_size[0]}x"
        f"{original_size[1]}",
        flush=True,
    )

    print(
        f"[OCR] OCR resolution: "
        f"{optimized_size[0]}x"
        f"{optimized_size[1]}",
        flush=True,
    )

    print(
        f"[OCR] OCR image size: "
        f"{optimized_mb:.2f} MB",
        flush=True,
    )

    print(
        f"[OCR] Preprocessing: "
        f"{preprocess_seconds:.2f}s",
        flush=True,
    )


    tmp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".jpg",
        ) as tmp:

            tmp.write(
                optimized_data
            )

            tmp_path = (
                tmp.name
            )


        # =================================================
        # FAST MODE
        # =================================================

        if mode == "fast":

            result = run_model(
                fast_ocr,
                tmp_path,
            )

            text = result[
                "text"
            ]

            selected_model = (
                "en_PP-OCRv5_mobile_rec"
            )

            selected_pass = (
                "fast"
            )

            total_ocr_seconds = (
                result["seconds"]
            )

            print(
                f"[OCR] OCR predict(): "
                f"{result['seconds']:.2f}s",
                flush=True,
            )


        # =================================================
        # ACCURATE MODE
        # =================================================

        else:

            # First candidate:
            # specialized English handwriting OCR.
            english_result = run_model(
                fast_ocr,
                tmp_path,
            )

            print(
                f"[OCR] English handwriting pass: "
                f"{english_result['seconds']:.2f}s",
                flush=True,
            )


            # Second candidate:
            # stronger general PP-OCRv5 recognizer.
            strong_ocr = (
                get_accurate_ocr()
            )

            server_result = run_model(
                strong_ocr,
                tmp_path,
            )

            print(
                f"[OCR] Server recognition pass: "
                f"{server_result['seconds']:.2f}s",
                flush=True,
            )


            (
                winner_name,
                result,
                quality_score,
            ) = choose_best_result(
                english_result,
                server_result,
            )


            text = result[
                "text"
            ]


            if (
                winner_name
                == "english_handwriting"
            ):

                selected_model = (
                    "en_PP-OCRv5_mobile_rec"
                )

            else:

                selected_model = (
                    "PP-OCRv5_server_rec"
                )


            selected_pass = (
                winner_name
            )


            total_ocr_seconds = (
                english_result[
                    "seconds"
                ]
                + server_result[
                    "seconds"
                ]
            )


            print(
                f"[OCR] Selected: "
                f"{selected_pass}",
                flush=True,
            )

            print(
                f"[OCR] Selection score: "
                f"{quality_score:.4f}",
                flush=True,
            )


    finally:

        if tmp_path:

            Path(
                tmp_path
            ).unlink(
                missing_ok=True
            )


    if not text:

        raise HTTPException(
            status_code=422,
            detail=(
                "PP-OCRv5 did not "
                "detect readable text."
            ),
        )


    total_seconds = (
        time.perf_counter()
        - total_start
    )


    print(
        f"[OCR] Total OCR model time: "
        f"{total_ocr_seconds:.2f}s",
        flush=True,
    )

    print(
        f"[OCR] Total request: "
        f"{total_seconds:.2f}s",
        flush=True,
    )

    print(
        f"[OCR] Lines detected: "
        f"{len(result['lines'])}",
        flush=True,
    )


    if (
        result[
            "average_confidence"
        ]
        is not None
    ):

        print(
            f"[OCR] Average confidence: "
            f"{result['average_confidence']:.3f}",
            flush=True,
        )


    if (
        result[
            "minimum_confidence"
        ]
        is not None
    ):

        print(
            f"[OCR] Lowest confidence: "
            f"{result['minimum_confidence']:.3f}",
            flush=True,
        )


    print(
        f"[OCR] Recognizer used: "
        f"{selected_model}",
        flush=True,
    )

    print(
        "[OCR] ==================================",
        flush=True,
    )


    return {
        "extracted_text":
            text,

        "engine":
            "PP-OCRv5",

        "mode":
            mode,

        "selected_pass":
            selected_pass,

        "models": {
            "detector":
                "PP-OCRv5_mobile_det",

            "recognizer":
                selected_model,
        },

        "confidence": {
            "average": (
                round(
                    result[
                        "average_confidence"
                    ],
                    4,
                )
                if result[
                    "average_confidence"
                ] is not None
                else None
            ),

            "minimum": (
                round(
                    result[
                        "minimum_confidence"
                    ],
                    4,
                )
                if result[
                    "minimum_confidence"
                ] is not None
                else None
            ),
        },

        "timing": {
            "preprocessing_seconds":
                round(
                    preprocess_seconds,
                    2,
                ),

            "ocr_seconds":
                round(
                    total_ocr_seconds,
                    2,
                ),

            "total_seconds":
                round(
                    total_seconds,
                    2,
                ),
        },
    }
