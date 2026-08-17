import json
import sys
from pathlib import Path


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

sys.path.insert(
    0,
    str(MODELS_DIR),
)


# ---------------------------------------------------------
# Grading components
# ---------------------------------------------------------

from baseline_scoring import keyword_match_score
from similarity_scoring import semantic_score
from trained_grader import trained_score


# ---------------------------------------------------------
# Hybrid weight configuration
# ---------------------------------------------------------

WEIGHTS_PATH = (
    ROOT
    / "models"
    / "artifacts"
    / "optimized_hybrid_weights.json"
)


# These are the optimized values we found.
DEFAULT_TRAINED_WEIGHT = 0.95
DEFAULT_KEYWORD_WEIGHT = 0.00
DEFAULT_SEMANTIC_WEIGHT = 0.05


def load_hybrid_weights():
    """
    Load optimized grading weights.

    If the optimizer output is unavailable,
    fall back to the currently validated:

        95% trained model
         0% keyword score
         5% semantic score
    """

    trained_weight = DEFAULT_TRAINED_WEIGHT
    keyword_weight = DEFAULT_KEYWORD_WEIGHT
    semantic_weight = DEFAULT_SEMANTIC_WEIGHT

    try:

        if WEIGHTS_PATH.exists():

            data = json.loads(
                WEIGHTS_PATH.read_text(
                    encoding="utf-8"
                )
            )

            optimized = data.get(
                "optimized_weights",
                {},
            )

            trained_weight = float(
                optimized.get(
                    "trained_weight",
                    trained_weight,
                )
            )

            keyword_weight = float(
                optimized.get(
                    "keyword_weight",
                    keyword_weight,
                )
            )

            semantic_weight = float(
                optimized.get(
                    "semantic_weight",
                    semantic_weight,
                )
            )


        total = (
            trained_weight
            + keyword_weight
            + semantic_weight
        )


        # Protect against a malformed weight file.
        if total <= 0:

            return (
                DEFAULT_TRAINED_WEIGHT,
                DEFAULT_KEYWORD_WEIGHT,
                DEFAULT_SEMANTIC_WEIGHT,
            )


        # Normalize in case rounding means
        # the values don't add up exactly to 1.
        trained_weight /= total
        keyword_weight /= total
        semantic_weight /= total


        return (
            trained_weight,
            keyword_weight,
            semantic_weight,
        )


    except Exception as exc:

        print(
            "[GRADING] Could not load optimized "
            f"weights: {exc}"
        )

        return (
            DEFAULT_TRAINED_WEIGHT,
            DEFAULT_KEYWORD_WEIGHT,
            DEFAULT_SEMANTIC_WEIGHT,
        )


# ---------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------

def score_all(
    question_text,
    model_answer,
    student_answer,
    rubric_keywords,
):

    # -----------------------------------------------------
    # Rubric keywords
    # -----------------------------------------------------

    keywords = [
        keyword.strip()

        for keyword
        in str(
            rubric_keywords or ""
        ).split(";")

        if keyword.strip()
    ]


    # -----------------------------------------------------
    # Keyword baseline
    #
    # We still calculate this because it is useful for:
    #
    # - matched concepts
    # - missing concepts
    # - explainability
    #
    # But the optimizer found that it should have
    # 0% direct influence on the final numeric grade.
    # -----------------------------------------------------

    baseline = keyword_match_score(
        student_answer,
        keywords,
    )


    # -----------------------------------------------------
    # Semantic baseline
    # -----------------------------------------------------

    try:

        semantic = semantic_score(
            student_answer,
            model_answer,
        )

        semantic_value = float(
            semantic["score"]
        )

    except Exception as exc:

        print(
            "[GRADING] Semantic scorer failed: "
            f"{exc}"
        )

        semantic_value = 0.0


    # -----------------------------------------------------
    # Trained model
    # -----------------------------------------------------

    trained = trained_score(
        question_text,
        model_answer,
        student_answer,
        rubric_keywords,
    )


    trained_value = float(
        trained["score"]
    )


    # -----------------------------------------------------
    # Optimized hybrid weights
    # -----------------------------------------------------

    (
        trained_weight,
        keyword_weight,
        semantic_weight,
    ) = load_hybrid_weights()


    # -----------------------------------------------------
    # Final grade
    #
    # Current optimized result:
    #
    #   95% ExtraTrees trained model
    #    0% keyword baseline
    #    5% semantic baseline
    #
    # These weights came from our leakage-safer
    # leave-one-question-out evaluation rather
    # than being manually chosen.
    # -----------------------------------------------------

    final_score = (
        trained_weight
        * trained_value

        + keyword_weight
        * float(
            baseline["score"]
        )

        + semantic_weight
        * semantic_value
    )


    # Keep result safely within exam range.
    final_score = max(
        0.0,
        min(
            10.0,
            final_score,
        ),
    )


    final_score = round(
        final_score,
        2,
    )


    # -----------------------------------------------------
    # Response
    # -----------------------------------------------------

    return {
        "baseline_score":
            baseline["score"],

        "matched_keywords":
            baseline["matched_keywords"],

        "missing_keywords":
            baseline["missing_keywords"],

        "semantic_score":
            semantic_value,

        "trained_score":
            trained["score"],

        "final_score":
            final_score,

        "trained_model_name":
            trained["model_name"],
    }
