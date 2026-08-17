import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "models"))

from trained_grader import trained_score


def test_trained_grader_returns_bounded_score():
    result = trained_score(
        "What is Machine Learning?",
        "Machine learning is a branch of AI where models learn patterns from data to make predictions.",
        "Machine learning lets systems learn from examples and make predictions from data.",
        "AI;patterns;data;predictions",
    )
    assert "score" in result
    assert 0 <= result["score"] <= 10
