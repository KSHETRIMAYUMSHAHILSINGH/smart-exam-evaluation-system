import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "models"))
from similarity_scoring import semantic_score

def test_identical_answers_score_high():
    text = "DNS resolves domain names into IP addresses."
    assert semantic_score(text, text)["score"] >= 9.5

def test_unrelated_answers_score_low():
    result = semantic_score(
        "The recipe calls for two cups of flour.",
        "DNS resolves domain names into IP addresses."
    )
    assert result["score"] < 3

def test_score_within_bounds():
    result = semantic_score("some answer", "some reference")
    assert 0 <= result["score"] <= 10