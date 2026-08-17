import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "models"))
from baseline_scoring import keyword_match_score

def test_full_keyword_match():
    result = keyword_match_score("The cat sat on the mat", ["cat", "mat"])
    assert result["score"] == 10
    assert result["coverage"] == 100.0

def test_partial_keyword_match():
    result = keyword_match_score("The cat sat down", ["cat", "mat"])
    assert result["score"] == 5.0
    assert "cat" in result["matched_keywords"]
    assert "mat" in result["missing_keywords"]

def test_no_keywords_matched():
    assert keyword_match_score("hello world", ["cat", "mat"])["score"] == 0

def test_empty_rubric():
    assert keyword_match_score("hello world", [])["score"] == 0