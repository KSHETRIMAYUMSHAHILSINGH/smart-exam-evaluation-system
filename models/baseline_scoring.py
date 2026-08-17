import spacy

nlp = spacy.load("en_core_web_sm")

def preprocess(text):
    doc = nlp(text.lower())
    return [token.lemma_ for token in doc if not token.is_punct and not token.is_space]

def keyword_match_score(student_answer, rubric_keywords, max_score=10):
    student_tokens = set(preprocess(student_answer))
    matched = []
    for keyword in rubric_keywords:
        keyword_tokens = preprocess(keyword)
        if all(kt in student_tokens for kt in keyword_tokens):
            matched.append(keyword)

    coverage = len(matched) / len(rubric_keywords) if rubric_keywords else 0
    return {
        "score": round(coverage * max_score, 2),
        "max_score": max_score,
        "matched_keywords": matched,
        "missing_keywords": [k for k in rubric_keywords if k not in matched],
        "coverage": round(coverage * 100, 1)
    }

if __name__ == "__main__":
    rubric_keywords = ["resolver", "IP address", "hierarchical", "caching", "domain name"]
    student_answer = """
    DNS stands for Domain Name System. It converts domain names into IP addresses.
    It works in a hierarchical manner with root servers, TLD servers, and authoritative servers.
    """

    result = keyword_match_score(student_answer, rubric_keywords)
    print(f"Score: {result['score']}/{result['max_score']}")
    print(f"Coverage: {result['coverage']}%")
    print(f"Matched: {result['matched_keywords']}")
    print(f"Missing: {result['missing_keywords']}")