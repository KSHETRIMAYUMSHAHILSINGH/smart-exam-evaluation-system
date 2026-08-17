from __future__ import annotations

RUBRICS: dict[str, list[str]] = {
    "Q1": [
        "Artificial Intelligence builds computer systems that perform tasks associated with human intelligence.",
        "AI systems can learn from examples and improve their behaviour from data.",
        "AI systems can reason and support decision making.",
        "AI systems can solve problems instead of following only one fixed sequence of instructions.",
        "AI systems can understand language and recognize patterns in text, images, or other data.",
        "AI applications include systems such as chatbots, recommendation systems, image recognition, or automated evaluation.",
    ],
    "Q2": [
        "Machine Learning is a branch of AI in which models learn from data or examples.",
        "A machine learning model identifies useful patterns or relationships in training data.",
        "Learned patterns are used to make predictions or decisions on new or unseen inputs.",
        "Machine learning does not require a separate hard-coded rule for every individual case.",
        "Training examples are used to adjust the model so its performance can improve.",
        "Common machine learning tasks include classification, regression, or prediction.",
    ],
    "Q3": [
        "Natural Language Processing is an area of AI that enables computers to work with human language.",
        "NLP processes or analyzes human-written or spoken language.",
        "NLP can perform tasks such as text classification or sentiment analysis.",
        "NLP can be used for machine translation or question answering.",
        "Semantic representations can be used to measure text similarity and meaning.",
        "NLP can support automated grading by comparing a student's response with a reference answer.",
    ],
    "Q4": [
        "Cloud computing delivers computing resources or services over the internet.",
        "Cloud services can include storage, databases, servers, networking, or software.",
        "Users can access cloud resources without maintaining all physical hardware themselves.",
        "Cloud resources can be scaled up or down according to demand.",
        "Cloud computing improves flexibility or accessibility from different locations or devices.",
        "Cloud computing can reduce infrastructure or maintenance costs for organizations.",
    ],
    "Q5": [
        "A Database Management System is software for storing, organizing, managing, and retrieving data.",
        "A DBMS supports efficient storage, organization, querying, or retrieval of records.",
        "A DBMS helps maintain data accuracy, integrity, or consistency.",
        "A DBMS provides security or controlled access to information.",
        "A DBMS can support multiple users or applications while reducing unnecessary duplication of data.",
        "DBMSs are used in applications such as banking, hospitals, education systems, or e-commerce.",
    ],
}


def get_qwen_rubric(question_id: str) -> list[str]:
    qid = str(question_id).strip().upper()
    criteria = RUBRICS.get(qid)
    if criteria is None:
        raise KeyError(
            f"No frozen Qwen V3.3.7 rubric is registered for {question_id!r}. "
            "The validated final grader currently supports Q1-Q5."
        )
    return list(criteria)
