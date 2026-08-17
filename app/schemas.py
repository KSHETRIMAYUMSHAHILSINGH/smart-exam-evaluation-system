from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class QuestionCreate(BaseModel):
    question_id: str
    question_text: str
    model_answer: str
    rubric_keywords: Optional[str] = ""


class QuestionOut(BaseModel):
    question_id: str
    question_text: str
    model_answer: str
    rubric_keywords: Optional[str]
    model_config = ConfigDict(from_attributes=True)


class EvaluateRequest(BaseModel):
    question_id: str
    student_answer: str
    human_score: Optional[float] = None


class RubricCriterionResult(BaseModel):
    criterion_id: int
    criterion: str
    supported: bool
    evidence: Optional[str] = None
    qwen_claimed_supported: bool = False
    guard_rejected: bool = False


class EvaluateResponse(BaseModel):
    question_id: str
    final_score: float
    raw_qwen_score: float
    grader_name: str
    grader_version: str
    core: int
    supported_count: int
    rubric_coverage: float
    supported_ids: List[int] = Field(default_factory=list)
    contradiction_penalty: float
    contradiction_quote: str = ""
    wrong_topic: bool
    non_answer: bool
    deterministic_non_answer: bool
    non_answer_reason: str = ""
    preprocessed: bool
    duplicate_count: int
    quarantined_count: int
    evidence_guard_applied: bool
    rejected_criterion_ids: List[int] = Field(default_factory=list)
    rubric_results: List[RubricCriterionResult] = Field(default_factory=list)
    elapsed_sec: float

    baseline_score: Optional[float] = None
    semantic_score: Optional[float] = None
    trained_score: Optional[float] = None
    trained_model_name: Optional[str] = None
    matched_keywords: List[str] = Field(default_factory=list)
    missing_keywords: List[str] = Field(default_factory=list)

    human_score: Optional[float] = None


class MCQQuestionCreate(BaseModel):
    question_id: str
    question_text: str
    options: dict
    correct_answer: str
    max_score: int = 1


class MCQQuestionPublicOut(BaseModel):
    question_id: str
    question_text: str
    options: dict
    max_score: int
    model_config = ConfigDict(from_attributes=True)


class MCQQuestionAdminOut(MCQQuestionPublicOut):
    correct_answer: str


class MCQEvaluateRequest(BaseModel):
    question_id: str
    student_answer: str
