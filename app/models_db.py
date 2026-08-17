from sqlalchemy import Column, Integer, String, Float, Text, DateTime, JSON
from sqlalchemy.sql import func
from app.database import Base


class Question(Base):
    __tablename__ = "questions"

    id = Column(Integer, primary_key=True, index=True)
    question_id = Column(String, unique=True, index=True)
    question_text = Column(Text, nullable=False)
    model_answer = Column(Text, nullable=False)
    rubric_keywords = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Evaluation(Base):
    __tablename__ = "evaluations"

    id = Column(Integer, primary_key=True, index=True)
    question_id = Column(String, index=True)
    student_answer = Column(Text, nullable=False)
    human_score = Column(Float, nullable=True)
    baseline_score = Column(Float, nullable=True)
    semantic_score = Column(Float, nullable=True)
    trained_score = Column(Float, nullable=True)
    # Kept only for backward compatibility with the earlier experiment/database.
    finetuned_score = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class MCQQuestion(Base):
    __tablename__ = "mcq_questions"

    id = Column(Integer, primary_key=True, index=True)
    question_id = Column(String, unique=True, index=True)
    question_text = Column(Text, nullable=False)
    options = Column(JSON, nullable=False)
    correct_answer = Column(String, nullable=False)
    max_score = Column(Integer, default=1)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
