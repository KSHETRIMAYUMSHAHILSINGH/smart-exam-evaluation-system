from contextlib import asynccontextmanager
from typing import List

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session

from app.auth import verify_api_key
from app.custom_grading import router as custom_grading_router
from app.database import Base, engine, get_db
from app.models_db import Evaluation, MCQQuestion, Question
from app.ocr_utils import extract_text_from_image
from app.qwen_runtime import (
    grade_qwen,
    start_qwen,
    status as qwen_status,
    stop_qwen,
    transcribe_handwriting,
)
from app.schemas import (
    EvaluateRequest,
    EvaluateResponse,
    MCQEvaluateRequest,
    MCQQuestionAdminOut,
    MCQQuestionCreate,
    MCQQuestionPublicOut,
    QuestionCreate,
    QuestionOut,
)
from app.scoring import score_all

Base.metadata.create_all(bind=engine)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state = start_qwen()
        print(f"[QWEN] V3.3.7 ready on local port {state.get('port')}.")
        app.state.qwen_startup_error = None
    except Exception as exc:
        app.state.qwen_startup_error = str(exc)
        print(f"[QWEN] Startup failed: {exc}")

    yield
    stop_qwen()


app = FastAPI(
    title="Smart Exam Evaluation System API",
    version="2.0-final",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def read_ui():
    return """
    <html>
      <head><title>Smart Exam Evaluation System</title></head>
      <body>
        <h2>Smart Exam Evaluation System API</h2>
        <p>FastAPI backend is running.</p>
        <p>Use <code>/health</code> for Qwen runtime status.</p>
      </body>
    </html>
    """


@app.get("/health")
def health():
    qwen = qwen_status()
    return {
        "status": "ok" if qwen["ready"] else "degraded",
        "qwen": qwen,
        "ocr": "bridge-enabled",
        "mcq": "enabled",
    }


@app.get("/questions", response_model=List[QuestionOut])
def list_questions(db: Session = Depends(get_db)):
    return db.query(Question).all()


@app.post(
    "/questions",
    response_model=QuestionOut,
    dependencies=[Depends(verify_api_key)],
)
def create_question(payload: QuestionCreate, db: Session = Depends(get_db)):
    existing = db.query(Question).filter_by(question_id=payload.question_id).first()
    if existing:
        raise HTTPException(status_code=409, detail="question_id already exists")

    question = Question(**payload.model_dump())
    db.add(question)
    db.commit()
    db.refresh(question)
    return question


@app.get("/evaluations/{question_id}")
def get_evaluations(question_id: str, db: Session = Depends(get_db)):
    evaluations = (
        db.query(Evaluation)
        .filter(Evaluation.question_id == question_id)
        .all()
    )
    return [
        {
            "student_answer": evaluation.student_answer,
            "human_score": evaluation.human_score,
            "baseline_score": evaluation.baseline_score,
            "semantic_score": evaluation.semantic_score,
            "trained_score": evaluation.trained_score,
        }
        for evaluation in evaluations
    ]


def _legacy_comparison(question: Question, student_answer: str) -> dict:
    try:
        return score_all(
            question.question_text,
            question.model_answer,
            student_answer,
            question.rubric_keywords,
        )
    except Exception as exc:
        print(f"[LEGACY-ML] Comparison scorer failed: {exc}")
        return {
            "baseline_score": None,
            "semantic_score": None,
            "trained_score": None,
            "trained_model_name": None,
            "matched_keywords": [],
            "missing_keywords": [],
        }


@app.post(
    "/evaluate",
    response_model=EvaluateResponse,
    dependencies=[Depends(verify_api_key)],
)
def evaluate(payload: EvaluateRequest, db: Session = Depends(get_db)):
    question = db.query(Question).filter_by(question_id=payload.question_id).first()
    if not question:
        raise HTTPException(status_code=404, detail="question not found")

    qwen = qwen_status()
    if not qwen["ready"]:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Frozen Qwen V3.3.7 grader is not ready.",
                "qwen": qwen,
            },
        )

    try:
        qwen_result = grade_qwen(
            question_id=question.question_id,
            question_text=question.question_text,
            model_answer=question.model_answer,
            student_answer=payload.student_answer,
        )
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Qwen V3.3.7 grading failed: {exc}",
        ) from exc

    legacy = _legacy_comparison(question, payload.student_answer)

    evaluation = Evaluation(
        question_id=payload.question_id,
        student_answer=payload.student_answer,
        human_score=payload.human_score,
        baseline_score=legacy.get("baseline_score"),
        semantic_score=legacy.get("semantic_score"),
        trained_score=legacy.get("trained_score"),
    )
    db.add(evaluation)
    db.commit()

    return EvaluateResponse(
        question_id=payload.question_id,
        human_score=payload.human_score,
        final_score=qwen_result["score"],
        raw_qwen_score=qwen_result["raw_qwen_score"],
        grader_name=qwen_result["grader_name"],
        grader_version=qwen_result["grader_version"],
        core=qwen_result["core"],
        supported_count=qwen_result["supported_count"],
        rubric_coverage=qwen_result["rubric_coverage"],
        supported_ids=qwen_result["supported_ids"],
        contradiction_penalty=qwen_result["contradiction_penalty"],
        contradiction_quote=qwen_result["contradiction_quote"],
        wrong_topic=qwen_result["wrong_topic"],
        non_answer=qwen_result["non_answer"],
        deterministic_non_answer=qwen_result["deterministic_non_answer"],
        non_answer_reason=qwen_result["non_answer_reason"],
        preprocessed=qwen_result["preprocessed"],
        duplicate_count=qwen_result["duplicate_count"],
        quarantined_count=qwen_result["quarantined_count"],
        evidence_guard_applied=qwen_result["evidence_guard_applied"],
        rejected_criterion_ids=qwen_result["rejected_criterion_ids"],
        rubric_results=qwen_result["rubric_results"],
        elapsed_sec=qwen_result["elapsed_sec"],
        baseline_score=legacy.get("baseline_score"),
        semantic_score=legacy.get("semantic_score"),
        trained_score=legacy.get("trained_score"),
        trained_model_name=legacy.get("trained_model_name"),
        matched_keywords=legacy.get("matched_keywords", []),
        missing_keywords=legacy.get("missing_keywords", []),
    )


@app.post(
    "/mcq/questions",
    response_model=MCQQuestionAdminOut,
    dependencies=[Depends(verify_api_key)],
)
def create_mcq_question(payload: MCQQuestionCreate, db: Session = Depends(get_db)):
    existing = db.query(MCQQuestion).filter_by(question_id=payload.question_id).first()
    if existing:
        raise HTTPException(status_code=409, detail="question_id already exists")

    question = MCQQuestion(**payload.model_dump())
    db.add(question)
    db.commit()
    db.refresh(question)
    return question


@app.get("/mcq/questions", response_model=List[MCQQuestionPublicOut])
def list_mcq_questions(db: Session = Depends(get_db)):
    return db.query(MCQQuestion).all()


@app.post(
    "/mcq/evaluate",
    dependencies=[Depends(verify_api_key)],
)
def evaluate_mcq(payload: MCQEvaluateRequest, db: Session = Depends(get_db)):
    question = db.query(MCQQuestion).filter_by(question_id=payload.question_id).first()
    if not question:
        raise HTTPException(status_code=404, detail="MCQ question not found")

    student_answer = payload.student_answer.strip().upper()
    correct_answer = question.correct_answer.strip().upper()
    is_correct = student_answer == correct_answer
    score = question.max_score if is_correct else 0

    return {
        "question_id": payload.question_id,
        "student_answer": payload.student_answer,
        "correct_answer": question.correct_answer,
        "is_correct": is_correct,
        "score": score,
        "max_score": question.max_score,
    }


@app.post("/ocr", dependencies=[Depends(verify_api_key)])
async def ocr_extract(
    file: UploadFile = File(...),
    mode: str = Form("fast"),
):
    mode = mode.strip().lower()
    if mode not in {"fast", "accurate"}:
        raise HTTPException(
            status_code=400,
            detail="OCR mode must be 'fast' or 'accurate'.",
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    try:
        if mode == "fast":
            text = extract_text_from_image(contents, mode="fast")
            engine_name = "PP-OCRv5 fast"
            elapsed = None
        else:
            result = transcribe_handwriting(contents)
            text = result["text"]
            engine_name = result["engine"]
            elapsed = result["seconds"]
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"{mode.capitalize()} OCR failed: {exc}",
        ) from exc

    if not text.strip():
        raise HTTPException(
            status_code=422,
            detail="OCR could not detect readable text.",
        )

    return {
        "extracted_text": text,
        "mode": mode,
        "engine": engine_name,
        "elapsed_sec": elapsed,
    }


# General arbitrary-question grader.
app.include_router(custom_grading_router)
