import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
os.makedirs("test_data", exist_ok=True)
os.environ["DATABASE_URL"] = "sqlite:///./test_data/test_exam_eval.db"

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.database import Base, engine
from app.config import settings

client = TestClient(app)
API_KEY = settings.api_key

@pytest.fixture(scope="module", autouse=True)
def cleanup():
    yield
    Base.metadata.drop_all(bind=engine)

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}

def test_list_questions_ok():
    resp = client.get("/questions")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)

def test_create_question_requires_auth():
    resp = client.post("/questions", json={
        "question_id": "TEST1", "question_text": "Test?", "model_answer": "Test",
    })
    assert resp.status_code in (401, 422)

def test_create_and_evaluate_question():
    payload = {
        "question_id": "TEST1", "question_text": "What is 2+2?",
        "model_answer": "The answer is four.", "rubric_keywords": "four;answer",
    }
    resp = client.post("/questions", json=payload, headers={"X-API-Key": API_KEY})
    assert resp.status_code == 200
    assert resp.json()["question_id"] == "TEST1"

    eval_resp = client.post("/evaluate", json={
        "question_id": "TEST1", "student_answer": "The answer is four.",
    }, headers={"X-API-Key": API_KEY})
    assert eval_resp.status_code == 200
    data = eval_resp.json()
    assert "baseline_score" in data and "semantic_score" in data

def test_evaluate_unknown_question_404():
    resp = client.post("/evaluate", json={
        "question_id": "DOES_NOT_EXIST", "student_answer": "whatever",
    }, headers={"X-API-Key": API_KEY})
    assert resp.status_code == 404