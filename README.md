# Smart Exam Evaluation System

ML-assisted exam evaluation system with descriptive-answer
grading, arbitrary-question grading, handwriting OCR and MCQ
auto-grading.

## Current architecture

- FastAPI backend — port 8000
- Custom frontend V6.3 — port 5500
- PP-OCRv5 service — port 8001
- Qwen3.5-4B MLX runtime — dynamic local port

## Grading

The general descriptive grader uses the frozen Qwen V3.3.7
pipeline with rubric generation, evidence validation and
contradiction-aware scoring.

The Q1-Q5 benchmark also includes the project-trained
GradientBoosting semantic-hybrid comparison model.

## OCR

Fast mode uses PP-OCRv5.

Accurate mode uses Qwen3.5 Vision OCR.

Tesseract is retained as the emergency Fast-OCR fallback.

## Qwen model

Large Qwen weights are intentionally not stored in Git.

The runtime expects the compatible model at:

models/Qwen3.5-4B-MLX-4bit

## Backend

From the project root:

    source venv/bin/activate
    uvicorn app.main:app --host 127.0.0.1 --port 8000

## PP-OCRv5

From paddle-ocr-test:

    source venv/bin/activate
    uvicorn paddle_ocr_service:app --host 127.0.0.1 --port 8001

## Frontend

From the project root:

    source venv/bin/activate
    python smart-exam-frontend-v6.3/serve.py

Open:

http://127.0.0.1:5500

## Privacy

The GitHub database contains the benchmark questions and MCQs
but does not contain stored student evaluation history.

Never commit `.env` or local Qwen model weights.
