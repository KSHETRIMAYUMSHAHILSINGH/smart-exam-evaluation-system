import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from app.database import engine, SessionLocal, Base
from app.models_db import Question, Evaluation

Base.metadata.create_all(bind=engine)

def migrate(path="data/manifest.csv"):
    df = pd.read_csv(path)
    db = SessionLocal()
    
    # Track which questions we've already added to avoid the UNIQUE constraint error
    added_questions = set()

    for _, row in df.iterrows():
        qid = row["question_id"]
        
        # Only add the question to the Question table the first time we see it
        if qid not in added_questions:
            db.add(Question(
                question_id=qid,
                question_text=row["question_text"],
                model_answer=row["model_answer"],
                rubric_keywords=row.get("rubric_keywords", ""),
            ))
            added_questions.add(qid)
            
        # Add every student answer to the Evaluation table
        db.add(Evaluation(
            question_id=qid,
            student_answer=row["student_answer"],
            human_score=row.get("human_score"),
        ))

    db.commit()
    db.close()
    print(f"Migrated {len(added_questions)} questions and {len(df)} evaluations into the database.")

if __name__ == "__main__":
    migrate()