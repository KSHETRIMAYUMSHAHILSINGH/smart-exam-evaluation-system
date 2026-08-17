import pandas as pd
import numpy as np

from baseline_scoring import keyword_match_score
from similarity_scoring import semantic_score
from trained_grader import trained_score
from train_hybrid_grader import split_by_question

from sklearn.metrics import mean_absolute_error, mean_squared_error


df = pd.read_csv("data/training_dataset.csv")
df = split_by_question(df)
test = df[df["split"] == "test"].copy()

results = []

for _, row in test.iterrows():

    keywords = [
        k.strip()
        for k in str(row["rubric_keywords"]).split(";")
        if k.strip()
    ]

    baseline = keyword_match_score(
        row["student_answer"],
        keywords
    )["score"]

    semantic = semantic_score(
        row["student_answer"],
        row["model_answer"]
    )["score"]

    trained = trained_score(
        row["question_text"],
        row["model_answer"],
        row["student_answer"],
        row["rubric_keywords"]
    )["score"]

    results.append({
        "question_id": row["question_id"],
        "human_score": row["human_score"],
        "keyword_model": baseline,
        "semantic_model": semantic,
        "trained_model": trained
    })


result = pd.DataFrame(results)

print("\nMODEL COMPARISON\n")
print(result)

y = result["human_score"]

for model in ["keyword_model", "semantic_model", "trained_model"]:

    pred = result[model]

    mae = mean_absolute_error(y, pred)
    rmse = np.sqrt(mean_squared_error(y, pred))

    print(f"\n{model}")
    print(f"MAE  : {mae:.3f}")
    print(f"RMSE : {rmse:.3f}")


result.to_csv(
    "data/final_model_comparison.csv",
    index=False
)

print("\nSaved: data/final_model_comparison.csv")