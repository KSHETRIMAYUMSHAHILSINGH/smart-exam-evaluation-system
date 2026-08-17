import pandas as pd
from baseline_scoring import keyword_match_score
from similarity_scoring import semantic_score

def run_comparison(path="data/manifest.csv"):
    df = pd.read_csv(path)
    results = []

    for _, row in df.iterrows():
        keywords = [k.strip() for k in str(row.get("rubric_keywords", "")).split(";") if k.strip()]
        baseline = keyword_match_score(row["student_answer"], keywords)
        semantic = semantic_score(row["student_answer"], row["model_answer"])
        results.append({
            "question_id": row["question_id"],
            "human_score": row.get("human_score"),
            "baseline_score": baseline["score"],
            "semantic_score": semantic["score"],
        })

    results_df = pd.DataFrame(results)

    if results_df["human_score"].notna().any():
        results_df["baseline_error"] = (results_df["baseline_score"] - results_df["human_score"]).abs()
        results_df["semantic_error"] = (results_df["semantic_score"] - results_df["human_score"]).abs()
        print(f"Baseline MAE: {results_df['baseline_error'].mean():.2f}")
        print(f"Semantic MAE: {results_df['semantic_error'].mean():.2f}")

    results_df.to_csv("data/comparison_results.csv", index=False)
    print(results_df)

if __name__ == "__main__":
    run_comparison()