from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_score(student_answer, model_answer, max_score=10):
    # Split the model answer by '||' to allow multiple valid references
    possible_answers = [ans.strip() for ans in model_answer.split('||')]
    
    # Encode the student answer once
    student_emb = model.encode([student_answer], convert_to_tensor=True)
    
    best_score = 0
    best_similarity = 0
    
    # Compare against each reference answer and keep the highest score
    for ans in possible_answers:
        ans_emb = model.encode([ans], convert_to_tensor=True)
        similarity = util.cos_sim(student_emb, ans_emb).item()
        score = round(similarity * max_score, 2)
        
        if score > best_score:
            best_score = score
            best_similarity = similarity

    return {"score": best_score, "max_score": max_score, "similarity": round(best_similarity * 100, 1)}