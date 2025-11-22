import json
import os
import sys

# Fix OpenMP conflict on macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import MODEL_NAME, DATA_DIR

def evaluate_model(model_path=None, qa_file='qa_pairs.json', k_values=[1, 3, 5]):
    """
    Evaluates the model (base or fine-tuned) on the QA dataset using Recall@k.
    """
    qa_path = os.path.join(DATA_DIR, qa_file)
    if not os.path.exists(qa_path):
        print(f"Dataset not found at {qa_path}")
        return

    # Load Data
    with open(qa_path, 'r') as f:
        data = json.load(f)

    questions = [item['question'] for item in data]
    answers = [item['answer'] for item in data]

    # Load Model
    if model_path and os.path.exists(model_path):
        print(f"Loading fine-tuned model from {model_path}")
        model = SentenceTransformer(model_path)
    else:
        print(f"Loading base model {MODEL_NAME}")
        model = SentenceTransformer(MODEL_NAME)

    # Create Index from Answers (Simulating retrieval from corpus)
    print("Encoding answers...")
    answer_embeddings = model.encode(answers)
    index = faiss.IndexFlatL2(answer_embeddings.shape[1])
    index.add(np.array(answer_embeddings).astype('float32'))

    print("Encoding questions...")
    question_embeddings = model.encode(questions)

    # Search
    print("Evaluating...")
    print(f"Query shape: {question_embeddings.shape}, Index size: {index.ntotal}")
    D, I = index.search(np.array(question_embeddings).astype('float32'), k=max(k_values))

    # Compute Metrics
    results = {}
    for k in k_values:
        correct = 0
        for i, indices in enumerate(I):
            # The correct answer index for question i is i (since we indexed answers in order)
            if i in indices[:k]:
                correct += 1
        recall = correct / len(questions)
        results[f"Recall@{k}"] = recall
        print(f"Recall@{k}: {recall:.4f}")

    return results

if __name__ == "__main__":
    print("--- Base Model Evaluation ---")
    evaluate_model()
    
    fine_tuned_path = os.path.join(os.path.dirname(DATA_DIR), 'models/fine_tuned_model')
    if os.path.exists(fine_tuned_path):
        print("\n--- Fine-tuned Model Evaluation ---")
        evaluate_model(model_path=fine_tuned_path)
