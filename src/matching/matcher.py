import numpy as np
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

class ResumeMatcher:
    def __init__(self, 
                 model_path="models/best_classifier.pkl",
                 scaler_path="models/scaler.pkl",
                 embeddings_path="data/embeddings/resume_embeddings.npy",
                 processed_data_path="data/processed/resumes_cleaned.csv"):
        
        print("Initializing Semantic Matcher...")
        # Resolve paths relative to the project root
        self.root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        
        abs_model_path = os.path.join(self.root, model_path)
        abs_scaler_path = os.path.join(self.root, scaler_path)
        abs_embeddings_path = os.path.join(self.root, embeddings_path)
        abs_processed_data_path = os.path.join(self.root, processed_data_path)

        # Load necessary artifacts
        self.st_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.scaler = joblib.load(abs_scaler_path)
        
        # Load pre-computed resume embeddings
        if os.path.exists(abs_embeddings_path):
            self.resume_embeddings = np.load(abs_embeddings_path)
        else:
            raise FileNotFoundError(f"Resume embeddings not found at {abs_embeddings_path}. Please check your data directory.")
            
        # Load processed data for mapping results
        self.df = pd.read_csv(abs_processed_data_path)
        
    def match(self, job_description, top_k=5):
        """
        Calculates similarity between a job description and all resumes.
        Returns the top_k matching resumes with their metadata and scores.
        """
        # 1. Transform JD to Embedding
        print(f"Generating embedding for job description...")
        jd_embedding = self.st_model.encode([job_description])
        
        # 2. Scale Embedding (must match the model's training space)
        jd_embedding_scaled = self.scaler.transform(jd_embedding)
        
        # 3. Compute Cosine Similarity
        # Reshaping jd_embedding_scaled for cosine_similarity (1, N)
        scores = cosine_similarity(jd_embedding_scaled, self.resume_embeddings)[0]
        
        # 4. Get Top K Indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # 5. Format Results
        results = []
        for idx in top_indices:
            results.append({
                "resume_id": idx,
                "score": round(float(scores[idx]), 4),
                "category": self.df.iloc[idx]['Category'],
                "text_snippet": self.df.iloc[idx]['cleaned_text'][:200] + "..." 
            })
            
        return results

if __name__ == "__main__":
    # Quick Test Loop
    matcher = ResumeMatcher()
    
    test_jd = "Looking for an experienced Data Scientist with proficiency in Python, SQL, and Machine Learning. Knowledge of NLP is a plus."
    print(f"\nTesting Matcher with JD: '{test_jd}'")
    
    matches = matcher.match(test_jd, top_k=3)
    
    print("\n--- Top Matches ---")
    for i, m in enumerate(matches):
        print(f"{i+1}. [Score: {m['score']}] [Category: {m['category']}]")
        print(f"   Snippet: {m['text_snippet']}\n")
