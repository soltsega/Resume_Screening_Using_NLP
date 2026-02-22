# Phase IV: Feature Engineering Report

## 1. Overview
Phase IV focused on converting the cleaned text data into numerical representations that can be processed by machine learning models. We implemented a dual-path approach: **Sparse Vectorization** (TF-IDF) for classification and **Dense Embeddings** (Sentence Transformers) for semantic matching.

## 2. Methodology

### A. TF-IDF Vectorization (Keyword-Based)
We implemented TF-IDF to capture the importance of specific terms within the resume and job corpora.
- **Tokens**: Combined unigrams and bigrams to capture phrases like "Machine Learning" or "Business Development".
- **Vocabulary Size**: Restricted to the top **10,000 features** to maintain computational efficiency while capturing 95%+ of meaningful professional terminology.
- **Persistence**: The vectorizer is saved as `models/tfidf_vectorizer.pkl` to ensure consistency during inference in the Streamlit application.

### B. Dense Embeddings (Semantic-Based)
To overcome the "Language Gap" identified in Phase II, we utilized dense vector representations.
- **Model**: `all-MiniLM-L6-v2` (Sentence Transformers). This model strikes an optimal balance between embedding quality (384 dimensions) and inference speed.
- **Application**: 
  - Entire cleaned resumes were embedded into a high-dimensional vector space.
  - Job descriptions (Combined: description + skills + qualifications) were embedded into the same vector space.
- **Persistence**: Embeddings are stored as NumPy arrays (`.npy`) in `data/embeddings/` for rapid loading during the matching phase.

## 3. Visualization & Validation (`notebooks/04_feature_engineering.ipynb`)

### A. t-SNE Analysis
We applied t-SNE (t-Distributed Stochastic Neighbor Embedding) to a sample of resume embeddings to visualize the underlying structure.
- **Clustering**: The visualization confirmed that similar professional categories (e.g., "Data Science" vs. "HR") form distinct clusters in the 384-dimensional space.
- **Overlap**: Identified intentional overlap between adjacent fields (e.g., "Finance" and "Accounting"), validating that the model captures semantic relationships.

### B. Unit Testing
Created `tests/test_embeddings.py` to ensure:
- Model loading integrity.
- Expected vector dimensionality (384).
- Reliable I/O operations for NumPy files.

## 4. Summary of Output Artifacts
| Artifact | Path | Purpose |
| :--- | :--- | :--- |
| **TF-IDF Vectorizer** | `models/tfidf_vectorizer.pkl` | Keyword feature extraction for classifiers. |
| **Resume Embeddings** | `data/embeddings/resume_embeddings.npy` | Semantic search index for candidates. |
| **Job Embeddings** | `data/embeddings/job_embeddings.npy` | Semantic search index for roles. |

## 5. Conclusion
With both sparse and dense features generated, the project has the necessary mathematical foundation for both:
1. **Classification**: Categorizing resumes into industries (Phase V).
2. **Ranking**: Calculating cosine similarity between job requirements and candidate profiles (Phase VI).
