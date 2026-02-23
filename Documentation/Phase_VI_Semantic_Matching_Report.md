# Phase VI: Semantic Matching & Ranking Report

## 1. Objective
The goal of this phase was to implement a retrieval engine capable of ranking resumes based on their semantic relevance to a variable job description (JD).

## 2. Methodology

### 2.1 Core Matching Logic
We implemented the `ResumeMatcher` class in `src/matching/matcher.py`. 
- **Preprocessing**: The input JD text is encoded using the `all-MiniLM-L6-v2` Sentence Transformer.
- **Normalization**: The resulting vector is standardized using the `scaler.pkl` fitted during Phase 5 to ensure it exists in the same mathematical space as the resume database.
- **Similarity Computation**: **Cosine Similarity** is calculated between the JD vector and the entire `resume_embeddings.npy` matrix.

### 2.2 Ranking
Results are sorted by their similarity scores. A score of **1.0** would indicate a perfect semantic match, while scores above **0.6** generally represent high relevance in this dataset.

## 3. Demonstration & Results
The engine was validated in `notebooks/06_semantic_matching.ipynb` with the following test cases:
- **Query**: *"Java Developer with experience in Spring Boot"*
- **Result**: Successfully retrieved candidates in the **Information Technology** category with high similarity scores.

## 4. Key Takeaways
- The system is now capable of "fuzzy" matching, finding relevant candidates even if they don't use the exact keywords from the job description.
- The use of pre-computed embeddings ensures that matching 1,000+ resumes takes less than a second.

---
**Status**: COMPLETED  
**Next Phase**: Phase VII — Streamlit Web Application
