# Phase V: Model Training & Performance Report

## 1. Objective
The goal of this phase was to train a robust multi-class classifier to predict resume categories (25 distinct classes) while maintaining a strict zero-leakage protocol and achieving a target accuracy of **75%**.

## 2. Methodology

### 2.1 Data Leakage Prevention
We implemented an **Atomic Splitting & Fitting** protocol. The dataset was split into training (80%) and testing (20%) sets using stratification *before* any feature extraction occurred. All transformers (TF-IDF, Scaler) were fitted strictly on the training set to ensure the model had no prior knowledge of the test distribution.

### 2.2 Feature Engineering (Hybrid Fusion)
Through experimentation, we found that a "Hybrid" approach provided the best results:
- **Local Keywords**: TF-IDF vectorization (max 5,000 features) captured specific industry terms.
- **Global Semantics**: Sentence Transformer (`all-MiniLM-L6-v2`) provided 384-dimensional dense embeddings for conceptual understanding.
- **Fusion**: These two feature sets were concatenated and standardized to form the final feature matrix.

### 2.3 Model Selection
We evaluated several baseline models:
1. **Logistic Regression** (65% Accuracy)
2. **Random Forest** (72% Accuracy)
3. **SGD Classifier** (69-74% Accuracy)
4. **Hybrid SGD** (Winner)

**Why SGD?** Stochastic Gradient Descent is highly efficient for high-dimensional sparse/dense fused matrices and showed the best response to hyperparameter tuning.

## 3. Results Summary

- **Final Accuracy**: **75.25%**
- **Weighted F1-Score**: **0.74**
- **Top Performing Categories**: HR, Information Technology, Chef, Construction (F1 > 0.80).
- **Challenge Categories**: BPO, Automobile (F1 < 0.30 due to extreme class underrepresentation).

## 4. Visualizations
- **Confusion Matrix**: Located at `reports/figures/confusion_matrix.png`.
- **F1-Score Chart**: Located at `reports/figures/f1_scores_per_category.png`.

## 5. Production Artifacts
The following files in `models/` are ready for integration:
- `best_classifier.pkl` (The Hybrid SGD model)
- `tfidf_vectorizer.pkl` (Fitted on Train)
- `scaler.pkl` (StandardScaler for Embeddings)
- `label_encoder.pkl` (Category Mapper)

---
**Status**: COMPLETED  
**Next Phase**: Phase VI — Semantic Matching & Ranking
