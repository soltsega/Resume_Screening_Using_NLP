# AI-Powered Resume Screening & Ranking System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://soltsega-resume-screening-using-nlp-srcappmain-phase-vii-ec3qps.streamlit.app/)

An end-to-end NLP project that categorizes resumes into 25 industry segments and ranks candidates against Job Descriptions (JDs) using neural semantic matching.

## Key Features
- **Intelligent Classification**: Automatically classifies resumes (PDF/TXT) with **75.25% accuracy**.
- **Semantic Matching**: Ranks candidates based on semantic meaning using **Cosine Similarity**, not just keyword matches.
- **Hybrid NLP Pipeline**: Fuses **TF-IDF** (keyword importance) with **Dense Embeddings** (contextual understanding).
- **Leakage-Proof Pipeline**: Robust train/test isolation protocol ensured high model integrity.

## 🛠️ Project Structure
```text
Resume_Screening_Using_NLP/
├── data/
│   ├── raw/                 # Original dataset
│   ├── processed/           # Cleaned and engineered text
│   └── embeddings/          # Pre-computed neural embeddings
├── models/                  # Saved .pkl artifacts (Classifier, Scaler, TF-IDF)
├── notebooks/               # Exploratory Research (EDA, Feature Engineering, Modeling)
├── src/
│   ├── preprocessing/       # Text cleaning utility
│   ├── models/              # Training and optimization scripts
│   ├── matching/            # Semantic similarity engine
│   └── app/                 # Streamlit dashboard
└── Documentation/           # Detailed phase-by-phase reports
```

## Tech Stack
- **Languages**: Python 3.11
- **Preprocessing**: NLTK, Regex
- **Feature Engineering**: Scikit-Learn (TF-IDF), Sentence-Transformers (Neural)
- **Modeling**: Hybrid SGD Classifier (Linear SVM with Hinge Loss)
- **UI/UX**: Streamlit, Plotly, Seaborn

## How to Run

### 1. Prerequisites
Ensure you have the dependencies installed:
```bash
pip install pandas numpy scikit-learn sentence-transformers streamlit PyPDF2 plotly matplotlib seaborn
```

### 2. Launch the Dashboard
Run the following command from the project root:
```bash
streamlit run src/app/main.py
```
Once launched, the dashboard will be available at:  
👉 **[http://localhost:8501](http://localhost:8501)**

### 3. Using the App
- **Tab 1 (Candidate Matching)**: Paste a Job Description to rank the existing resume database.
- **Tab 2 (Resume Classifier)**: Upload a single resume to predict its industry category instantly.

## Performance
| Model | Accuracy | Weighted F1-Score |
| :--- | :--- | :--- |
| **Hybrid SGD (Final)** | **75.25%** | **0.74** |
| Random Forest (Baseline) | 72.03% | 0.70 |

---


