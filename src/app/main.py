import streamlit as st
import sys
import os
import joblib
import pandas as pd
import numpy as np
import PyPDF2
import plotly.express as px
from sentence_transformers import SentenceTransformer

# Add the project root to sys.path to import local modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.matching.matcher import ResumeMatcher

# --- Page Configuration ---
st.set_page_config(
    page_title="Resume Screener Pro",
    page_icon="📄",
    layout="wide",
)

# --- Helper Functions ---
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

@st.cache_resource
def load_matcher():
    return ResumeMatcher(
        model_path="models/best_classifier.pkl",
        scaler_path="models/scaler.pkl",
        embeddings_path="data/embeddings/resume_embeddings.npy",
        processed_data_path="data/processed/resumes_cleaned.csv"
    )

@st.cache_resource
def load_classifier():
    model = joblib.load("models/best_classifier.pkl")
    le = joblib.load("models/label_encoder.pkl")
    tfidf = joblib.load("models/tfidf_vectorizer.pkl")
    scaler = joblib.load("models/scaler.pkl")
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    return model, le, tfidf, scaler, st_model

# --- UI Components ---
st.title("📄 AI-Powered Resume Screening Dashboard")
st.markdown("""
Welcome to the professional resume screening tool. This app uses **Neural Embeddings** and **Machine Learning** 
to categorize candidates and match them against your specific job requirements.
""")

tabs = st.tabs(["🔍 Candidate Matching", "🏷️ Resume Classifier"])

# --- Tab 1: Candidate Matching ---
with tabs[0]:
    st.header("Semantic Job-Candidate Matching")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Job Description")
        jd_text = st.text_area("Paste the job description here...", height=300, 
                              placeholder="e.g., We are looking for a Senior Python Developer with experience in Django, AWS, and PostgreSQL...")
        
        top_k = st.slider("Number of top candidates to rank", 1, 20, 5)
        
        if st.button("Rank Candidates", type="primary"):
            if jd_text:
                with st.spinner("Analyzing and ranking resumes..."):
                    try:
                        matcher = load_matcher()
                        results = matcher.match(jd_text, top_k=top_k)
                        st.session_state['match_results'] = results
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Please enter a job description.")

    with col2:
        st.subheader("Ranked Candidates")
        if 'match_results' in st.session_state:
            results = st.session_state['match_results']
            
            # Display metrics
            df_res = pd.DataFrame(results)
            
            # Bar chart of scores
            fig = px.bar(df_res, x='score', y='category', orientation='h', 
                         title="Semantic Relevance Scores",
                         labels={'score': 'Confidence Score', 'category': 'Predicted Fit'},
                         color='score', color_continuous_scale='Magma')
            st.plotly_chart(fig, use_container_width=True)
            
            # Result Details
            for i, res in enumerate(results):
                with st.expander(f"Top {i+1}: {res['category']} (Score: {res['score']})"):
                    st.write(f"**Relevance:** {res['score']*100:.1f}%")
                    st.write("**Snippet:**")
                    st.info(res['text_snippet'])
        else:
            st.info("Rank candidates to see results here.")

# --- Tab 2: Resume Classifier ---
with tabs[1]:
    st.header("Individual Resume Classification")
    
    uploaded_file = st.file_uploader("Upload a resume (PDF or Text)", type=["pdf", "txt"])
    
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            file_text = extract_text_from_pdf(uploaded_file)
        else:
            file_text = uploaded_file.read().decode("utf-8")
        
        st.subheader("Extracted Content Preview")
        st.text(file_text[:500] + "...")
        
        if st.button("Predict Category"):
            with st.spinner("Predicting..."):
                try:
                    model, le, tfidf, scaler, st_model = load_classifier()
                    
                    # 1. TF-IDF
                    features_tfidf = tfidf.transform([file_text])
                    
                    # 2. Embedding
                    features_emb = st_model.encode([file_text])
                    features_emb_scaled = scaler.transform(features_emb)
                    
                    # 3. Fusion
                    from scipy.sparse import hstack
                    features_fused = hstack([features_tfidf, features_emb_scaled])
                    
                    # 4. Prediction
                    pred_idx = model.predict(features_fused)[0]
                    category = le.classes_[pred_idx]
                    
                    # 5. Probabilities
                    try:
                        probs = model.decision_function(features_fused)[0]
                        # Normalize to sum to 1 for visualization
                        exp_probs = np.exp(probs - np.max(probs))
                        probs = exp_probs / exp_probs.sum()
                        
                        top_3_idx = np.argsort(probs)[-3:][::-1]
                        
                        st.success(f"### Predicted Category: **{category}**")
                        
                        st.subheader("Confidence Scores (Top 3)")
                        for idx in top_3_idx:
                            st.progress(float(probs[idx]), text=f"{le.classes_[idx]}: {probs[idx]*100:.1f}%")
                            
                    except:
                        st.success(f"### Predicted Category: **{category}**")
                        
                except Exception as e:
                    st.error(f"Prediction Error: {e}")

st.sidebar.title("About")
st.sidebar.info("""
This dashboard is the final deliverable of the **Resume Screening using NLP** project.
It leverages:
- **TF-IDF** for keyword importance.
- **Sentence Transformers** for semantic context.
- **Hybrid SGD Classifier** (75.25% Acc) for robust sorting.
""")
