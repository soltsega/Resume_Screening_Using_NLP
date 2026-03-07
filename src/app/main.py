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
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    model = joblib.load(os.path.join(root, "models/best_classifier.pkl"))
    le = joblib.load(os.path.join(root, "models/label_encoder.pkl"))
    tfidf = joblib.load(os.path.join(root, "models/tfidf_vectorizer.pkl"))
    scaler = joblib.load(os.path.join(root, "models/scaler.pkl"))
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    return model, le, tfidf, scaler, st_model

# --- Custom CSS for Premium Look ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&family=Inter:wght@300;400;600&display=swap');

    :root {
        --primary: #EB1D36;
        --bg-dark: #0A192F;
        --card-bg: rgba(23, 42, 69, 0.7);
        --text-main: #CCD6F6;
        --text-dim: #8892B0;
    }

    .main {
        background-color: var(--bg-dark);
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        color: var(--text-main) !important;
        font-weight: 600 !important;
    }

    .stButton>button {
        background-color: var(--primary) !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(235, 29, 54, 0.3) !important;
    }

    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(235, 29, 54, 0.5) !important;
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: var(--card-bg);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 2rem;
        margin-bottom: 1.5rem;
    }

    .metric-card {
        text-align: center;
        padding: 1.5rem;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 10px;
        border-left: 4px solid var(--primary);
    }

    .result-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: transform 0.2s ease;
    }

    .result-card:hover {
        transform: scale(1.02);
        background: rgba(255, 255, 255, 0.08);
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: transparent !important;
        border-bottom: 2px solid transparent !important;
        color: var(--text-dim) !important;
        font-weight: 600 !important;
    }

    .stTabs [aria-selected="true"] {
        color: var(--primary) !important;
        border-bottom: 2px solid var(--primary) !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: var(--bg-dark); }
    ::-webkit-scrollbar-thumb { background: #172A45; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--primary); }
</style>
""", unsafe_allow_html=True)

# --- UI Components ---
with st.container():
    col_t1, col_t2 = st.columns([3, 1])
    with col_t1:
        st.title("📄 AI Resume Intelligence")
        st.markdown(f'<p style="color: #8892B0; font-size: 1.1rem; margin-top: -10px;">Professional Semantic Screening & Classification Engine</p>', unsafe_allow_html=True)
    with col_t2:
        st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
        # st.image(..., width=120) # Placeholder for logo

tabs = st.tabs(["🔍 Semantic Matcher", "🏷️ Skill Classifier", "📊 Analytics"])

# --- Tab 1: Candidate Matching ---
with tabs[0]:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1.5], gap="large")
    
    with col1:
        st.subheader("Job Specification")
        jd_text = st.text_area("Paste descriptions or requirements...", height=250, 
                             placeholder="e.g., We are looking for a Senior Python Developer...")
        
        top_k = st.select_slider("Ranking Depth", options=[3, 5, 10, 15, 20], value=5)
        
        if st.button("Start AI Ranking", type="primary", use_container_width=True):
            if jd_text:
                with st.spinner("Processing neural embeddings..."):
                    try:
                        matcher = load_matcher()
                        results = matcher.match(jd_text, top_k=top_k)
                        st.session_state['match_results'] = results
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Please provide a job description.")

    with col2:
        st.subheader("Neural Match Results")
        if 'match_results' in st.session_state:
            results = st.session_state['match_results']
            
            # Summary Metrics
            m1, m2 = st.columns(2)
            with m1:
                st.markdown(f'<div class="metric-card"><h4>Top Match</h4><h3>{results[0]["category"]}</h3></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-card"><h4>Avg Score</h4><h3>{np.mean([r["score"] for r in results]):.2f}</h3></div>', unsafe_allow_html=True)
            
            st.markdown('<div style="height: 20px;"></div>', unsafe_allow_html=True)
            
            # Result Details
            for i, res in enumerate(results):
                with st.expander(f"RANK {i+1} | {res['category']} (Match: {res['score']*100:.1f}%)"):
                    st.write("**Candidate Profile Snippet:**")
                    st.info(res['text_snippet'])
                    st.progress(float(res['score']) if res['score'] > 0 else 0.0)
        else:
            st.info("Input requirements and start ranking to view candidate profiles.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- Tab 2: Resume Classifier ---
with tabs[1]:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Individual Assessment")
    
    up_col, res_col = st.columns([1, 1], gap="large")
    
    with up_col:
        uploaded_file = st.file_uploader("Upload Dossier (PDF/TXT)", type=["pdf", "txt"])
        if uploaded_file:
            st.success("File Received")
    
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            file_text = extract_text_from_pdf(uploaded_file)
        else:
            file_text = uploaded_file.read().decode("utf-8")
        
        with res_col:
            if st.button("Execute Classification", use_container_width=True):
                with st.spinner("Analyzing semantics..."):
                    try:
                        model, le, tfidf, scaler, st_model = load_classifier()
                        features_tfidf = tfidf.transform([file_text])
                        features_emb = st_model.encode([file_text])
                        features_emb_scaled = scaler.transform(features_emb)
                        
                        from scipy.sparse import hstack
                        features_fused = hstack([features_tfidf, features_emb_scaled])
                        pred_idx = model.predict(features_fused)[0]
                        category = le.classes_[pred_idx]
                        
                        st.markdown(f"""
                        <div style="background: rgba(235, 29, 54, 0.1); padding: 1.5rem; border-radius: 12px; border: 1px solid var(--primary);">
                            <h4 style="margin:0; color:var(--text-main);">PREDICTED DOMAIN</h4>
                            <h2 style="margin:0; color:var(--primary);">{category}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Probabilities
                        try:
                            probs = model.decision_function(features_fused)[0]
                            exp_probs = np.exp(probs - np.max(probs))
                            probs = exp_probs / exp_probs.sum()
                            top_3_idx = np.argsort(probs)[-3:][::-1]
                            
                            st.markdown('<div style="height: 15px;"></div>', unsafe_allow_html=True)
                            st.write("**Top Probability Distribution**")
                            for idx in top_3_idx:
                                st.progress(float(probs[idx]), text=f"{le.classes_[idx]}: {probs[idx]*100:.1f}%")
                        except: pass
                    except Exception as e:
                        st.error(f"Error: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

# --- Tab 3: Analytics ---
with tabs[2]:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Semantic Landscape")
    if 'match_results' in st.session_state:
        results = st.session_state['match_results']
        df_res = pd.DataFrame(results)
        
        fig = px.bar(df_res, x='score', y='category', orientation='h', 
                     title="Score Distribution across Domains",
                     labels={'score': 'Semantic Relevance', 'category': 'Predicted Category'},
                     color='score', color_continuous_scale='Bluered_r')
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#CCD6F6',
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run a matching query to see semantic analytics.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem;">
    <h2 style="color: #EB1D36 !important;">RSU NLP</h2>
    <p style="color: #8892B0;">v2.0 Premium Dashboard</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.info("""
**Core Architecture:**
- **TF-IDF** Kernel
- **BERT** Embeddings
- **Hybrid SGD** Classifier
- **Cosine** Similarity Ranking
""")

st.sidebar.markdown("---")
st.sidebar.write("Developed by **Soltsega**")
st.sidebar.markdown("[LinkedIn](https://linkedin.com) | [GitHub](https://github.com)")
