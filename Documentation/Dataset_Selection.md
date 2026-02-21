# Dataset Selection & Project Architecture

> **Project**: Resume Screening Using NLP  
> **Date**: February 22, 2026  
> **Phase**: Project Initialization

---

## 1. Dataset Selection Rationale

### 1.1 Task Requirements

The project requires **two datasets** working together:
1. **Resume Dataset** — contains resume text mapped to job categories
2. **Job Description Dataset** — contains structured job postings with descriptions, qualifications, and responsibilities

The system must preprocess both, generate embeddings, compute similarity, and rank resumes against job descriptions.

### 1.2 Resume Dataset

| Property | Value |
|---|---|
| **Source** | [Kaggle — snehaanbhawal/resume-dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) |
| **Size** | ~56 MB (CSV) + PDF copies |
| **Rows** | 2,484 resumes |
| **Columns** | `ID`, `Resume_str`, `Resume_html`, `Category` |
| **Categories** | 24 distinct job categories |
| **License** | CC0: Public Domain |
| **Downloads** | 60,000+ |

**Categories Covered:**
HR, Designer, Information-Technology, Teacher, Advocate, Business-Development, Healthcare, Fitness, Agriculture, BPO, Sales, Consultant, Digital-Media, Automobile, Chef, Finance, Apparel, Engineering, Accountant, Construction, Public-Relations, Banking, Arts, Aviation

**Why This Dataset:**
- **Real-world data** scraped from livecareer.com — contains authentic resume language, formatting artifacts, and HTML noise, forcing a robust NLP preprocessing pipeline
- **Dual format** — both raw string (`Resume_str`) and HTML (`Resume_html`) allow exploring different parsing strategies
- **PDF copies** — enables testing document upload/parsing (bonus requirement)
- **24 categories** — broad cross-industry coverage mirrors real recruiter scenarios
- **2,484 entries** — substantial enough for training embeddings and classifiers with statistical significance
- **CC0 license** — no usage restrictions for academic or commercial applications

**Preprocessing Challenges (Industry-Realistic):**
- HTML tags embedded in `Resume_html` → requires BeautifulSoup cleaning
- Special characters, URLs, and email addresses in text
- Varying resume lengths (some very short, some very detailed)
- Class imbalance across categories → stratified sampling needed

### 1.3 Job Description Dataset

| Property | Value |
|---|---|
| **Source** | [Kaggle — ravindrasinghrana/job-description-dataset](https://www.kaggle.com/datasets/ravindrasinghrana/job-description-dataset) |
| **Size** | ~1.74 GB |
| **File** | `job_descriptions.csv` |
| **Columns** | `Job Id`, `Experience`, `Qualifications`, `Salary Range`, `Location`, `Country`, `Latitude`, `Longitude`, `Work Type`, `Company Size`, `Job Posting Date`, `Preference`, `Contact Person`, `Contact`, `Job Title`, `Role`, `Job Portal`, `Job Description`, `Benefits`, `skills`, `Responsibilities`, `Company`, `Company Profile` |
| **License** | Public |

**Why This Dataset:**
- **Massive scale (1.74 GB)** — industry-level volume that tests scalability of our NLP pipeline
- **Rich structured fields** — job title, role, description, qualifications, skills, and responsibilities provide multiple matching dimensions
- **Metadata-rich** — includes company size, location, salary range, work type for multi-factor ranking
- **Cross-industry coverage** — spans tech, finance, healthcare, marketing, and more
- **Skills column** — enables direct skill-to-skill matching with resume content

### 1.4 Alternatives Considered

| Dataset | Rows | Size | Reason Not Primary |
|---|---|---|---|
| `gauravduttakiit/resume-dataset` | 962 | ~3 MB | Too small; link was also broken at time of download |
| AI-Powered Resume Screening 2025 | 1,000 | ~1 MB | Synthetic data, too small for industry-level work |
| LinkedIn Job Postings 2023 | 124,000 | Large | Licensing restrictions, requires API access |
| Job Descriptions 2025 (Tech & Non-Tech) | 1,100 | ~611 KB | Too small and entirely synthetic |
| Resume Ranking Dataset | Small | Small | Narrow scope, limited categories |

### 1.5 Combined Dataset Strategy

```
Resume Dataset (2,484 resumes, 24 categories)
         ↕ Embedding + Cosine Similarity
Job Description Dataset (massive, multi-industry)
         ↓
Ranked Resume-Job Matches with Justifications
```

The two datasets complement each other:
- **Resumes** provide the candidate pool with category labels for supervised classification
- **Job Descriptions** provide the query side — what employers are looking for
- **Matching** is done via semantic embeddings (Sentence Transformers) and cosine similarity
- **Classification** can also be trained using resume categories to auto-label new resumes

---

## 2. Project Folder Structure

```
Resume_Screening_Using_NLP/
│
├── .github/                        # GitHub configuration
├── .gitignore                      # Git ignore rules (venv, data, models excluded)
├── README.md                       # Project overview
├── requirements.txt                # Python dependencies
├── download_datasets.py            # Dataset download script
├── verify_data.py                  # Dataset verification script
│
├── config/
│   └── config.yaml                 # Centralized project configuration
│                                   #   - Data paths, model hyperparams
│                                   #   - Preprocessing options
│                                   #   - Matching thresholds
│
├── data/
│   ├── raw/                        # Original untouched datasets
│   │   ├── resumes/                # Resume CSV + PDFs (56 MB)
│   │   │   ├── Resume/Resume.csv   #   2,484 rows × 4 cols
│   │   │   └── data/               #   PDF resume files by category
│   │   └── jobs/                   # Job descriptions (1.74 GB)
│   │       └── job_descriptions.csv
│   ├── processed/                  # Cleaned & preprocessed outputs
│   └── embeddings/                 # Generated vector embeddings
│
├── notebooks/                      # Jupyter notebooks for exploration
│   ├── 01_eda_resumes.ipynb        #   Resume data analysis
│   ├── 02_eda_jobs.ipynb           #   Job data analysis
│   ├── 03_preprocessing.ipynb      #   Preprocessing pipeline demo
│   ├── 04_modeling.ipynb           #   Model training & evaluation
│   └── 05_demo.ipynb              #   End-to-end demonstration
│
├── src/                            # Source code package
│   ├── data/
│   │   ├── loader.py               # Config-aware dataset loading
│   │   └── preprocessor.py         # Text cleaning pipeline
│   │                               #   (HTML removal, stopwords, lemmatization)
│   ├── features/
│   │   └── embeddings.py           # Sentence Transformer encoding
│   ├── models/
│   │   ├── similarity.py           # Cosine similarity ranking engine
│   │   └── classifier.py           # Category classification (LR, SGD, RF)
│   ├── evaluation/
│   │   └── metrics.py              # P@K, MRR, NDCG@K metrics
│   └── utils/
│       └── helpers.py              # Config loading, model I/O, utilities
│
├── app/                            # Bonus: Web interface
│   ├── streamlit_app.py            # Streamlit resume upload & matching UI
│   ├── static/                     # CSS, images
│   └── templates/                  # HTML templates
│
├── models/                         # Saved trained models (.pkl, .pt)
├── reports/                        # Generated analysis reports
│   └── figures/                    # Plots and visualizations
│
├── tests/                          # Unit tests
│   ├── test_preprocessor.py
│   ├── test_embeddings.py
│   └── test_similarity.py
│
└── Documentation/
    ├── Project_Understanding.md    # Task description & requirements
    └── Dataset_Selection.md        # This document
```

### 2.1 Design Principles

| Principle | Implementation |
|---|---|
| **Separation of Concerns** | `src/` split into `data`, `features`, `models`, `evaluation`, `utils` |
| **Reproducibility** | `config/config.yaml` centralizes all parameters; `requirements.txt` pins dependencies |
| **Data Isolation** | Raw data never modified; processed outputs go to `data/processed/` |
| **Scalability** | Modular code in `src/` can be imported into notebooks, scripts, or the web app |
| **Testability** | `tests/` directory with dedicated test files per module |
| **Documentation** | `Documentation/` folder + inline docstrings in every module |

### 2.2 Key Configuration (config.yaml)

```yaml
model:
  embedding_model: "all-MiniLM-L6-v2"   # Fast, high-quality embeddings
  max_seq_length: 512
  batch_size: 32

preprocessing:
  remove_html: true
  lowercase: true
  remove_stopwords: true
  lemmatize: true

matching:
  method: "cosine"
  top_k: 10
  similarity_threshold: 0.5
```

---

## 3. Virtual Environment & Dependencies

### Core Stack

| Category | Packages | Purpose |
|---|---|---|
| **Data Processing** | pandas, numpy | DataFrame operations, numerical computing |
| **NLP** | nltk, spacy, beautifulsoup4 | Tokenization, stopwords, HTML cleaning |
| **Embeddings** | sentence-transformers, transformers, torch | Dense vector representations |
| **ML** | scikit-learn | Classification, cosine similarity, metrics |
| **Visualization** | matplotlib, seaborn, wordcloud, plotly | EDA plots and reports |
| **Web UI** | streamlit | Interactive resume screening interface |
| **Document Parsing** | PyPDF2, python-docx | PDF/DOCX resume upload support |
| **Utilities** | pyyaml, tqdm, jupyter | Config, progress bars, notebooks |

### Setup Commands

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Download datasets
python download_datasets.py
```
