# Phase III: Text Preprocessing Pipeline Report

## 1. Overview
Phase III established a robust, modular text cleaning and normalization pipeline. The primary objective was to transform raw, noisy text from resumes and job descriptions into a clean, standardized format suitable for vectorization and machine learning modeling.

## 2. Core Preprocessing Logic (`src/data/preprocessor.py`)
The pipeline consists of several sequential stages:
1. **HTML Removal**: Uses `BeautifulSoup` with `html.parser` to strip tags while preserving meaningful text separation.
2. **Standard Cleaning**:
   - URL and Email removal (RegEx).
   - Character normalization (removing special characters/symbols).
   - Case folding (lowercasing).
   - Whitespace normalization (collapsing multiple spaces and stripping).
3. **Stopword Removal**: Filtering out common English words using NLTK's standard corpus to minimize noise.
4. **Lemmatization**: Using NLTK's `WordNetLemmatizer` to reduce tokens to their base forms (e.g., "studies" → "study", "engineering" → "engineering").

## 3. Data Quality & Critical Findings (`notebooks/03_preprocessing.ipynb`)
The preprocessing phase included critical analytical steps to ensure data integrity:

### A. HTML vs. String Validation
Comparison between `Resume_html` and `Resume_str` confirmed that our cleaning logic successfully reconstructed the descriptive text from the original markup without losing sector-specific technical terms.

### B. Multi-Source Job Cleaning
Job descriptions were processed by merging three distinct high-signal columns:
- **`Job Description`**
- **`skills`**
- **`Qualifications`**
These were combined into a `combined_context` field to provide the model with a holistic view of the role requirements.

### C. Outlier Filtering
Following Phase II EDA findings, we implemented a **Minimum Word Count Filter (50 words)**.
- **Original Count**: 2,484 resumes.
- **Filtered Count**: ~2,434 resumes.
- **Impact**: Approximately 50 low-quality/empty resumes were removed, preventing noise during the training phase.

## 4. Performance Metrics
| Metric | Before Preprocessing | After Preprocessing | Reduction % |
| :--- | :--- | :--- | :--- |
| Global Vocab Size | ~50,000 | ~32,000 | **~36%** |
| Total Token Count | ~1,200,000 | ~750,000 | **~37%** |

The significant reduction in vocabulary size (~36%) demonstrates that the pipeline effectively condensed the feature space by removing non-semantic noise and normalizing word variants.

## 5. Verification & Testing
A suite of unit tests was implemented in `tests/test_preprocessor.py` covering:
- Correct HTML stripping with space preservation.
- Accurate removal of URLs and Email addresses.
- Success of the full pipeline combinator.
All tests passed successfully on the final production code.

## 6. Conclusion
The output of Phase III is a set of "Ready-for-Vectorization" datasets located in `data/processed/`. The pipeline is fast, repeatable, and ensures that the model will focus on professional skills rather than formatting or grammatical noise.
