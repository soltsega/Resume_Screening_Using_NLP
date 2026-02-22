# Phase II: Exploratory Data Analysis (EDA) Report

## 1. Overview
Phase II focused on deeply understanding the two primary data sources: the Resume dataset and the Job Description dataset. The goal was to identify patterns, data quality issues, and the "vocabulary gap" between candidates and recruiters.

## 2. Resume Dataset Analysis (`01_eda_resumes.ipynb`)
The analysis of 2,484 resumes across 24 professional categories revealed:
- **Category Balance**: The dataset is relatively balanced, with Information Technology and Business being the largest sectors.
- **Terminology**: Frequent unigrams and bigrams are heavily focused on software tools (MS Office, SQL) and administrative tasks.
- **Cosine Similarity**: High similarity (~0.8+) was found between Business and Management roles, while technical roles like Engineering are more isolated in their terminology.
- **Data Quality**: ~2% of resumes were identified as "low-content" (under 50 words), suggesting these entries may need exclusion during training to avoid noise.

## 3. Job Description Analysis (`02_eda_jobs.ipynb`)
Using a sample of 50,000 job descriptions, we observed:
- **Skill Anchoring**: Job descriptions are highly standardized. Specific roles (e.g., 'Data Scientist') have extremely consistent skill requirements (Tableau, Hadoop, Python).
- **Complexity**: Job descriptions are significantly longer on average than resumes and include sections on benefits, company culture, and legal compliance, which adds noise to the "technical" signal.

## 4. Comparative Analysis & Language Gap (`03_eda_comparative.ipynb`)
This was the most critical analytical stage, uncovering the following:
- **Vocabulary Overlap**: There is a significant mismatch in word frequency. Jobs focus on *results-oriented verbs* (implementing, optimizing, ensuring), while resumes focus on *static toolsets* and *historical descriptions*.
- **Overlap Statistics**: In the top 50 terms of both datasets, only 15 terms (30%) were common. 
- **Sector Gap**: In specific sectors like Data Science, Job Descriptions prioritize "Data Analysis" and "Machine Learning" frequency significantly higher than candidates do in their resume text.

## 5. Key Insights for Phase III & IV
1. **Preprocessing is Critical**: Stop-word removal must be aggressive to remove noise from job benefits and generic administrative terms in resumes.
2. **Beyond Keywords**: A simple TF-IDF similarity might fail due to the language gap. The use of **Dense Embeddings** (Word2Vec/BERT) is recommended to capture the semantic relationship between different descriptions of the same skill.
3. **Filtering**: Resumes with extremely low word counts will be removed to maintain the quality of the training set.

## 6. Conclusion
Phase II successfully identified that matching resumes to jobs is not just a keyword search but a "translation" task. The findings provide a clear roadmap for the feature engineering techniques required in the next phases.
