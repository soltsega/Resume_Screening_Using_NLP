import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sentence_transformers import SentenceTransformer
from scipy.sparse import hstack
import os

def generate_visualizations():
    print("Loading model and artifacts...")
    model = joblib.load("models/best_classifier.pkl")
    tfidf = joblib.load("models/tfidf_vectorizer.pkl")
    scaler = joblib.load("models/scaler.pkl")
    le = joblib.load("models/label_encoder.pkl")

    print("Preparing test data split...")
    df = pd.read_csv("data/processed/resumes_cleaned.csv")
    X_text = df['cleaned_text'].fillna('')
    y = df['Category']
    
    # Use exact same split as training
    _, X_test_raw, _, y_test_raw = train_test_split(
        X_text, y, test_size=0.20, stratify=y, random_state=42
    )
    y_test = le.transform(y_test_raw)

    print("Generating features for test data...")
    X_test_tfidf = tfidf.transform(X_test_raw)
    
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    X_test_emb = st_model.encode(X_test_raw.tolist(), show_progress_bar=True)
    X_test_emb_scaled = scaler.transform(X_test_emb)
    
    X_test_fused = hstack([X_test_tfidf, X_test_emb_scaled])

    print("Predicting...")
    y_pred = model.predict(X_test_fused)
    acc = accuracy_score(y_test, y_pred)
    
    # 1. Plot Confusion Matrix
    print("Plotting Confusion Matrix...")
    plt.figure(figsize=(18, 14))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap='Blues')
    plt.title(f'Confusion Matrix (Accuracy: {acc:.4f})', fontsize=16)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    
    os.makedirs("reports/figures", exist_ok=True)
    plt.savefig("reports/figures/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Extract and Save Classification Report as CSV for better analysis
    report_dict = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv("reports/classification_report.csv")
    
    # 3. Plot F1-Scores per Category
    print("Plotting F1-Scores...")
    f1_scores = report_df.iloc[:-3]['f1-score'].sort_values(ascending=False)
    plt.figure(figsize=(12, 10))
    sns.barplot(x=f1_scores.values, y=f1_scores.index, palette='viridis')
    plt.title('F1-Score per Category (Hybrid Model)', fontsize=15)
    plt.xlabel('F1-Score')
    plt.axvline(x=0.75, color='red', linestyle='--', label='75% Target')
    plt.legend()
    plt.savefig("reports/figures/f1_scores_per_category.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("Visualizations saved to reports/figures/")

if __name__ == "__main__":
    generate_visualizations()
