"""
FINAL MODEL SELECTION FILE
--------------------------
After evaluating Logistic Regression, SVM, Naive Bayes,
Random Forest, and XGBoost, Logistic Regression was selected
as the FINAL model based on accuracy, interpretability, and
deployment efficiency.
"""

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load processed dataset
df = pd.read_csv("data/processed/all_tickets_prepared_for_model.csv")

X = df["clean_document"]
y_topic = df["Topic_ID"]
y_priority = df["Priority_ID"]

# TF-IDF
tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
X_tfidf = tfidf.fit_transform(X)

# Train FINAL Topic model
topic_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)
topic_model.fit(X_tfidf, y_topic)

# Train FINAL Priority model
priority_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)
priority_model.fit(X_tfidf, y_priority)

# Save models
joblib.dump(topic_model, "models/final_topic_model.pkl")
joblib.dump(priority_model, "models/final_priority_model.pkl")
joblib.dump(tfidf, "models/final_tfidf_vectorizer.pkl")

print("✅ FINAL Logistic Regression models saved successfully")
