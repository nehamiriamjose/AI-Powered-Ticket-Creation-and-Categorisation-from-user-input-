import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
df = pd.read_csv("data/processed/all_tickets_prepared_for_model.csv")

print("Dataset loaded")
print(df.columns)

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 2)
)

# Feature matrix
X = tfidf.fit_transform(df['clean_document'])

print("TF-IDF shape:", X.shape)
print("Sample features:", tfidf.get_feature_names_out()[:10])
