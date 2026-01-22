import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# 0. Create output folder
# -----------------------------
os.makedirs("outputs", exist_ok=True)

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv("data/processed/all_tickets_prepared_for_model.csv")
print("Dataset loaded successfully")

# -----------------------------
# 2. TF-IDF Feature Extraction
# -----------------------------
tfidf = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 2)
)

X = tfidf.fit_transform(df["clean_document"])

# =========================================================
# 3. TOPIC (CATEGORY) CLASSIFICATION – NAIVE BAYES
# =========================================================
y_topic = df["Topic_ID"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_topic, test_size=0.2, random_state=42
)

nb_topic = MultinomialNB()
nb_topic.fit(X_train, y_train)

y_pred_topic = nb_topic.predict(X_test)

print("\n=== Naive Bayes : TOPIC CLASSIFICATION ===")
print("Accuracy:", accuracy_score(y_test, y_pred_topic))
print(classification_report(y_test, y_pred_topic))

cm_topic = confusion_matrix(y_test, y_pred_topic)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_topic, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix – Topic Classification (Naive Bayes)")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("outputs/confusion_matrix_topic_naive_bayes.png")
plt.show()

# =========================================================
# 4. PRIORITY CLASSIFICATION – NAIVE BAYES
# =========================================================
y_priority = df["Priority_ID"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_priority, test_size=0.2, random_state=42
)

nb_priority = MultinomialNB()
nb_priority.fit(X_train, y_train)

y_pred_priority = nb_priority.predict(X_test)

print("\n=== Naive Bayes : PRIORITY CLASSIFICATION ===")
print("Accuracy:", accuracy_score(y_test, y_pred_priority))
print(classification_report(y_test, y_pred_priority))

cm_priority = confusion_matrix(y_test, y_pred_priority)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_priority, annot=True, fmt="d", cmap="Greens")
plt.title("Confusion Matrix – Priority Classification (Naive Bayes)")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("outputs/confusion_matrix_priority_naive_bayes.png")
plt.show()

print("\n✅ Naive Bayes evaluation completed. Confusion matrix images saved in 'outputs/' folder.")
