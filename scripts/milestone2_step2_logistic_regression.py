import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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
# 3. TOPIC (CATEGORY) CLASSIFICATION
# =========================================================
y_topic = df["Topic_ID"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_topic, test_size=0.2, random_state=42
)

topic_model = LogisticRegression(max_iter=1000)
topic_model.fit(X_train, y_train)

y_pred_topic = topic_model.predict(X_test)

print("\nLogistic Regression – TOPIC Classification")
print("Accuracy:", accuracy_score(y_test, y_pred_topic))
print(classification_report(y_test, y_pred_topic))

cm_topic = confusion_matrix(y_test, y_pred_topic)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_topic, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix – Topic Classification (Logistic Regression)")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("outputs/confusion_matrix_topic_logistic_regression.png")
plt.show()

# =========================================================
# 4. PRIORITY CLASSIFICATION
# =========================================================
y_priority = df["Priority_ID"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_priority, test_size=0.2, random_state=42
)

priority_model = LogisticRegression(max_iter=1000)
priority_model.fit(X_train, y_train)

y_pred_priority = priority_model.predict(X_test)

print("\nLogistic Regression – PRIORITY Classification")
print("Accuracy:", accuracy_score(y_test, y_pred_priority))
print(classification_report(y_test, y_pred_priority))

cm_priority = confusion_matrix(y_test, y_pred_priority)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_priority, annot=True, fmt="d", cmap="Greens")
plt.title("Confusion Matrix – Priority Classification (Logistic Regression)")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("outputs/confusion_matrix_priority_logistic_regression.png")
plt.show()

print("\n✅ Evaluation completed. Confusion matrix images saved in 'outputs/' folder.")
