import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
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
# 3. TOPIC (CATEGORY) CLASSIFICATION – SVM
# =========================================================
y_topic = df["Topic_ID"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_topic, test_size=0.2, random_state=42
)

svm_topic = LinearSVC()
svm_topic.fit(X_train, y_train)

y_pred_topic = svm_topic.predict(X_test)

print("\n=== SVM : TOPIC CLASSIFICATION ===")
print("Accuracy:", accuracy_score(y_test, y_pred_topic))
print(classification_report(y_test, y_pred_topic))

cm_topic = confusion_matrix(y_test, y_pred_topic)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_topic, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix – Topic Classification (SVM)")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("outputs/confusion_matrix_topic_svm.png")
plt.show()

# =========================================================
# 4. PRIORITY CLASSIFICATION – SVM
# =========================================================
y_priority = df["Priority_ID"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_priority, test_size=0.2, random_state=42
)

svm_priority = LinearSVC()
svm_priority.fit(X_train, y_train)

y_pred_priority = svm_priority.predict(X_test)

print("\n=== SVM : PRIORITY CLASSIFICATION ===")
print("Accuracy:", accuracy_score(y_test, y_pred_priority))
print(classification_report(y_test, y_pred_priority))

cm_priority = confusion_matrix(y_test, y_pred_priority)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_priority, annot=True, fmt="d", cmap="Greens")
plt.title("Confusion Matrix – Priority Classification (SVM)")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("outputs/confusion_matrix_priority_svm.png")
plt.show()

print("\n✅ SVM evaluation completed. Confusion matrix images saved in 'outputs/' folder.")
