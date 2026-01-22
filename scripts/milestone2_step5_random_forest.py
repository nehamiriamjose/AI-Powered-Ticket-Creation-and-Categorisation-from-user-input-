import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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
# 3. TOPIC (CATEGORY) CLASSIFICATION – RANDOM FOREST
# =========================================================
y_topic = df["Topic_ID"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_topic, test_size=0.2, random_state=42
)

rf_topic = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
rf_topic.fit(X_train, y_train)

y_pred_topic = rf_topic.predict(X_test)

print("\n=== Random Forest : TOPIC CLASSIFICATION ===")
print("Accuracy:", accuracy_score(y_test, y_pred_topic))
print(classification_report(y_test, y_pred_topic))

cm_topic = confusion_matrix(y_test, y_pred_topic)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_topic, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix – Topic Classification (Random Forest)")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("outputs/confusion_matrix_topic_random_forest.png")
plt.show()

# =========================================================
# 4. PRIORITY CLASSIFICATION – RANDOM FOREST
# =========================================================
y_priority = df["Priority_ID"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y_priority, test_size=0.2, random_state=42
)

rf_priority = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
rf_priority.fit(X_train, y_train)

y_pred_priority = rf_priority.predict(X_test)

print("\n=== Random Forest : PRIORITY CLASSIFICATION ===")
print("Accuracy:", accuracy_score(y_test, y_pred_priority))
print(classification_report(y_test, y_pred_priority))

cm_priority = confusion_matrix(y_test, y_pred_priority)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_priority, annot=True, fmt="d", cmap="Greens")
plt.title("Confusion Matrix – Priority Classification (Random Forest)")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("outputs/confusion_matrix_priority_random_forest.png")
plt.show()

print("\n✅ Random Forest evaluation completed. Confusion matrix images saved in 'outputs/' folder.")
