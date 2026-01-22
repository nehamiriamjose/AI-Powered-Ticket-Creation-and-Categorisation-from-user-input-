import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False
    print("⚠️ XGBoost not installed — skipping XGBoost")

# ==================================================
# 1. LOAD PROCESSED DATA (FROM MILESTONE 1)
# ==================================================
df = pd.read_csv("data/processed/all_tickets_prepared_for_model.csv")

X = df["clean_document"]
y_topic = df["Topic_ID"]
y_priority = df["Priority_ID"]

print("Dataset loaded successfully")
print(df.columns)

# ==================================================
# 2. TRAIN–TEST SPLIT
# ==================================================
X_train, X_test, y_topic_train, y_topic_test = train_test_split(
    X, y_topic, test_size=0.2, random_state=42, stratify=y_topic
)

_, _, y_pr_train, y_pr_test = train_test_split(
    X, y_priority, test_size=0.2, random_state=42, stratify=y_priority
)

# ==================================================
# 3. TF-IDF FEATURE EXTRACTION
# ==================================================
tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("TF-IDF feature matrix shape:", X_train_tfidf.shape)

# ==================================================
# 4. MODEL DICTIONARY (ENGINEERED PART)
# ==================================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": LinearSVC(),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1
    )
}

if xgb_available:
    models["XGBoost"] = XGBClassifier(
        objective="multi:softmax",
        eval_metric="mlogloss"
    )

# ==================================================
# 5. TRAINING & EVALUATION FUNCTION
# ==================================================
def train_and_evaluate(X_train, y_train, X_test, y_test, task_name):
    results = {}
    best_model = None
    best_f1 = 0

    print("\n" + "=" * 70)
    print(f"MODEL COMPARISON — {task_name.upper()}")
    print("=" * 70)

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")
        results[name] = (acc, f1)

        print(f"{name:<20} | Accuracy: {acc:.4f} | F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_model = model

    print("\nBest Model:", best_model)
    return best_model, results

# ==================================================
# 6. TOPIC (INTENT) CLASSIFICATION
# ==================================================
topic_model, topic_results = train_and_evaluate(
    X_train_tfidf, y_topic_train,
    X_test_tfidf, y_topic_test,
    "Topic Classification"
)

topic_preds = topic_model.predict(X_test_tfidf)
topic_cm = confusion_matrix(y_topic_test, topic_preds)

print("\nTopic Classification Report:")
print(classification_report(y_topic_test, topic_preds))

# ==================================================
# 7. PRIORITY CLASSIFICATION
# ==================================================
priority_model, priority_results = train_and_evaluate(
    X_train_tfidf, y_pr_train,
    X_test_tfidf, y_pr_test,
    "Priority Classification"
)

priority_preds = priority_model.predict(X_test_tfidf)
priority_cm = confusion_matrix(y_pr_test, priority_preds)

print("\nPriority Classification Report:")
print(classification_report(y_pr_test, priority_preds))

# ==================================================
# 8. CONFUSION MATRIX VISUALIZATION
# ==================================================
plt.figure(figsize=(8, 6))
sns.heatmap(topic_cm, annot=True, fmt="d", cmap="Blues")
plt.title("Topic Classification — Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("topic_confusion_matrix.png")
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(priority_cm, annot=True, fmt="d", cmap="Greens")
plt.title("Priority Classification — Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("priority_confusion_matrix.png")
plt.show()

# ==================================================
# 9. SUMMARY TABLE (ENGINEERED OUTPUT)
# ==================================================
summary_rows = []

for model_name in topic_results:
    topic_acc, topic_f1 = topic_results[model_name]
    pr_acc, pr_f1 = priority_results.get(model_name, (None, None))

    summary_rows.append([
        model_name,
        topic_acc, topic_f1,
        pr_acc, pr_f1
    ])

summary_df = pd.DataFrame(
    summary_rows,
    columns=[
        "Model",
        "Topic_Accuracy", "Topic_F1",
        "Priority_Accuracy", "Priority_F1"
    ]
)

print("\nFINAL MODEL COMPARISON TABLE")
print(summary_df)

summary_df.to_csv("milestone2_model_comparison.csv", index=False)

print("\n✅ Milestone 2 — ENGINEERED PIPELINE COMPLETED SUCCESSFULLY")
