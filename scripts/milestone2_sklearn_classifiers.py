import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("data/processed/all_tickets_prepared_for_model.csv")

print("Dataset loaded successfully")
print(df.columns)

# -----------------------------
# 2. Feature Extraction (TF-IDF)
# -----------------------------
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=3
)

X = tfidf.fit_transform(df["clean_document"])

# Targets
y_topic = df["Topic_ID"]
y_priority = df["Priority_ID"]

# -----------------------------
# 3. Train-Test Split
# -----------------------------
X_train, X_test, y_topic_train, y_topic_test = train_test_split(
    X, y_topic, test_size=0.2, random_state=42, stratify=y_topic
)

_, _, y_priority_train, y_priority_test = train_test_split(
    X, y_priority, test_size=0.2, random_state=42, stratify=y_priority
)

# -----------------------------
# 4. Define Models
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000, class_weight="balanced"
    ),
    "SVM (Linear)": LinearSVC(
        class_weight="balanced"
    ),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced"
    )
}

# -----------------------------
# 5. Training & Evaluation
# -----------------------------
def evaluate_model(model, X_train, X_test, y_train, y_test, title):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n{title} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {title}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

# -----------------------------
# 6. Run All Models
# -----------------------------
for name, model in models.items():
    print("\n" + "="*60)
    print(f"MODEL: {name} (TOPIC CLASSIFICATION)")
    evaluate_model(
        model,
        X_train,
        X_test,
        y_topic_train,
        y_topic_test,
        f"{name} - Topic"
    )

    print("\n" + "="*60)
    print(f"MODEL: {name} (PRIORITY CLASSIFICATION)")
    evaluate_model(
        model,
        X_train,
        X_test,
        y_priority_train,
        y_priority_test,
        f"{name} - Priority"
    )

print("\n✅ Scikit-learn classification completed successfully")
