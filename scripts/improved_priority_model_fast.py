"""
IMPROVED PRIORITY MODEL - FAST VERSION
========================================
Uses same optimization as category model:
- TF-IDF with 5000 features
- SMOTE balancing
- Pre-tuned Logistic Regression
- Runtime: ~1 minute
"""

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
import time

start_time = time.time()

print("=" * 70)
print("IMPROVED PRIORITY MODEL (Fast Version)")
print("=" * 70)

# Load data
print("\n📥 Loading dataset...")
df = pd.read_csv("data/processed/all_tickets_prepared_for_model.csv")
X = df["clean_document"]
y_priority = df["Priority_ID"]

print(f"   Dataset: {len(df)} tickets")
print(f"   Priority distribution:\n{y_priority.value_counts().sort_index()}")

# ================================================================
# STEP 1: ENHANCED TF-IDF (matching category model)
# ================================================================
print("\n🔄 Step 1: TF-IDF Vectorization...")
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.9,
    sublinear_tf=True,
    stop_words='english'
)
X_tfidf = tfidf.fit_transform(X)
print(f"   ✓ TF-IDF shape: {X_tfidf.shape}")

# ================================================================
# STEP 2: SMOTE CLASS BALANCING
# ================================================================
print("\n⚖️  Step 2: SMOTE Class Balancing...")
smote = SMOTE(random_state=42, k_neighbors=5)
X_balanced, y_balanced = smote.fit_resample(X_tfidf, y_priority)
print(f"   ✓ Balanced samples: {len(y_balanced)}")
print(f"   Distribution:\n{pd.Series(y_balanced).value_counts().sort_index()}")

# ================================================================
# STEP 3: TRAIN WITH OPTIMIZED PARAMETERS
# ================================================================
print("\n🚀 Step 3: Training Logistic Regression...")
print("   Parameters: C=10, solver='saga', max_iter=500")

priority_model = LogisticRegression(
    C=10,
    solver='saga',
    max_iter=500,
    random_state=42,
    class_weight=None,
    n_jobs=-1
)

priority_model.fit(X_balanced, y_balanced)
print("   ✓ Model trained")

# ================================================================
# STEP 4: VALIDATION
# ================================================================
print("\n📊 Step 4: Validation...")

cv_scores = cross_val_score(priority_model, X_balanced, y_balanced, cv=3, scoring='accuracy', n_jobs=-1)
print(f"   ✓ 3-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Predictions
y_pred = priority_model.predict(X_balanced)
accuracy = accuracy_score(y_balanced, y_pred)
f1 = f1_score(y_balanced, y_pred, average='weighted')

print(f"\n🎯 Model Performance:")
print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   F1-Score: {f1:.4f}")

# Classification report
print("\n" + "=" * 70)
print("CLASSIFICATION REPORT BY PRIORITY")
print("=" * 70)
priority_labels = {0: "High", 1: "Low", 2: "Medium"}
print(classification_report(y_balanced, y_pred, 
      target_names=[priority_labels[i] for i in sorted(y_balanced.unique())]))

# ================================================================
# STEP 5: SAVE MODELS
# ================================================================
print("\n💾 Saving improved models...")
joblib.dump(priority_model, "models/final_priority_model.pkl")
joblib.dump(tfidf, "models/final_tfidf_vectorizer.pkl")
print("   ✓ Models saved:")
print("     - models/final_priority_model.pkl")
print("     - models/final_tfidf_vectorizer.pkl")

# ================================================================
# SUMMARY
# ================================================================
elapsed = time.time() - start_time
print("\n" + "=" * 70)
print("✅ PRIORITY MODEL IMPROVEMENT COMPLETE")
print("=" * 70)
print(f"Baseline Priority Accuracy:  68.16%")
print(f"Improved Priority Accuracy:  {accuracy*100:.2f}%")
print(f"Improvement:                 {(accuracy*100 - 68.16):.2f}% ⬆️")
print(f"Training Time:               {elapsed:.1f} seconds")
print("=" * 70)
