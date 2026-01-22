import os
import joblib
import numpy as np

# ============================================================
#                 FIX MODEL PATHS (CRITICAL)
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

TOPIC_MODEL_PATH = os.path.join(MODELS_DIR, "final_topic_model.pkl")
PRIORITY_MODEL_PATH = os.path.join(MODELS_DIR, "final_priority_model.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "final_tfidf_vectorizer.pkl")

# ============================================================
#                   LOAD MODELS SAFELY
# ============================================================
topic_model = None
priority_model = None
vectorizer = None
models_loaded = False

try:
    if os.path.exists(TOPIC_MODEL_PATH) and os.path.exists(PRIORITY_MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        topic_model = joblib.load(TOPIC_MODEL_PATH)
        priority_model = joblib.load(PRIORITY_MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        models_loaded = True
    else:
        print(f"⚠️  Warning: Model files not found at {MODELS_DIR}")
except Exception as e:
    print(f"⚠️  Warning: Model loading failed: {e}")


# ============================================================
#                 PREDICTION FUNCTION
# ============================================================
def predict_category_priority(text: str):
    # If models are not loaded, return dummy values
    if not models_loaded or vectorizer is None or topic_model is None or priority_model is None:
        # Default: return category 0 (first category) and priority 0 (lowest priority)
        return 0, 0, 0.0, 0.0

    X = vectorizer.transform([text])

    category_id = int(topic_model.predict(X)[0])
    priority_id = int(priority_model.predict(X)[0])
    
    # Get confidence scores
    category_probs = topic_model.predict_proba(X)[0]
    priority_probs = priority_model.predict_proba(X)[0]
    
    category_confidence = float(category_probs[category_id])
    priority_confidence = float(priority_probs[priority_id])

    return category_id, priority_id, category_confidence, priority_confidence
