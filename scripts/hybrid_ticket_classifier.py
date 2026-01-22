import re
import joblib

# ==================================================
# 1. LOAD MODELS & TF-IDF
# ==================================================
topic_model = joblib.load("models/final_topic_model.pkl")
priority_model = joblib.load("models/final_priority_model.pkl")
tfidf = joblib.load("models/final_tfidf_vectorizer.pkl")

print("✅ Models & TF-IDF loaded")

# ==================================================
# 2. LABEL MAPS (FROM DATASET)
# ==================================================
TOPIC_MAP = {
    0: "Access",
    1: "Administrative Rights",
    2: "HR Support",
    3: "Hardware",
    4: "Internal Project",
    5: "Miscellaneous",
    6: "Purchase",
    7: "Storage"
}

PRIORITY_MAP = {
    0: "High",
    1: "Low",
    2: "Medium"
}

# ==================================================
# 3. SAFE RULE-BASED ENTITY EXTRACTION
# ==================================================
def extract_entities(text):
    machines = re.findall(r'\b(?:PC|SERVER|LAPTOP)[-_]?\d+\b', text, re.I)
    error_codes = re.findall(r'\b(ERR[_-]?\d+|0x[a-fA-F0-9]+)\b', text)
    return machines, error_codes

# ==================================================
# 4. SAFE HYBRID LOGIC (PRIORITY ONLY)
# ==================================================
def refine_priority_safe(text, ml_priority):
    """
    Upgrade priority ONLY when strong technical evidence exists
    """
    text_lower = text.lower()
    machines, error_codes = extract_entities(text)

    # Strong phrases (very limited)
    strong_phrases = [
        "server down",
        "system down",
        "production down",
        "service unavailable"
    ]

    # Only upgrade, never downgrade
    if ml_priority != "High":
        if (
            (machines and error_codes) or
            any(phrase in text_lower for phrase in strong_phrases)
        ):
            return "High"

    return ml_priority

# ==================================================
# 5. FINAL PREDICTION FUNCTION
# ==================================================
def predict_ticket(text):
    text_tfidf = tfidf.transform([text])

    topic_id = topic_model.predict(text_tfidf)[0]
    priority_id = priority_model.predict(text_tfidf)[0]

    ml_topic = TOPIC_MAP.get(topic_id, "Unknown")
    ml_priority = PRIORITY_MAP.get(priority_id, "Unknown")

    final_priority = refine_priority_safe(text, ml_priority)

    return {
        "User_Input": text,
        "Predicted_Category": ml_topic,
        "Predicted_Priority": final_priority
    }

# ==================================================
# 6. REAL-TIME TEST
# ==================================================
if __name__ == "__main__":
    print("\n===== SAFE HYBRID AI TICKET SYSTEM =====")
    user_text = input("Enter ticket issue: ")

    result = predict_ticket(user_text)

    print("\n========== FINAL OUTPUT ==========")
    print(f"User Input         : {result['User_Input']}")
    print(f"Predicted Category : {result['Predicted_Category']}")
    print(f"Predicted Priority : {result['Predicted_Priority']}")
    print("=================================")
