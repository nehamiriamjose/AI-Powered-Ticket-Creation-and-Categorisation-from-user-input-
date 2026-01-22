import joblib
import pandas as pd

# ==================================================
# 1. LOAD SAVED FINAL MODELS & VECTORIZER
# ==================================================

topic_model = joblib.load("models/final_topic_model.pkl")
priority_model = joblib.load("models/final_priority_model.pkl")
tfidf = joblib.load("models/final_tfidf_vectorizer.pkl")

print("✅ Models and TF-IDF Vectorizer loaded successfully")

# ==================================================
# 2. LOAD DATASET & AUTO-EXTRACT LABEL MAPPINGS
# ==================================================

df = pd.read_csv("data/processed/all_tickets_prepared_for_model.csv")

# Automatically build mappings from dataset
TOPIC_MAP = (
    df[["Topic_ID", "Topic_group"]]
    .drop_duplicates()
    .sort_values("Topic_ID")
    .set_index("Topic_ID")["Topic_group"]
    .to_dict()
)

PRIORITY_MAP = (
    df[["Priority_ID", "Priority"]]
    .drop_duplicates()
    .sort_values("Priority_ID")
    .set_index("Priority_ID")["Priority"]
    .to_dict()
)

print("✅ Category and Priority mappings loaded from dataset")

# ==================================================
# 3. PREDICTION FUNCTION
# ==================================================

def predict_ticket(user_text):
    """
    Predicts category and priority using final models
    and dataset-derived label mappings
    """

    # Transform input text using saved TF-IDF
    text_tfidf = tfidf.transform([user_text])

    # Predict encoded IDs
    topic_id = int(topic_model.predict(text_tfidf)[0])
    priority_id = int(priority_model.predict(text_tfidf)[0])

    # Decode using dataset mappings
    topic_name = TOPIC_MAP.get(topic_id, "Unknown Category")
    priority_name = PRIORITY_MAP.get(priority_id, "Unknown Priority")

    return {
        "User_Input": user_text,
        "Predicted_Topic_ID": topic_id,
        "Predicted_Category": topic_name,
        "Predicted_Priority_ID": priority_id,
        "Predicted_Priority": priority_name
    }

# ==================================================
# 4. REAL-TIME TESTING
# ==================================================

if __name__ == "__main__":
    print("\n===== AI TICKET CLASSIFICATION TEST =====")
    user_input = input("Enter user issue description: ")

    result = predict_ticket(user_input)

    print("\n========== PREDICTION RESULT ==========")
    print(f"User Input           : {result['User_Input']}")
    print(f"Category ID          : {result['Predicted_Topic_ID']}")
    print(f"Predicted Category   : {result['Predicted_Category']}")
    print(f"Priority ID          : {result['Predicted_Priority_ID']}")
    print(f"Predicted Priority   : {result['Predicted_Priority']}")
    print("=======================================")
