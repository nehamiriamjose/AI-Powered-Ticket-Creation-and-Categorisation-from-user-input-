import uuid
from datetime import datetime
import re
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.milestone3_model_inference import predict_category_priority

# Try to import spacy, fall back to None if not available
try:
    from scripts.spacy_ner import extract_entities as extract_entities_spacy
    spacy_available = True
except Exception as e:
    print(f"⚠️  spaCy not available: {e}")
    spacy_available = False
    extract_entities_spacy = None

# ==================================================
# CATEGORY & PRIORITY MAPS (FROM DATASET)
# ==================================================
CATEGORY_MAP = {
    0: "Access",
    1: "Administrative rights",
    2: "HR Support",
    3: "Hardware",
    4: "Internal project",
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
# INPUT VALIDATION (SMALL INPUT HANDLING)
# ==================================================
def validate_input(text: str, min_words: int = 3):
    if not text or not text.strip():
        raise ValueError("Ticket description cannot be empty")

    if len(text.split()) < min_words:
        raise ValueError(
            "Please provide more details about the issue (at least 3 words)."
        )

# ==================================================
# TEXT PREPROCESSING
# ==================================================
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ==================================================
# ENTITY EXTRACTION (SPACY-BASED NER)
# ==================================================
def extract_entities(text: str):
    """
    Extract entities using spaCy ML-based NER model.
    Falls back to regex if spaCy model is unavailable.
    """
    entities = {
        "users": [],
        "devices": [],
        "error_codes": []
    }
    
    # Try spaCy first if available
    if spacy_available and extract_entities_spacy is not None:
        try:
            entities = extract_entities_spacy(text)
            # If spaCy found something, return it
            if any(entities.values()):
                return entities
        except Exception as e:
            print(f"⚠️  spaCy extraction failed: {e}")
    
    # Use regex as fallback
    usernames = re.findall(r"\buser_\w+\b|\b[a-z]+_[a-z]+\b", text, re.IGNORECASE)
    devices = re.findall(r"\b(laptop|server|pc|desktop|workstation|device|SERVER_[A-Z0-9]+)\b", text, re.IGNORECASE)
    error_codes = re.findall(r"\bERR[_-]?\d+\b|\b0x[0-9A-F]+\b", text.upper())
    
    entities = {
        "users": list(set(usernames)),
        "devices": list(set(devices)),
        "error_codes": list(set(error_codes))
    }
    
    return entities

# ==================================================
# PRIORITY BUSINESS LOGIC (SAFE ESCALATION)
# ==================================================
def refine_priority(text: str, ml_priority: str) -> str:
    critical_keywords = [
        "server down",
        "system down",
        "service unavailable",
        "production down"
    ]

    text_lower = text.lower()

    if ml_priority != "High":
        for keyword in critical_keywords:
            if keyword in text_lower:
                return "High"

    return ml_priority

# ==================================================
# MAIN EXECUTION
# ==================================================
if __name__ == "__main__":
    print("\n=== AI Ticket Generation Engine ===")

    user_input = input("Enter ticket description: ")

    # Step 1: Validate input
    validate_input(user_input)

    # Step 2: Preprocess
    clean_text = preprocess_text(user_input)

    # Step 3: ML Prediction
    category_id, priority_id, category_confidence, priority_confidence = predict_category_priority(clean_text)

    # Step 4: Map IDs to labels
    category = CATEGORY_MAP.get(category_id, "Unknown")
    ml_priority = PRIORITY_MAP.get(priority_id, "Unknown")

    # Step 5: Apply business logic
    priority = refine_priority(user_input, ml_priority)

    # Step 6: Entity extraction
    entities = extract_entities(clean_text)

    # Step 7: Generate ticket
    ticket = {
        "ticket_id": f"INC-{uuid.uuid4().hex[:6].upper()}",
        "title": f"{category} Issue",
        "description": user_input,
        "cleaned_description": clean_text,
        "category": category,
        "category_confidence": round(category_confidence, 4),
        "priority": priority,
        "priority_confidence": round(priority_confidence, 4),
        "entities": {
            "users": entities.get("users", []),
            "devices": entities.get("devices", []),
            "error_codes": entities.get("error_codes", [])
        },
        "status": "Open",
        "created_at": datetime.now().isoformat()
    }

    # Step 8: Output
    print("\n--- GENERATED TICKET (JSON) ---\n")
    for key, value in ticket.items():
        print(f"{key}: {value}")
