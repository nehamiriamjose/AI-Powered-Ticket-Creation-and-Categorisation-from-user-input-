import spacy
import os
import sys

# ============================================================
#                 LOAD SPACY MODEL WITH ERROR HANDLING
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "spacy_ner_model")

nlp = None
model_loaded = False

try:
    if os.path.exists(MODEL_PATH):
        nlp = spacy.load(MODEL_PATH)
        model_loaded = True
    else:
        print(f"⚠️  Warning: spaCy NER model not found at {MODEL_PATH}")
except Exception as e:
    print(f"⚠️  Warning: Failed to load spaCy model: {e}")

# ============================================================
#                 ENTITY EXTRACTION FUNCTION
# ============================================================
def extract_entities(text: str):
    """Extract entities using spaCy ML-based NER model"""
    entities = {
        "users": [],
        "devices": [],
        "error_codes": []
    }

    # If model is not loaded, return empty entities
    if not model_loaded or nlp is None:
        return entities

    try:
        doc = nlp(text)

        for ent in doc.ents:
            if ent.label_ == "USER":
                entities["users"].append(ent.text)
            elif ent.label_ == "DEVICE":
                entities["devices"].append(ent.text)
            elif ent.label_ == "ERROR_CODE":
                entities["error_codes"].append(ent.text)
    except Exception as e:
        print(f"⚠️  Error during entity extraction: {e}")

    return entities
