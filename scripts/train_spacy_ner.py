import spacy
from spacy.training.example import Example
from pathlib import Path
import os
import sys

# ============================================================
#                 SETUP PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(MODELS_DIR, "spacy_ner_model")

# Create models directory if it doesn't exist
Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)

print(f"📁 Models directory: {MODELS_DIR}")
print(f"📁 Output directory: {OUTPUT_DIR}")

# ============================================================
#                 LOAD BASE MODEL
# ============================================================
print("\n📥 Loading base English model (en_core_web_sm)...")
nlp = spacy.load("en_core_web_sm")

# Get NER pipe
ner = nlp.get_pipe("ner")

# Custom entity labels
LABELS = ["USER", "DEVICE", "ERROR_CODE"]

for label in LABELS:
    ner.add_label(label)

print(f"✅ Added entity labels: {LABELS}")

# -------------------------------------------------
# TRAINING DATA (ML learns patterns, not rules)
# -------------------------------------------------
TRAIN_DATA = [
    ("User john_doe cannot login to SERVER_12 due to ERR_403",
     {"entities": [(5, 13, "USER"), (31, 40, "DEVICE"), (48, 55, "ERROR_CODE")]}),

    ("alice failed to access PC-101 with error 0x80070005",
     {"entities": [(0, 5, "USER"), (24, 30, "DEVICE"), (42, 52, "ERROR_CODE")]}),

    ("bob cannot connect to LAPTOP_45",
     {"entities": [(0, 3, "USER"), (23, 32, "DEVICE")]}),

    ("mark gets ERR_401 while accessing SERVER-99",
     {"entities": [(0, 4, "USER"), (30, 39, "DEVICE"), (11, 18, "ERROR_CODE")]}),
]

print(f"\n🎯 Training with {len(TRAIN_DATA)} examples...")

# -------------------------------------------------
# TRAIN MODEL (Using modern spaCy API)
# -------------------------------------------------
from spacy.util import get_lang_class
from spacy.training import Corpuses

# Initialize the training with modern API
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
with nlp.disable_pipes(*other_pipes):
    # Use n_process=1 for single-threaded training
    optimizer = nlp.create_pipe("tok2vec").begin_training()
    
    for epoch in range(40):
        losses = {}
        for text, annotations in TRAIN_DATA:
            try:
                example = Example.from_dict(nlp.make_doc(text), annotations)
                nlp.update([example], drop=0.5, sgd=optimizer, losses=losses)
            except Exception as e:
                print(f"⚠️  Error training on '{text}': {e}")
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch + 1}/40 - Loss: {losses.get('ner', 0):.4f}")

# -------------------------------------------------
# SAVE MODEL
# -------------------------------------------------
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
nlp.to_disk(OUTPUT_DIR)

print(f"\n✅ spaCy ML-based NER model trained and saved to {OUTPUT_DIR}")
