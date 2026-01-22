import pandas as pd
import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize

# download required NLTK data
nltk.download('stopwords', quiet=True)

# load raw dataset
df = pd.read_csv("data/raw/raw_it_support_tickets.csv")

print("Columns in dataset:", df.columns)

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = wordpunct_tokenize(text)           # ✅ safer tokenizer
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)

# apply preprocessing
df['clean_document'] = df['Document'].apply(preprocess_text)

# preview
print(df[['Document', 'clean_document']].head())

# ensure output folder exists
os.makedirs("data/processed", exist_ok=True)

# save processed dataset
df.to_csv("data/processed/final_preprocessed_dataset.csv", index=False)

print("✅ Text preprocessing completed successfully")
