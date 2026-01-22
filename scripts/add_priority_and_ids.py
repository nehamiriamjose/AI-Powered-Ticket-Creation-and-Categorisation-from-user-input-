
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# load processed dataset
df = pd.read_csv("data/processed/final_preprocessed_dataset.csv")

# -----------------------------
# STEP 1: RULE-BASED PRIORITY
# -----------------------------

def assign_priority(text):
    text = str(text).lower()

    if any(word in text for word in ['down', 'outage', 'crash', 'urgent', 'error', 'failed']):
        return 'High'
    elif any(word in text for word in ['slow', 'delay', 'issue', 'problem']):
        return 'Medium'
    elif any(word in text for word in ['request', 'query', 'information', 'ask']):
        return 'Low'
    else:
        return 'Medium'   # default

# apply priority logic on cleaned text
df['Priority'] = df['clean_document'].apply(assign_priority)

# -----------------------------
# STEP 2: LABEL ENCODING
# -----------------------------

# encode Topic_group
topic_encoder = LabelEncoder()
df['Topic_ID'] = topic_encoder.fit_transform(df['Topic_group'])

# encode Priority
priority_encoder = LabelEncoder()
df['Priority_ID'] = priority_encoder.fit_transform(df['Priority'])

# preview
print(df[['Topic_group', 'Topic_ID', 'Priority', 'Priority_ID']].head())

# save updated dataset
df.to_csv("data/processed/all_tickets_prepared_for_model.csv", index=False)

print("✅ Priority, Topic_ID, and Priority_ID added successfully")
