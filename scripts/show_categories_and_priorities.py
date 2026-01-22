import pandas as pd

# --------------------------------------------------
# 1. Load processed dataset
# --------------------------------------------------
df = pd.read_csv("data/processed/all_tickets_prepared_for_model.csv")

print("Dataset loaded successfully\n")

# --------------------------------------------------
# 2. Extract Category (Topic) mapping
# --------------------------------------------------
print("===== CATEGORY (TOPIC) MAPPING =====")

topic_mapping = (
    df[["Topic_ID", "Topic_group"]]
    .drop_duplicates()
    .sort_values("Topic_ID")
)

for _, row in topic_mapping.iterrows():
    print(f"Topic_ID {row['Topic_ID']} → {row['Topic_group']}")

# --------------------------------------------------
# 3. Extract Priority mapping
# --------------------------------------------------
print("\n===== PRIORITY MAPPING =====")

priority_mapping = (
    df[["Priority_ID", "Priority"]]
    .drop_duplicates()
    .sort_values("Priority_ID")
)

for _, row in priority_mapping.iterrows():
    print(f"Priority_ID {row['Priority_ID']} → {row['Priority']}")

print("\n✅ Category and Priority mappings extracted from dataset")
