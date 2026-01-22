import pandas as pd

df = pd.read_csv("data/processed/all_tickets_prepared_for_model.csv")

# Merge Admin Rights → Access
df.loc[df["Topic_group"] == "Administrative Rights", "Topic_group"] = "Access"

# Re-encode Topic_ID
df["Topic_ID"] = df["Topic_group"].astype("category").cat.codes

df.to_csv("data/processed/all_tickets_prepared_for_model.csv", index=False)

print("✅ Labels merged: Administrative Rights → Access")
