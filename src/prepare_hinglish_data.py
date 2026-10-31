
import pandas as pd
from pathlib import Path

source = Path("data/multilingual/hinglish_candidates_review.csv")
output = Path("data/multilingual/hinglish_reviewed.csv")

df = pd.read_csv(source)

# Remove empty messages and exact duplicate messages.
df = df.dropna(subset=["message"])
df["message"] = df["message"].astype(str).str.strip()
df = df[df["message"] != ""]
df = df.drop_duplicates(subset=["message"])

# Preserve original labels for manual verification.
df["verified"] = "NO"
df["language"] = "HINGLISH_OR_MIXED"

df.to_csv(output, index=False, encoding="utf-8-sig")

print("Hinglish review file created successfully.")
print("Total candidate messages:", len(df))
print("Output:", output)
print("\nColumns:", df.columns.tolist())
print("\nLabel distribution:")
print(df["label"].value_counts())
