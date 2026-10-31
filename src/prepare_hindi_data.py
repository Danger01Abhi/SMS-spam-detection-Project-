import pandas as pd
from pathlib import Path

source = Path("data/multilingual/multilingual_sms_source.csv")
output = Path("data/multilingual/hindi_review.csv")

df = pd.read_csv(source, encoding="utf-8-sig")

# Keep only required columns
hindi_df = df[["labels", "text", "text_hi"]].copy()

# Rename columns
hindi_df = hindi_df.rename(columns={
    "labels": "label",
    "text": "original_english",
    "text_hi": "message"
})

# Remove empty Hindi messages
hindi_df["message"] = hindi_df["message"].fillna("").astype(str).str.strip()
hindi_df = hindi_df[hindi_df["message"] != ""]

# Mark every message as unverified
hindi_df["verified"] = "NO"
hindi_df["language"] = "HINDI_TRANSLATED"

# Save separately
hindi_df.to_csv(output, index=False, encoding="utf-8-sig")

print("Hindi review dataset created.")
print("Total messages:", len(hindi_df))
print("Output:", output)
print("\nLabel distribution:")
print(hindi_df["label"].value_counts())