import pandas as pd
from pathlib import Path

source = Path("data/multilingual/hindi_review.csv")

unique_output = Path(
    "data/multilingual/hindi_unique_review.csv"
)

conflict_output = Path(
    "data/multilingual/hindi_label_conflicts.csv"
)

df = pd.read_csv(source, encoding="utf-8-sig")

df["message"] = df["message"].fillna("").astype(str).str.strip()
df["label"] = df["label"].fillna("").astype(str).str.lower().str.strip()

# Separate invalid labels
valid_mask = df["label"].isin(["ham", "spam"])
invalid = df[~valid_mask].copy()
df = df[valid_mask].copy()

# Find messages that have conflicting labels
conflict_mask = (
    df.groupby("message")["label"]
    .transform("nunique") > 1
)

conflicts = df[conflict_mask].copy()

# Keep only messages with consistent labels
safe = df[~conflict_mask].copy()

# Remove exact duplicate Hindi messages
unique = safe.drop_duplicates(
    subset=["message"],
    keep="first"
).copy()

# Keep all messages unverified
unique["verified"] = "NO"

unique.to_csv(
    unique_output,
    index=False,
    encoding="utf-8-sig"
)

conflicts.to_csv(
    conflict_output,
    index=False,
    encoding="utf-8-sig"
)

if not invalid.empty:
    invalid.to_csv(
        "data/multilingual/hindi_invalid_labels.csv",
        index=False,
        encoding="utf-8-sig"
    )

print("Hindi dataset preparation completed.")
print("Original rows:", len(df) + len(invalid))
print("Unique candidate messages:", len(unique))
print("Conflicting message groups:", conflicts["message"].nunique())
print("Rows held for label conflicts:", len(conflicts))
print("Invalid label rows:", len(invalid))
print("\nHAM/SPAM distribution:")
print(unique["label"].value_counts())
print("\nSaved:", unique_output)
print("Conflict file:", conflict_output)