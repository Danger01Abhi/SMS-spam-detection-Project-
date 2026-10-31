import pandas as pd
from pathlib import Path

source = Path("data/multilingual/hindi_review.csv")
output = Path("data/multilingual/hindi_duplicate_audit.csv")

df = pd.read_csv(source, encoding="utf-8-sig")

# Group identical Hindi messages
grouped = (
    df.groupby("message", dropna=False)
    .agg(
        occurrences=("message", "size"),
        labels=(
            "label",
            lambda x: ", ".join(sorted(set(x.astype(str).str.lower())))
        ),
        original_examples=(
            "original_english",
            lambda x: " || ".join(x.astype(str).head(3))
        )
    )
    .reset_index()
)

# Keep only repeated messages
duplicates = grouped[grouped["occurrences"] > 1].copy()

# Identify messages associated with both HAM and SPAM labels
duplicates["label_conflict"] = (
    duplicates["labels"].str.contains(",")
)

duplicates = duplicates.sort_values(
    "occurrences",
    ascending=False
)

duplicates.to_csv(output, index=False, encoding="utf-8-sig")

print("Duplicate audit completed.")
print("Repeated message groups:", len(duplicates))
print(
    "Rows involved in repeated groups:",
    int(duplicates["occurrences"].sum())
)
print(
    "Groups with conflicting labels:",
    int(duplicates["label_conflict"].sum())
)

print("\nFirst 15 repeated messages:")
print(
    duplicates[
        ["message", "occurrences", "labels", "label_conflict"]
    ].head(15).to_string(index=False, max_colwidth=80)
)

print("\nSaved report:", output)