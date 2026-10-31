import pandas as pd
from pathlib import Path

source = Path("data/multilingual/hindi_unique_review.csv")
output = Path("data/multilingual/hindi_manual_review_100.csv")

df = pd.read_csv(source, encoding="utf-8-sig")

# Select 50 HAM and 50 SPAM messages
ham = df[df["label"] == "ham"].sample(
    n=50, random_state=42
)

spam = df[df["label"] == "spam"].sample(
    n=50, random_state=42
)

sample = pd.concat([ham, spam]).sample(
    frac=1, random_state=42
).reset_index(drop=True)

sample["translation_quality"] = ""
sample["corrected_hindi"] = ""
sample["review_notes"] = ""

sample.to_csv(output, index=False, encoding="utf-8-sig")

print("Manual review sample created.")
print("Total messages:", len(sample))
print(sample["label"].value_counts())
print("Saved:", output)