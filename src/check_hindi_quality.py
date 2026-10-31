import pandas as pd
from pathlib import Path

source = Path("data/multilingual/hindi_review.csv")
output = Path("data/multilingual/hindi_quality_report.csv")

df = pd.read_csv(source, encoding="utf-8-sig")

messages = df["message"].fillna("").astype(str).str.strip()

# Quality checks
df["has_hindi_script"] = messages.str.contains(
    r"[\u0900-\u097F]", regex=True
)

df["is_empty"] = messages.eq("")

df["is_duplicate"] = messages.duplicated(keep=False)

df["is_very_short"] = messages.str.len() < 10

df["english_word_count"] = messages.str.count(
    r"\b[a-zA-Z]{2,}\b"
)

df["needs_review"] = (
    ~df["has_hindi_script"]
    | df["is_empty"]
    | df["is_duplicate"]
    | df["is_very_short"]
    | (df["english_word_count"] > 8)
)

# Save report separately
df.to_csv(output, index=False, encoding="utf-8-sig")

print("Hindi quality report created.")
print("Total messages:", len(df))
print("Without Hindi script:", int((~df["has_hindi_script"]).sum()))
print("Empty messages:", int(df["is_empty"].sum()))
print("Duplicate messages:", int(df["is_duplicate"].sum()))
print("Very short messages:", int(df["is_very_short"].sum()))
print("Messages needing review:", int(df["needs_review"].sum()))
print("Output:", output)