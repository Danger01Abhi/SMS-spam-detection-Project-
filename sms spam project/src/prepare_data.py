
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# 1. Set up project paths
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Find the original SMS dataset
# --------------------------------------------------
matches = [
    path for path in DATA_DIR.rglob("*")
    if path.is_file()
    and path.name.lower().startswith("smsspamcollection")
    and path.suffix.lower() not in {".zip", ".gz"}
    and "processed" not in path.parts
]

if not matches:
    raise FileNotFoundError(
        "Dataset not found. Check that SMSSpamCollection "
        "has been extracted into the data folder."
    )

DATA_PATH = next(
    (
        path for path in matches
        if path.name.lower() == "smsspamcollection"
    ),
    matches[0]
)

print("Dataset found:", DATA_PATH)


# --------------------------------------------------
# 3. Load and validate the data
# --------------------------------------------------
df = pd.read_csv(
    DATA_PATH,
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="latin-1"
)

original_count = len(df)

df = df.dropna(subset=["label", "message"]).copy()

df["label"] = df["label"].astype(str).str.strip().str.lower()
df["message"] = df["message"].astype(str)

# Keep only recognized labels
df = df[df["label"].isin(["ham", "spam"])].copy()

# Remove empty messages
df = df[df["message"].str.strip().ne("")].copy()


# --------------------------------------------------
# 4. Create a normalized text key for deduplication
# --------------------------------------------------
# This key is used only for data splitting and duplicate
# checks. We preserve the original message text.
df["_normalized_message"] = (
    df["message"]
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

# Remove messages that become empty after normalization
df = df[df["_normalized_message"].ne("")].copy()


# --------------------------------------------------
# 5. Handle messages with conflicting labels
# --------------------------------------------------
# The same normalized message should not be both ham and spam.
label_counts = df.groupby("_normalized_message")["label"].nunique()

conflicting_texts = set(
    label_counts[label_counts > 1].index
)

conflicting_rows = df["_normalized_message"].isin(
    conflicting_texts
).sum()

if conflicting_rows:
    print(
        "Rows excluded due to conflicting labels:",
        conflicting_rows
    )

    df = df[
        ~df["_normalized_message"].isin(conflicting_texts)
    ].copy()


# --------------------------------------------------
# 6. Remove repeated messages
# --------------------------------------------------
before_deduplication = len(df)

df = df.drop_duplicates(
    subset=["label", "_normalized_message"]
).copy()

duplicates_removed = before_deduplication - len(df)

print("Original rows:", original_count)
print("Duplicate messages removed:", duplicates_removed)
print("Rows available for splitting:", len(df))


# --------------------------------------------------
# 7. Split into training and testing datasets
# --------------------------------------------------
train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

# Check that normalized messages do not overlap
train_messages = set(train_df["_normalized_message"])
test_messages = set(test_df["_normalized_message"])

overlap = train_messages.intersection(test_messages)

if overlap:
    raise ValueError(
        "Data leakage detected: some messages appear "
        "in both training and testing sets."
    )


# --------------------------------------------------
# 8. Save the two datasets
# --------------------------------------------------
train_path = PROCESSED_DIR / "train.csv"
test_path = PROCESSED_DIR / "test.csv"

train_df[["label", "message"]].to_csv(
    train_path,
    index=False,
    encoding="utf-8"
)

test_df[["label", "message"]].to_csv(
    test_path,
    index=False,
    encoding="utf-8"
)


# --------------------------------------------------
# 9. Display class distributions
# --------------------------------------------------
def show_distribution(name, data):
    print(f"\n{name}")
    print("-" * 40)

    counts = data["label"].value_counts()
    percentages = (
        data["label"].value_counts(normalize=True) * 100
    )

    for label in ["ham", "spam"]:
        count = int(counts.get(label, 0))
        percentage = float(percentages.get(label, 0))

        print(
            f"{label.upper():<6}: {count:>5} "
            f"({percentage:.2f}%)"
        )

    print("Total:", len(data))


print("\n" + "=" * 45)
print("DATA PREPARATION SUMMARY")
print("=" * 45)

show_distribution("TRAINING DATA (80%)", train_df)
show_distribution("TESTING DATA (20%)", test_df)

print("\nMessage overlap between sets:", len(overlap))
print("Training file:", train_path)
print("Testing file:", test_path)
print("\nDATA PREPARATION COMPLETED SUCCESSFULLY!")
