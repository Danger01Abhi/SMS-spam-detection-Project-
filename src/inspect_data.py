
from pathlib import Path
import pandas as pd

# 1. Find the project and dataset folders
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# 2. Find the extracted SMS dataset
if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Data folder not found: {DATA_DIR}"
    )

matches = [
    path
    for path in DATA_DIR.rglob("*")
    if path.is_file()
    and path.name.lower().startswith("smsspamcollection")
    and path.suffix.lower() not in {".zip", ".gz"}
]

if not matches:
    raise FileNotFoundError(
        "SMSSpamCollection file not found inside the data folder. "
        "Download and extract the dataset first."
    )

# Prefer the original filename without an extension
DATA_PATH = next(
    (p for p in matches
     if p.name.lower() == "smsspamcollection"),
    matches[0]
)

print("=" * 55)
print("SMS SPAM DETECTION - DATA INSPECTION")
print("=" * 55)
print(f"\nDataset file: {DATA_PATH}")

# 3. Read the tab-separated dataset
df = pd.read_csv(
    DATA_PATH,
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="latin-1"
)

# 4. Standardize label formatting
df["label"] = df["label"].str.strip().str.lower()

# 5. Display dataset dimensions
print("\n1. DATASET DIMENSIONS")
print("-" * 35)
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

# 6. Display column names and data types
print("\n2. COLUMN INFORMATION")
print("-" * 35)
print(df.dtypes)

# 7. Show sample messages
print("\n3. FIRST FIVE MESSAGES")
print("-" * 35)
print(df.head().to_string(index=False))

# 8. Count normal and spam messages
print("\n4. CLASS DISTRIBUTION")
print("-" * 35)

counts = df["label"].value_counts()
percentages = (df["label"].value_counts(normalize=True) * 100).round(2)

distribution = pd.DataFrame({
    "Count": counts,
    "Percentage": percentages
})

print(distribution.to_string())

# 9. Check for missing values
print("\n5. MISSING VALUES")
print("-" * 35)
print(df.isna().sum())

# 10. Check duplicate rows
print("\n6. DUPLICATE ROWS")
print("-" * 35)
print(f"Duplicate rows: {df.duplicated().sum()}")

# 11. Check empty messages
print("\n7. EMPTY MESSAGES")
print("-" * 35)

empty_messages = df["message"].fillna("").str.strip().eq("").sum()
print(f"Empty messages: {empty_messages}")

# 12. Check which labels exist
print("\n8. LABEL VALIDATION")
print("-" * 35)

labels_found = set(df["label"].dropna().unique())
print(f"Labels found: {sorted(labels_found)}")

unexpected = labels_found - {"ham", "spam"}
print(f"Unexpected labels: {unexpected}")

# 13. Show example messages from each class
print("\n9. MESSAGE EXAMPLES")
print("-" * 35)

for label in ["ham", "spam"]:
    print(f"\n{label.upper()} examples:")

    examples = df.loc[
        df["label"] == label, "message"
    ].head(3)

    for number, message in enumerate(examples, start=1):
        print(f"{number}. {message}")

print("\n" + "=" * 55)
print("DATA INSPECTION COMPLETED")
print("=" * 55)
