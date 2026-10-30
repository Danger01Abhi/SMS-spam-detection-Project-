
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# 1. Set up project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Find the dataset, including if it has a .txt extension
files = [
    p for p in DATA_DIR.rglob("*")
    if p.is_file()
    and p.name.lower().startswith("smsspamcollection")
    and p.suffix.lower() not in {".zip", ".gz"}
]

if not files:
    raise FileNotFoundError(
        "Dataset not found. Check your project's data folder."
    )

DATA_PATH = next(
    (p for p in files
     if p.name.lower() == "smsspamcollection"),
    files[0]
)

# --------------------------------------------------
# 2. Load the dataset
# --------------------------------------------------

df = pd.read_csv(
    DATA_PATH,
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="latin-1"
)

df["label"] = df["label"].str.strip().str.lower()

# Basic validation
if df[["label", "message"]].isna().any().any():
    raise ValueError(
        "Missing labels or messages found. Inspect the dataset first."
    )

if not set(df["label"].unique()).issubset({"ham", "spam"}):
    raise ValueError("Unexpected labels found in the dataset.")

# Add analysis columns without altering the original text
df["char_count"] = df["message"].str.len()
df["word_count"] = df["message"].str.split().str.len()

sns.set_theme(style="whitegrid")

print("=" * 50)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 50)

# --------------------------------------------------
# 3. Class distribution
# --------------------------------------------------

class_counts = (
    df["label"]
    .value_counts()
    .reindex(["ham", "spam"], fill_value=0)
)

print("\nMessage counts:")
print(class_counts)

print("\nClass percentages:")
print((class_counts / len(df) * 100).round(2))

plt.figure(figsize=(7, 5))

ax = sns.barplot(
    x=class_counts.index,
    y=class_counts.values
)

ax.bar_label(ax.containers[0], fmt="%d", padding=4)

plt.title("Distribution of Normal and Spam Messages")
plt.xlabel("Message category")
plt.ylabel("Number of messages")
plt.tight_layout()

plt.savefig(
    REPORTS_DIR / "class_distribution.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()

# --------------------------------------------------
# 4. Message length distribution
# --------------------------------------------------

print("\nCharacter count summary:")
print(df.groupby("label")["char_count"].describe().round(2))

plt.figure(figsize=(10, 5))

sns.histplot(
    data=df,
    x="char_count",
    hue="label",
    bins=40,
    element="step",
    stat="count",
    common_norm=False
)

plt.title("SMS Message Length Distribution")
plt.xlabel("Number of characters")
plt.ylabel("Number of messages")
plt.xlim(left=0)
plt.tight_layout()

plt.savefig(
    REPORTS_DIR / "message_length_distribution.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()

# --------------------------------------------------
# 5. Compare message lengths by class
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="label",
    y="char_count",
    showfliers=False
)

plt.title("Message Length Comparison: Ham vs Spam")
plt.xlabel("Message category")
plt.ylabel("Number of characters")
plt.tight_layout()

plt.savefig(
    REPORTS_DIR / "message_length_comparison.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close()

# --------------------------------------------------
# 6. Compare average message lengths
# --------------------------------------------------

summary = df.groupby("label").agg(
    message_count=("message", "size"),
    average_characters=("char_count", "mean"),
    median_characters=("char_count", "median"),
    average_words=("word_count", "mean")
).round(2)

print("\nMessage length summary:")
print(summary)

# Save summary for your project report
summary.to_csv(REPORTS_DIR / "eda_summary.csv")

print("\nGraphs and summary saved in:")
print(REPORTS_DIR)

print("\nEDA COMPLETED SUCCESSFULLY")
