
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Locate project folders and dataset
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

matches = [
    path for path in DATA_DIR.rglob("*")
    if path.is_file()
    and path.name.lower().startswith("smsspamcollection")
    and path.suffix.lower() not in {".zip", ".gz"}
]

if not matches:
    raise FileNotFoundError(
        "SMSSpamCollection was not found inside the data folder."
    )

DATA_PATH = next(
    (p for p in matches
     if p.name.lower() == "smsspamcollection"),
    matches[0]
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

print("=" * 55)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 55)

print("\nOriginal dataset shape:", df.shape)

# Check missing values before making analysis columns
print("\nMissing values:")
print(df.isna().sum())

print("\nDuplicate rows:", df.duplicated().sum())

# Use recognized labels for class-specific analysis
valid_labels = df["label"].isin(["ham", "spam"])
invalid_count = int((~valid_labels).sum())

if invalid_count:
    print("\nRows with invalid labels:", invalid_count)

df = df.loc[valid_labels].copy()
df["message"] = df["message"].fillna("").astype(str)

# --------------------------------------------------
# 3. Class distribution
# --------------------------------------------------
class_counts = df["label"].value_counts().reindex(
    ["ham", "spam"], fill_value=0
)

class_percentages = (
    df["label"].value_counts(normalize=True)
    .reindex(["ham", "spam"], fill_value=0) * 100
)

print("\nCLASS DISTRIBUTION")
print("-" * 35)

for label in ["ham", "spam"]:
    print(
        f"{label.upper()}: {class_counts[label]} messages "
        f"({class_percentages[label]:.2f}%)"
    )

fig, ax = plt.subplots(figsize=(7, 5))

bars = ax.bar(
    ["Ham (Normal)", "Spam"],
    [class_counts["ham"], class_counts["spam"]]
)

ax.set_title("SMS Dataset: Class Distribution")
ax.set_xlabel("Message Category")
ax.set_ylabel("Number of Messages")

for bar, label in zip(bars, ["ham", "spam"]):
    count = class_counts[label]
    percent = class_percentages[label]
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{count}\n({percent:.1f}%)",
        ha="center",
        va="bottom"
    )

fig.tight_layout()
fig.savefig(
    REPORTS_DIR / "class_distribution.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close(fig)

# --------------------------------------------------
# 4. Calculate message lengths
# --------------------------------------------------
df["character_count"] = df["message"].str.len()
df["word_count"] = df["message"].str.split().str.len()

print("\nMESSAGE LENGTH STATISTICS")
print("-" * 35)

length_stats = df.groupby("label")[
    ["character_count", "word_count"]
].agg(["mean", "median", "min", "max"])

print(length_stats.round(2).to_string())

# --------------------------------------------------
# 5. Compare message lengths by class
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

length_data = [
    df.loc[df["label"] == label, "character_count"]
    for label in ["ham", "spam"]
]

ax.boxplot(
    length_data,
    tick_labels=["Ham (Normal)", "Spam"],
    showfliers=True
)

ax.set_title("SMS Length by Message Category")
ax.set_xlabel("Message Category")
ax.set_ylabel("Number of Characters")

fig.tight_layout()
fig.savefig(
    REPORTS_DIR / "message_length_comparison.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close(fig)

# --------------------------------------------------
# 6. Save a text summary for the project report
# --------------------------------------------------
summary_path = REPORTS_DIR / "eda_summary.txt"

with summary_path.open("w", encoding="utf-8") as file:
    file.write("SMS SPAM DETECTION - EDA SUMMARY\n")
    file.write("=" * 40 + "\n\n")

    file.write(f"Dataset rows after label validation: {len(df)}\n")
    file.write(f"Invalid-label rows excluded from analysis: {invalid_count}\n")
    file.write(
        f"Duplicate rows (before any removal): "
        f"{pd.read_csv(DATA_PATH, sep=chr(9), header=None).duplicated().sum()}\n"
    )

    file.write("\nClass distribution:\n")
    for label in ["ham", "spam"]:
        file.write(
            f"{label}: {class_counts[label]} "
            f"({class_percentages[label]:.2f}%)\n"
        )

    file.write("\nMessage length statistics:\n")
    file.write(length_stats.round(2).to_string())
    file.write("\n")

print("\nEDA OUTPUT FILES")
print("-" * 35)
print("1. reports/class_distribution.png")
print("2. reports/message_length_comparison.png")
print("3. reports/eda_summary.txt")

print("\nEDA COMPLETED SUCCESSFULLY!")
