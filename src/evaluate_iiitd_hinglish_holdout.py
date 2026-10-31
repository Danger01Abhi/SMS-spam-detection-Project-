from pathlib import Path
import re
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]

candidate_path = ROOT / "data/multilingual/iiitd_hinglish_deduplicated.csv"
test_path = ROOT / "reports/multilingual/iiitd_experimental_test_predictions.csv"
report_path = ROOT / "reports/multilingual/iiitd_hinglish_holdout_metrics.txt"

if report_path.exists():
    raise FileExistsError(f"Report already exists: {report_path}")

candidates = pd.read_csv(candidate_path)
test = pd.read_csv(test_path)

def normalize(value):
    return re.sub(r"\s+", " ", str(value).casefold().strip())

candidate_map = {
    normalize(row["message"]): row["label"]
    for _, row in candidates.iterrows()
}

matched = []

for _, row in test.iterrows():
    key = normalize(row["message"])
    if key in candidate_map:
        matched.append({
            "message": row["message"],
            "source_label": candidate_map[key],
            "actual_label": row["actual_label"],
            "predicted_label": row["predicted_label"],
            "correct": row["actual_label"] == row["predicted_label"]
        })

if not matched:
    raise ValueError("No Hinglish candidates found in the held-out test set.")

df = pd.DataFrame(matched)

accuracy = accuracy_score(df["actual_label"], df["predicted_label"])
macro_f1 = f1_score(
    df["actual_label"],
    df["predicted_label"],
    average="macro"
)

classification = classification_report(
    df["actual_label"],
    df["predicted_label"],
    labels=["HAM", "SPAM"],
    zero_division=0
)

matrix = confusion_matrix(
    df["actual_label"],
    df["predicted_label"],
    labels=["HAM", "SPAM"]
)

report = f"""IIIT-D HINGLISH-KEYWORD CANDIDATE HOLDOUT EVALUATION

Held-out candidate messages: {len(df)}
HAM: {(df.actual_label == "HAM").sum()}
SPAM: {(df.actual_label == "SPAM").sum()}

Accuracy: {accuracy:.4f}
Macro F1: {macro_f1:.4f}

Classification report:
{classification}

Confusion matrix (rows=actual, columns=predicted):
{matrix}

NOTE:
These are keyword-selected candidate messages, not all manually
verified Hinglish messages. Results are preliminary and do not
represent a verified Hinglish-only benchmark.
"""

report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(report, encoding="utf-8")

print("\n===== HINGLISH CANDIDATE HOLDOUT RESULTS =====")
print("Held-out candidate messages:", len(df))
print("HAM:", (df.actual_label == "HAM").sum())
print("SPAM:", (df.actual_label == "SPAM").sum())
print(f"Accuracy: {accuracy:.2%}")
print(f"Macro F1: {macro_f1:.4f}")
print("\nClassification report:")
print(classification)
print("\nConfusion matrix:")
print(matrix)
print("\nReport saved:", report_path)
