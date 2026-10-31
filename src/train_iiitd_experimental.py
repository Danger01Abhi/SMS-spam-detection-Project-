from pathlib import Path
from collections import defaultdict
import re
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "multilingual" / "iiitd_extracted"
MODEL_DIR = ROOT / "models" / "multilingual"
REPORT_DIR = ROOT / "reports" / "multilingual"

MODEL_PATH = MODEL_DIR / "iiitd_experimental_sms_model.joblib"
METRICS_PATH = REPORT_DIR / "iiitd_experimental_metrics.txt"
PREDICTIONS_PATH = REPORT_DIR / "iiitd_experimental_test_predictions.csv"

# Never overwrite an existing model or report.
for path in (MODEL_PATH, METRICS_PATH, PREDICTIONS_PATH):
    if path.exists():
        raise FileExistsError(
            f"Output already exists: {path}. Nothing was overwritten."
        )

if not DATA_DIR.exists():
    raise FileNotFoundError(f"Dataset folder not found: {DATA_DIR}")

records = []

for folder, label in (("Ham SMSes", "HAM"), ("Spam SMSes", "SPAM")):
    folder_path = DATA_DIR / folder

    if not folder_path.exists():
        raise FileNotFoundError(f"Missing dataset folder: {folder_path}")

    files = sorted(folder_path.glob("*.txt"))

    for file_path in files:
        message = file_path.read_text(
            encoding="utf-8",
            errors="replace"
        ).replace("\ufeff", "").strip()

        if message:
            records.append({
                "message": message,
                "label": label,
                "filename": file_path.name
            })

if not records:
    raise ValueError("No SMS messages were loaded.")

# Normalize only for duplicate detection; preserve original messages.
groups = defaultdict(list)

for record in records:
    normalized = re.sub(
        r"\s+",
        " ",
        record["message"].casefold().strip()
    )
    groups[normalized].append(record)

unique_records = []
conflicting_rows = 0
duplicate_copies = 0

for group in groups.values():
    labels = {item["label"] for item in group}

    if len(labels) > 1:
        conflicting_rows += len(group)
        continue

    unique_records.append(group[0])
    duplicate_copies += len(group) - 1

if len(unique_records) < 100:
    raise ValueError("Too few unique messages to train safely.")

X = [item["message"] for item in unique_records]
y = [item["label"] for item in unique_records]

if len(set(y)) != 2:
    raise ValueError("Both HAM and SPAM classes are required.")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

features = FeatureUnion([
    (
        "word",
        TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_features=100000,
            sublinear_tf=True
        )
    ),
    (
        "character",
        TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            min_df=2,
            max_features=150000,
            sublinear_tf=True
        )
    )
])

model = Pipeline([
    ("features", features),
    ("classifier", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        C=2.0,
        random_state=42
    ))
])

print("\nTraining experimental IIIT-D SMS model...")
print("Training messages:", len(X_train))
print("Testing messages:", len(X_test))

model.fit(X_train, y_train)
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
macro_f1 = f1_score(y_test, predictions, average="macro")

report = classification_report(
    y_test,
    predictions,
    labels=["HAM", "SPAM"],
    zero_division=0
)

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=["HAM", "SPAM"]
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

artifact = {
    "model": model,
    "model_name": "IIIT-D experimental word + character TF-IDF",
    "classes": ["HAM", "SPAM"],
    "training_dataset": "IIIT-D SMS Spam Dataset",
    "random_state": 42
}

joblib.dump(artifact, MODEL_PATH)

pd.DataFrame({
    "message": X_test,
    "actual_label": y_test,
    "predicted_label": predictions,
    "correct": [
        actual == predicted
        for actual, predicted in zip(y_test, predictions)
    ]
}).to_csv(PREDICTIONS_PATH, index=False, encoding="utf-8-sig")

metrics_text = f"""IIIT-D EXPERIMENTAL SMS MODEL

Original messages loaded: {len(records)}
Exact duplicate copies removed: {duplicate_copies}
Rows excluded due to conflicting labels: {conflicting_rows}
Unique messages used: {len(unique_records)}

Training messages: {len(X_train)}
Testing messages: {len(X_test)}

HAM messages used: {y.count("HAM")}
SPAM messages used: {y.count("SPAM")}

Accuracy: {accuracy:.4f}
Macro F1: {macro_f1:.4f}

Classification report:
{report}

Confusion matrix (rows=actual, columns=predicted):
{matrix}

IMPORTANT:
This is an experimental model trained on the full IIIT-D dataset.
These results are not a verified Hinglish-only evaluation.
The model requires separate testing on manually verified Hinglish
and Hindi messages before claiming multilingual performance.
"""

METRICS_PATH.write_text(metrics_text, encoding="utf-8")

print("\n========== TRAINING COMPLETED ==========")
print(f"Unique dataset messages: {len(unique_records)}")
print(f"Training messages: {len(X_train)}")
print(f"Testing messages: {len(X_test)}")
print(f"Accuracy: {accuracy:.2%}")
print(f"Macro F1: {macro_f1:.4f}")
print("\nClassification report:")
print(report)
print("\nConfusion matrix:")
print(matrix)

print("\nMODEL SAVED:")
print(MODEL_PATH)
print("\nMETRICS SAVED:")
print(METRICS_PATH)
print("\nTEST PREDICTIONS SAVED:")
print(PREDICTIONS_PATH)
