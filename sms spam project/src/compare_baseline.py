
from pathlib import Path
import re

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# ==================================================
# 1. PROJECT PATHS
# ==================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"


# ==================================================
# 2. LOAD AND VALIDATE THE DATA
# ==================================================

if not TRAIN_PATH.exists() or not TEST_PATH.exists():
    raise FileNotFoundError(
        "train.csv or test.csv was not found. "
        "Run src/prepare_data.py first."
    )

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

for name, frame in [
    ("Training", train_df),
    ("Testing", test_df),
]:
    if not {"label", "message"}.issubset(frame.columns):
        raise ValueError(
            f"{name} CSV must contain label and message columns."
        )

    frame.dropna(subset=["label", "message"], inplace=True)
    frame["label"] = (
        frame["label"].astype(str).str.strip().str.lower()
    )
    frame["message"] = frame["message"].astype(str)

    frame.drop(
        frame[~frame["label"].isin(["ham", "spam"])].index,
        inplace=True,
    )

if train_df["label"].nunique() != 2:
    raise ValueError("Training data must contain both ham and spam.")

if test_df.empty:
    raise ValueError("Testing dataset is empty.")

label_map = {"ham": 0, "spam": 1}

X_all = train_df["message"]
y_all = train_df["label"].map(label_map).astype(int)

X_test = test_df["message"]
y_test = test_df["label"].map(label_map).astype(int)

print("=" * 55)
print("TF-IDF + LOGISTIC REGRESSION BASELINE")
print("=" * 55)

print("Training CSV rows:", len(X_all))
print("Testing rows:", len(X_test))


# ==================================================
# 3. RECREATE THE LSTM TRAINING SPLIT
# ==================================================

# Match train_model.py: 15% of the training CSV is
# reserved for validation, using the same random seed.

X_fit, X_val, y_fit, y_val = train_test_split(
    X_all,
    y_all,
    test_size=0.15,
    random_state=42,
    stratify=y_all,
)

print("Rows used to fit baseline:", len(X_fit))
print("Validation rows reserved:", len(X_val))
print("Test rows reserved:", len(X_test))


# ==================================================
# 4. BUILD THE BASELINE PIPELINE
# ==================================================

baseline = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            max_features=20000,
            sublinear_tf=True,
        ),
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
    ),
])


# ==================================================
# 5. TRAIN THE BASELINE
# ==================================================

print("\nTraining baseline model...")

baseline.fit(X_fit, y_fit)

print("Baseline training completed!")


# ==================================================
# 6. EVALUATE USING THE HELD-OUT TEST SET
# ==================================================

y_pred = baseline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test, y_pred, pos_label=1, zero_division=0
)
recall = recall_score(
    y_test, y_pred, pos_label=1, zero_division=0
)
f1 = f1_score(
    y_test, y_pred, pos_label=1, zero_division=0
)

report = classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Ham (Normal)", "Spam"],
    digits=4,
    zero_division=0,
)

print("\nBASELINE TEST RESULTS")
print("=" * 55)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

print("\nClassification Report:")
print(report)


# ==================================================
# 7. SAVE THE BASELINE CONFUSION MATRIX
# ==================================================

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Ham (Normal)", "Spam"],
)

display.plot(values_format="d")
plt.title("TF-IDF + Logistic Regression Confusion Matrix")
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "baseline_confusion_matrix.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ==================================================
# 8. SAVE THE BASELINE MODEL AND METRICS
# ==================================================

joblib.dump(
    baseline,
    MODEL_DIR / "tfidf_logistic_regression.joblib",
)

baseline_metrics = {
    "Model": "TF-IDF + Logistic Regression",
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1-score": f1,
}

comparison_rows = [baseline_metrics]

results_path = REPORT_DIR / "model_results.txt"

# Read the existing LSTM results so both models can
# be compared if that report was generated by training.

if results_path.exists():
    saved_results = results_path.read_text(encoding="utf-8")

    dl_metrics = {"Model": "Bidirectional LSTM"}
    for metric in ["Accuracy", "Precision", "Recall", "F1-score"]:
        pattern = rf"^{re.escape(metric)}:\s*([0-9]*\.?[0-9]+)"
        match = re.search(
            pattern, saved_results, flags=re.MULTILINE
        )

        if match:
            dl_metrics[metric] = float(match.group(1))

    if all(
        metric in dl_metrics
        for metric in ["Accuracy", "Precision", "Recall", "F1-score"]
    ):
        comparison_rows.append(dl_metrics)
    else:
        print(
            "\nNote: Could not read all LSTM metrics from "
            "reports/model_results.txt."
        )
else:
    print(
        "\nNote: reports/model_results.txt was not found. "
        "Baseline results will still be saved."
    )

comparison_df = pd.DataFrame(comparison_rows)

comparison_df.to_csv(
    REPORT_DIR / "model_comparison.csv",
    index=False,
)

baseline_report_path = REPORT_DIR / "baseline_results.txt"

with baseline_report_path.open("w", encoding="utf-8") as file:
    file.write("TF-IDF + LOGISTIC REGRESSION RESULTS\n")
    file.write("=" * 45 + "\n\n")
    file.write(f"Test samples: {len(y_test)}\n")
    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write(f"Precision: {precision:.4f}\n")
    file.write(f"Recall: {recall:.4f}\n")
    file.write(f"F1-score: {f1:.4f}\n\n")
    file.write("Classification Report:\n")
    file.write(report)
    file.write("\nConfusion Matrix:\n")
    file.write(str(cm))
    file.write("\n")


# ==================================================
# 9. PLOT THE MODEL COMPARISON
# ==================================================

if len(comparison_df) == 2:
    metric_columns = [
        "Accuracy", "Precision", "Recall", "F1-score"
    ]

    ax = (
        comparison_df
        .set_index("Model")[metric_columns]
        .plot(kind="bar", figsize=(10, 5))
    )

    ax.set_title("Deep Learning vs Traditional ML")
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "model_comparison.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


# ==================================================
# 10. FINAL SUMMARY
# ==================================================

print("\nMODEL COMPARISON")
print("=" * 55)
print(comparison_df.round(4).to_string(index=False))

print("\nSAVED FILES")
print("Model:", MODEL_DIR / "tfidf_logistic_regression.joblib")
print("Baseline results:", baseline_report_path)
print("Baseline confusion matrix:", REPORT_DIR / "baseline_confusion_matrix.png")
print("Comparison table:", REPORT_DIR / "model_comparison.csv")

if len(comparison_df) == 2:
    print("Comparison chart:", REPORT_DIR / "model_comparison.png")

print("\nBASELINE TRAINING AND EVALUATION COMPLETED!")
