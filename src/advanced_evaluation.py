
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    GridSearchCV
)

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    roc_auc_score,
    average_precision_score
)


# =====================================
# 1. PROJECT PATHS
# =====================================

ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data" / "processed" / "train.csv"
TEST_PATH = ROOT / "data" / "processed" / "test.csv"

REPORT_DIR = ROOT / "reports" / "advanced"
MODEL_DIR = ROOT / "models" / "advanced"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# =====================================
# 2. LOAD DATA
# =====================================

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

for df in [train, test]:

    if not {"label", "message"}.issubset(df.columns):
        raise ValueError(
            "Dataset must contain label and message columns."
        )

    if df[["label", "message"]].isna().any().any():
        raise ValueError("Dataset contains missing values.")

    df["label"] = df["label"].astype(str).str.lower().str.strip()

    if not df["label"].isin(["ham", "spam"]).all():
        raise ValueError("Unexpected label found.")

    df["message"] = df["message"].astype(str)


X_train = train["message"]
y_train = (train["label"] == "spam").astype(int)

X_test = test["message"]
y_test = (test["label"] == "spam").astype(int)


# =====================================
# 3. FIVE-FOLD CROSS VALIDATION
# =====================================

models = {

    "Logistic Regression": Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=20000
        )),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ))
    ]),

    "Naive Bayes": Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=20000
        )),
        ("classifier", MultinomialNB())
    ])
}


cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc"
]

cv_results = []

for name, model in models.items():

    print("\nCross-validation:", name)

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    result = {"Model": name}

    for metric in scoring:

        values = scores["test_" + metric]

        result[metric] = values.mean()
        result[metric + "_std"] = values.std()

        print(
            metric,
            round(values.mean(), 4),
            "+/-",
            round(values.std(), 4)
        )

    cv_results.append(result)


pd.DataFrame(cv_results).to_csv(
    REPORT_DIR / "cross_validation_results.csv",
    index=False
)


# =====================================
# 4. HYPERPARAMETER TUNING
# =====================================

print("\nStarting Logistic Regression tuning...")

pipeline = models["Logistic Regression"]

parameters = {

    "tfidf__ngram_range": [
        (1, 1),
        (1, 2)
    ],

    "tfidf__max_features": [
        10000,
        20000
    ],

    "classifier__C": [
        0.1,
        1,
        5
    ]
}

search = GridSearchCV(
    pipeline,
    parameters,
    scoring="f1",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

search.fit(X_train, y_train)

best_model = search.best_estimator_

joblib.dump(
    best_model,
    MODEL_DIR / "tuned_logistic_regression.joblib"
)

print("\nBest parameters:")
print(search.best_params_)

print("\nBest cross-validation F1:")
print(search.best_score_)


# =====================================
# 5. TEST SET EVALUATION
# =====================================

probabilities = best_model.predict_proba(
    X_test
)[:, 1]

predictions = (probabilities >= 0.5).astype(int)

print("\nClassification Report")

report = classification_report(
    y_test,
    predictions,
    target_names=["Ham", "Spam"],
    zero_division=0
)

print(report)

with open(
    REPORT_DIR / "classification_report.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(report)


# =====================================
# 6. CONFUSION MATRIX
# =====================================

ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    display_labels=["Ham", "Spam"],
    values_format="d"
)

plt.title("Tuned Logistic Regression")
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "tuned_confusion_matrix.png",
    dpi=150
)

plt.close()


# =====================================
# 7. ROC CURVE
# =====================================

RocCurveDisplay.from_predictions(
    y_test,
    probabilities
)

plt.title("SMS Spam Detection ROC Curve")
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "roc_curve.png",
    dpi=150
)

plt.close()


# =====================================
# 8. PRECISION RECALL CURVE
# =====================================

PrecisionRecallDisplay.from_predictions(
    y_test,
    probabilities
)

plt.title("SMS Spam Precision Recall Curve")
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "precision_recall_curve.png",
    dpi=150
)

plt.close()


# =====================================
# 9. ERROR ANALYSIS
# =====================================

error_data = pd.DataFrame({

    "message": X_test.reset_index(drop=True),

    "actual": y_test.reset_index(drop=True),

    "predicted": predictions,

    "spam_score": probabilities

})

false_positives = error_data[
    (error_data["actual"] == 0) &
    (error_data["predicted"] == 1)
]

false_negatives = error_data[
    (error_data["actual"] == 1) &
    (error_data["predicted"] == 0)
]

false_positives.to_csv(
    REPORT_DIR / "false_positives.csv",
    index=False
)

false_negatives.to_csv(
    REPORT_DIR / "false_negatives.csv",
    index=False
)


# =====================================
# 10. FINAL SUMMARY
# =====================================

print("\nROC-AUC:",
      round(roc_auc_score(y_test, probabilities), 4))

print("PR-AUC:",
      round(average_precision_score(y_test, probabilities), 4))

print("False positives:", len(false_positives))

print("False negatives:", len(false_negatives))

print("\nAll evaluation reports saved successfully!")

print(REPORT_DIR)
