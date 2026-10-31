
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# =========================================
# 1. PROJECT PATHS
# =========================================

ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data" / "processed" / "train.csv"
TEST_PATH = ROOT / "data" / "processed" / "test.csv"

BILSTM_PATH = ROOT / "models" / "sms_spam_bilstm.keras"

MODEL_DIR = ROOT / "models" / "advanced"
REPORT_DIR = ROOT / "reports" / "advanced"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.50


# =========================================
# 2. LOAD DATASET
# =========================================

def load_dataset():

    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {TRAIN_PATH}"
        )

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Testing dataset not found: {TEST_PATH}"
        )

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    required_columns = {"label", "message"}

    for name, df in [
        ("Training", train),
        ("Testing", test)
    ]:

        if not required_columns.issubset(df.columns):
            raise ValueError(
                f"{name} CSV needs these columns: "
                f"{required_columns}. "
                f"Found: {list(df.columns)}"
            )

        if df["message"].isna().any() or df["label"].isna().any():
            raise ValueError(
                f"{name} CSV contains missing values."
            )

        df["label"] = (
            df["label"].astype(str).str.lower().str.strip()
        )

        df["message"] = df["message"].astype(str)

        if not df["label"].isin(["ham", "spam"]).all():
            raise ValueError(
                f"{name} CSV contains unexpected labels."
            )

    if set(train["message"]) & set(test["message"]):
        raise ValueError(
            "Identical messages appear in both datasets. "
            "Check the train/test split before continuing."
        )

    X_train = train["message"].astype(str)
    y_train = (train["label"] == "spam").astype(int)

    X_test = test["message"].astype(str)
    y_test = (test["label"] == "spam").astype(int)

    return X_train, y_train, X_test, y_test


# =========================================
# 3. CREATE CLASSIFIERS
# =========================================

def create_models():

    try:
        calibrated_svm = CalibratedClassifierCV(
            estimator=LinearSVC(
                class_weight="balanced",
                random_state=42
            ),
            cv=3,
            method="sigmoid"
        )

    except TypeError:
        # Compatibility with older scikit-learn
        calibrated_svm = CalibratedClassifierCV(
            base_estimator=LinearSVC(
                class_weight="balanced",
                random_state=42
            ),
            cv=3,
            method="sigmoid"
        )

    classifiers = {

        "Naive Bayes": MultinomialNB(
            alpha=0.5
        ),

        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ),

        "SVM": calibrated_svm,

        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            class_weight="balanced_subsample",
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    }

    return classifiers


# =========================================
# 4. EVALUATION FUNCTION
# =========================================

def evaluate_model(name, y_true, scores):

    predictions = (scores >= THRESHOLD).astype(int)

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    result = {

        "Model": name,

        "Threshold": THRESHOLD,

        "Accuracy": accuracy_score(
            y_true,
            predictions
        ),

        "Precision": precision_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "Recall": recall_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "F1-score": f1_score(
            y_true,
            predictions,
            zero_division=0
        ),

        "ROC-AUC": roc_auc_score(
            y_true,
            scores
        ),

        "PR-AUC": average_precision_score(
            y_true,
            scores
        ),

        "False Positives": int(fp),

        "False Negatives": int(fn)
    }

    # Save confusion matrix

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Ham", "Spam"]
    )

    display.plot(values_format="d")

    plt.title(f"{name} Confusion Matrix")
    plt.tight_layout()

    filename = (
        name.lower()
        .replace(" ", "_")
        + "_confusion_matrix.png"
    )

    plt.savefig(
        REPORT_DIR / filename,
        dpi=150
    )

    plt.close()

    return result


# =========================================
# 5. TRAIN CLASSICAL ML MODELS
# =========================================

def main():

    X_train, y_train, X_test, y_test = load_dataset()

    print("Dataset loaded successfully.")
    print("Training messages:", len(X_train))
    print("Testing messages:", len(X_test))

    models = create_models()

    results = []

    trained_models = {}

    for name, classifier in models.items():

        print("\n" + "=" * 45)
        print("Training:", name)
        print("=" * 45)

        pipeline = Pipeline([

            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    max_features=20000,
                    sublinear_tf=True
                )
            ),

            ("classifier", classifier)

        ])

        pipeline.fit(
            X_train,
            y_train
        )

        scores = pipeline.predict_proba(
            X_test
        )[:, 1]

        result = evaluate_model(
            name,
            y_test,
            scores
        )

        results.append(result)

        trained_models[name] = pipeline

        filename = (
            name.lower().replace(" ", "_")
            + ".joblib"
        )

        joblib.dump(
            pipeline,
            MODEL_DIR / filename
        )

        print("Accuracy:", round(result["Accuracy"], 4))
        print("Precision:", round(result["Precision"], 4))
        print("Recall:", round(result["Recall"], 4))
        print("F1-score:", round(result["F1-score"], 4))

    # =====================================
    # 6. EVALUATE EXISTING BiLSTM
    # =====================================

    print("\nEvaluating existing BiLSTM...")

    if not BILSTM_PATH.exists():
        raise FileNotFoundError(
            "Your existing BiLSTM model was not found."
        )

    bilstm = tf.keras.models.load_model(
        BILSTM_PATH
    )

    test_messages = tf.constant(
        X_test.tolist(),
        dtype=tf.string
    )

    bilstm_scores = bilstm.predict(
        test_messages,
        verbose=0
    ).ravel()

    bilstm_result = evaluate_model(
        "BiLSTM",
        y_test,
        bilstm_scores
    )

    results.append(bilstm_result)

    # =====================================
    # 7. SAVE COMPARISON
    # =====================================

    comparison = pd.DataFrame(results)

    comparison = comparison.sort_values(
        by="F1-score",
        ascending=False
    )

    comparison.to_csv(
        REPORT_DIR / "model_comparison.csv",
        index=False
    )

    best_name = comparison.iloc[0]["Model"]

    if best_name in trained_models:

        joblib.dump(
            trained_models[best_name],
            MODEL_DIR / "best_classical_model.joblib"
        )

    # =====================================
    # 8. CREATE COMPARISON GRAPH
    # =====================================

    comparison_plot = comparison.set_index("Model")[
        ["Accuracy", "Precision", "Recall", "F1-score"]
    ]

    comparison_plot.plot(
        kind="bar",
        figsize=(12, 6)
    )

    plt.title("SMS Spam Detection Model Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=20)
    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / "model_comparison.png",
        dpi=150
    )

    plt.close()

    # =====================================
    # 9. DISPLAY RESULTS
    # =====================================

    print("\nFINAL MODEL COMPARISON")

    print(
        comparison.round(4).to_string(index=False)
    )

    print("\nBest model by test F1-score:", best_name)

    print("\nReports saved in:")
    print(REPORT_DIR)

    print("\nModels saved in:")
    print(MODEL_DIR)


if __name__ == "__main__":
    main()
