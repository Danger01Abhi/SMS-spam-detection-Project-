
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ==================================================
# 1. PROJECT SETTINGS
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

MODEL_PATH = MODEL_DIR / "sms_spam_bilstm.keras"

SEED = 42

tf.keras.utils.set_random_seed(SEED)


# ==================================================
# 2. LOAD PREPARED DATA
# ==================================================

if not TRAIN_PATH.exists() or not TEST_PATH.exists():
    raise FileNotFoundError(
        "Training or testing CSV is missing. "
        "Run src/prepare_data.py first."
    )

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

required_columns = {"label", "message"}

if not required_columns.issubset(train_df.columns):
    raise ValueError("Training CSV must contain label and message.")

if not required_columns.issubset(test_df.columns):
    raise ValueError("Testing CSV must contain label and message.")

train_df = train_df.dropna(subset=["label", "message"]).copy()
test_df = test_df.dropna(subset=["label", "message"]).copy()

for frame in [train_df, test_df]:
    frame["label"] = frame["label"].astype(str).str.lower().str.strip()
    frame["message"] = frame["message"].astype(str)

    frame.drop(
        frame[
            ~frame["label"].isin(["ham", "spam"])
        ].index,
        inplace=True
    )

if train_df["label"].nunique() != 2:
    raise ValueError("Training data must contain ham and spam examples.")

if test_df.empty:
    raise ValueError("Testing data is empty.")

# Convert the labels to numbers:
# ham = 0, spam = 1

label_map = {"ham": 0, "spam": 1}

X_all = train_df["message"].astype(str)
y_all = train_df["label"].map(label_map).astype("int32")


X_test = tf.constant(
    test_df["message"].astype(str).tolist(),
    dtype=tf.string
)
y_test = test_df["label"].map(label_map).astype("int32").to_numpy()

print("=" * 55)
print("SMS SPAM DETECTION - DEEP LEARNING TRAINING")
print("=" * 55)

print("\nTraining samples:", len(X_all))
print("Testing samples:", len(X_test))


# ==================================================
# 3. CREATE A VALIDATION SET
# ==================================================

X_train, X_val, y_train, y_val = train_test_split(
    X_all,
    y_all,
    test_size=0.15,
    random_state=SEED,
    stratify=y_all
)

# Reset indices to keep data alignment simple

X_train = X_train.reset_index(drop=True)
X_val = X_val.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_val = y_val.reset_index(drop=True)


# Convert text into TensorFlow string tensors
X_train_tf = tf.constant(X_train.tolist(), dtype=tf.string)
X_val_tf = tf.constant(X_val.tolist(), dtype=tf.string)

print("\nTraining samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Testing samples:", len(X_test))


# ==================================================
# 4. CONVERT TEXT INTO NUMERICAL TOKENS
# ==================================================

MAX_TOKENS = 10000
SEQUENCE_LENGTH = 100

vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=MAX_TOKENS,
    output_mode="int",
    output_sequence_length=SEQUENCE_LENGTH,
    standardize="lower_and_strip_punctuation"
)

# Learn the vocabulary ONLY from the training messages.
# This prevents validation and test data from influencing
# vocabulary construction.

vectorizer.adapt(X_train_tf)

vocabulary_size = len(vectorizer.get_vocabulary())

print("\nVocabulary size:", vocabulary_size)
print("Maximum sequence length:", SEQUENCE_LENGTH)


# ==================================================
# 5. HANDLE CLASS IMBALANCE
# ==================================================

class_values = np.array([0, 1])

weights = compute_class_weight(
    class_weight="balanced",
    classes=class_values,
    y=y_train.to_numpy()
)

class_weights = {
    int(label): float(weight)
    for label, weight in zip(class_values, weights)
}

print("\nClass weights:", class_weights)


# ==================================================
# 6. BUILD THE BIDIRECTIONAL LSTM MODEL
# ==================================================

text_input = tf.keras.Input(
    shape=(),
    dtype=tf.string,
    name="message"
)

# Convert raw text into token sequences
x = vectorizer(text_input)

# Learn word representations
x = tf.keras.layers.Embedding(
    input_dim=vocabulary_size,
    output_dim=64,
    mask_zero=True,
    name="word_embedding"
)(x)

# Learn relationships between words in both directions
x = tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(32),
    name="bidirectional_lstm"
)(x)

x = tf.keras.layers.Dropout(0.30)(x)

x = tf.keras.layers.Dense(
    32,
    activation="relu"
)(x)

x = tf.keras.layers.Dropout(0.20)(x)

# One output score for spam probability
output = tf.keras.layers.Dense(
    1,
    activation="sigmoid",
    name="spam_probability"
)(x)

model = tf.keras.Model(
    inputs=text_input,
    outputs=output,
    name="sms_spam_detector"
)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc")
    ]
)

print("\nMODEL ARCHITECTURE")
model.summary()


# ==================================================
# 7. TRAIN THE MODEL
# ==================================================

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        verbose=1
    )
]

print("\nStarting training...")

history = model.fit(
    X_train_tf,
    y_train.to_numpy(dtype="float32"),
    validation_data=(
        X_val_tf,
        y_val.to_numpy(dtype="float32")
    ),
    epochs=15,
    batch_size=32,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)


# ==================================================
# 8. SAVE TRAINING HISTORY GRAPH
# ==================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history["accuracy"], label="Training")
axes[0].plot(history.history["val_accuracy"], label="Validation")
axes[0].set_title("Model Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()

axes[1].plot(history.history["loss"], label="Training")
axes[1].plot(history.history["val_loss"], label="Validation")
axes[1].set_title("Model Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()

fig.tight_layout()

fig.savefig(
    REPORTS_DIR / "training_history.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)


# ==================================================
# 9. EVALUATE ON THE UNTOUCHED TEST SET
# ==================================================

print("\nEvaluating on test data...")

probabilities = model.predict(
    X_test,
    batch_size=32,
    verbose=0
).ravel()

# Threshold of 0.5: the initial decision rule
y_pred = (probabilities >= 0.5).astype("int32")

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test, y_pred, zero_division=0
)
recall = recall_score(
    y_test, y_pred, zero_division=0
)
f1 = f1_score(
    y_test, y_pred, zero_division=0
)

report = classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Ham (Normal)", "Spam"],
    digits=4,
    zero_division=0
)

print("\nTEST RESULTS")
print("=" * 55)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

print("\nClassification Report:")
print(report)


# ==================================================
# 10. SAVE CONFUSION MATRIX
# ==================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Ham (Normal)", "Spam"]
)

display.plot(values_format="d")

plt.title("SMS Spam Detection - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    REPORTS_DIR / "confusion_matrix.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ==================================================
# 11. SAVE THE MODEL AND METRICS
# ==================================================

# The saved model includes the TextVectorization layer,
# so it can accept raw SMS messages directly.

model.save(MODEL_PATH)

metrics_path = REPORTS_DIR / "model_results.txt"

with metrics_path.open("w", encoding="utf-8") as file:
    file.write("SMS SPAM DETECTION - TEST RESULTS\n")
    file.write("=" * 40 + "\n\n")
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

print("\nSAVED FILES")
print("=" * 55)
print("Model:", MODEL_PATH)
print("Training graph:", REPORTS_DIR / "training_history.png")
print("Confusion matrix:", REPORTS_DIR / "confusion_matrix.png")
print("Metrics:", metrics_path)

print("\nMODEL TRAINING AND EVALUATION COMPLETED!")
