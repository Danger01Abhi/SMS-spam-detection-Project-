
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from lime.lime_text import LimeTextExplainer


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "advanced"
    / "tuned_logistic_regression.joblib"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "advanced"


def main():
    # Check whether the trained model exists
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load the existing trained model
    model = joblib.load(MODEL_PATH)

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "The loaded model does not support predict_proba()."
        )

    # Get the model's class names
    classes = getattr(model, "classes_", None)

    if classes is None and hasattr(model, "named_steps"):
        final_estimator = list(model.named_steps.values())[-1]
        classes = getattr(final_estimator, "classes_", None)

    if classes is None:
        raise ValueError(
            "Could not identify the model's class labels."
        )

    class_names = [str(label) for label in classes]

    # Read an SMS message
    print("\n===== LIME SMS EXPLANATION =====")

    message = input(
        "\nEnter an SMS message to analyze: "
    ).strip()

    if not message:
        message = (
            "Congratulations! You have won a free prize. "
            "Click here now to claim your reward."
        )
        print("\nUsing a sample SMS message.")

    # Predict the message
    probabilities = model.predict_proba([message])[0]

    predicted_index = int(np.argmax(probabilities))
    prediction = class_names[predicted_index]
    probability = float(probabilities[predicted_index])

    # Convert the model's class into a readable label
    raw_label = str(prediction).strip().lower()

    if raw_label in ("spam", "1"):
        readable_prediction = "SPAM"
    elif raw_label in ("ham", "0"):
        readable_prediction = "HAM"
    else:
        readable_prediction = f"UNKNOWN CLASS ({prediction})"

    print("\nSMS MESSAGE:")
    print(message)

    print("\n========== SMS CLASSIFICATION ==========")
    print(f"FINAL RESULT: {readable_prediction}")
    print(f"CONFIDENCE: {probability * 100:.2f}%")
    print("========================================")
    print(
        f"Model probability for this prediction: "
        f"{probability * 100:.2f}%"
    )

    # Create the LIME explainer
    explainer = LimeTextExplainer(
        class_names=class_names
    )

    # Generate an explanation for the predicted class
    explanation = explainer.explain_instance(
        message,
        model.predict_proba,
        num_features=10,
        labels=[predicted_index]
    )

    # Display important words in the terminal
    print("\nIMPORTANT WORDS:")
    for word, weight in explanation.as_list(
        label=predicted_index
    ):
        print(f"{word}: {weight:.4f}")

    # Use a unique filename so previous explanations are preserved
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    graph_path = (
        OUTPUT_DIR / f"lime_explanation_{timestamp}.png"
    )

    html_path = (
        OUTPUT_DIR / f"lime_explanation_{timestamp}.html"
    )

    # Save the explanation graph
    figure = explanation.as_pyplot_figure(
        label=predicted_index
    )

    figure.savefig(
        graph_path,
        bbox_inches="tight",
        dpi=150
    )

    plt.close(figure)

    # Save the interactive HTML explanation
    explanation.save_to_file(str(html_path))

    print("\n===== EXPLANATION SAVED =====")
    print(f"Graph: {graph_path}")
    print(f"HTML report: {html_path}")


if __name__ == "__main__":
    main()
