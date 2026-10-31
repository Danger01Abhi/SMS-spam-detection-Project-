from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]

TFIDF_PATH = ROOT / "models" / "tfidf_logistic_regression.joblib"
BILSTM_PATH = ROOT / "models" / "sms_spam_bilstm.keras"

THRESHOLD = 0.5

TEST_CASES = [
    # English
    {"language": "English", "expected": "HAM",
     "message": "Are you coming to class today? I will save you a seat."},
    {"language": "English", "expected": "HAM",
     "message": "I will reach home in 10 minutes. Please wait for me."},
    {"language": "English", "expected": "SPAM",
     "message": "Congratulations! You have won a cash prize. Call now to claim."},
    {"language": "English", "expected": "SPAM",
     "message": "URGENT! Your mobile number has won a guaranteed reward. Reply WIN now."},

    # Hindi
    {"language": "Hindi", "expected": "HAM",
     "message": "क्या तुम आज कॉलेज आ रहे हो? मैं तुम्हारे लिए सीट रख दूँगा।"},
    {"language": "Hindi", "expected": "HAM",
     "message": "मैं दस मिनट में घर पहुँच जाऊँगा। कृपया मेरा इंतज़ार करना।"},
    {"language": "Hindi", "expected": "SPAM",
     "message": "बधाई हो! आपका मोबाइल नंबर नकद इनाम के लिए चुना गया है। दावा करने के लिए अभी कॉल करें।"},
    {"language": "Hindi", "expected": "SPAM",
     "message": "तुरंत जवाब दें! आपने बड़ा पुरस्कार जीता है। इनाम पाने के लिए इस नंबर पर संपर्क करें।"},

    # Hinglish
    {"language": "Hinglish", "expected": "HAM",
     "message": "Kya tum aaj college aa rahe ho? Main tumhare liye seat rakh dunga."},
    {"language": "Hinglish", "expected": "HAM",
     "message": "Main 10 minute mein ghar pahunch jaunga. Please mera wait karna."},
    {"language": "Hinglish", "expected": "SPAM",
     "message": "Mubarak ho! Aapka mobile number cash prize ke liye select hua hai. Claim karne ke liye abhi call karein."},
    {"language": "Hinglish", "expected": "SPAM",
     "message": "Urgent! Aapne bada reward jeeta hai. Prize lene ke liye WIN reply karein."},
]


def class_to_label(value):
    value = str(value).strip().lower()

    if value in {"spam", "1", "1.0", "true"}:
        return "SPAM"

    if value in {"ham", "normal", "not spam", "0", "0.0", "false"}:
        return "HAM"

    return str(value).upper()


def load_models():
    classical_model = None
    bilstm_model = None

    print("\nLoading existing models...")

    if TFIDF_PATH.exists():
        try:
            classical_model = joblib.load(TFIDF_PATH)
            print("TF-IDF Logistic Regression: LOADED")
            print("Class labels:", classical_model.classes_)
        except Exception as error:
            print("TF-IDF model loading failed:", error)
    else:
        print("TF-IDF model file not found:", TFIDF_PATH)

    if BILSTM_PATH.exists():
        try:
            bilstm_model = tf.keras.models.load_model(
                BILSTM_PATH, compile=False
            )
            print("BiLSTM model: LOADED")
        except Exception as error:
            print("BiLSTM model loading failed:", error)
    else:
        print("BiLSTM model file not found:", BILSTM_PATH)

    return classical_model, bilstm_model


def predict_classical(model, message):
    predicted_value = model.predict([message])[0]
    predicted_label = class_to_label(predicted_value)

    spam_score = None

    if hasattr(model, "predict_proba") and hasattr(model, "classes_"):
        probabilities = model.predict_proba([message])[0]

        spam_indices = [
            index
            for index, value in enumerate(model.classes_)
            if class_to_label(value) == "SPAM"
        ]

        if spam_indices:
            spam_score = float(probabilities[spam_indices[0]])

    return predicted_label, spam_score


def main():
    classical_model, bilstm_model = load_models()

    if classical_model is None and bilstm_model is None:
        print("\nNo models could be loaded. Test stopped.")
        return

    messages = [item["message"] for item in TEST_CASES]

    bilstm_scores = None

    if bilstm_model is not None:
        try:
            predictions = bilstm_model.predict(
                tf.constant(messages, dtype=tf.string),
                verbose=0
            )
            bilstm_scores = np.asarray(predictions).reshape(-1)
        except Exception as error:
            print("\nBiLSTM prediction failed:", error)

    results = []

    print("\n" + "=" * 125)
    print("MULTILINGUAL SMS MODEL TEST")
    print("=" * 125)

    for index, item in enumerate(TEST_CASES):
        expected = item["expected"]
        language = item["language"]
        message = item["message"]

        tfidf_label = "NOT TESTED"
        tfidf_score = None
        bilstm_label = "NOT TESTED"
        bilstm_score = None

        if classical_model is not None:
            try:
                tfidf_label, tfidf_score = predict_classical(
                    classical_model, message
                )
            except Exception as error:
                tfidf_label = "ERROR"
                print("\nTF-IDF prediction error:", error)

        if bilstm_scores is not None:
            bilstm_score = float(bilstm_scores[index])
            bilstm_label = (
                "SPAM" if bilstm_score >= THRESHOLD else "HAM"
            )

        tfidf_match = (
            "YES" if tfidf_label == expected else "NO"
        ) if tfidf_label not in {"NOT TESTED", "ERROR"} else "N/A"

        bilstm_match = (
            "YES" if bilstm_label == expected else "NO"
        ) if bilstm_label != "NOT TESTED" else "N/A"

        results.append({
            "language": language,
            "expected": expected,
            "tfidf_match": tfidf_match,
            "bilstm_match": bilstm_match
        })

        print(f"\nTest {index + 1} | {language}")
        print("Message:       ", message)
        print("Expected:      ", expected)

        if tfidf_score is not None:
            print(
                f"TF-IDF:         {tfidf_label} "
                f"(spam score: {tfidf_score:.2%}) | Match: {tfidf_match}"
            )
        else:
            print(
                f"TF-IDF:         {tfidf_label} | Match: {tfidf_match}"
            )

        if bilstm_score is not None:
            print(
                f"BiLSTM:         {bilstm_label} "
                f"(spam score: {bilstm_score:.2%}) | Match: {bilstm_match}"
            )
        else:
            print(
                f"BiLSTM:         {bilstm_label} | Match: {bilstm_match}"
            )

    print("\n" + "=" * 80)
    print("SAMPLE CHECK SUMMARY")
    print("=" * 80)

    for language in ["English", "Hindi", "Hinglish"]:
        language_results = [
            result for result in results
            if result["language"] == language
        ]

        print(f"\n{language}: {len(language_results)} test messages")

        for model_name, match_key, model in [
            ("TF-IDF", "tfidf_match", classical_model),
            ("BiLSTM", "bilstm_match", bilstm_model)
        ]:
            if model is None:
                continue

            checked = [
                result for result in language_results
                if result[match_key] in {"YES", "NO"}
            ]

            correct = sum(
                result[match_key] == "YES" for result in checked
            )

            print(
                f"  {model_name}: {correct}/{len(checked)} "
                "sample predictions matched"
            )

    print("\nIMPORTANT:")
    print("This is a small preliminary test, not a reliable accuracy benchmark.")
    print("No model was retrained or modified.")
    print("A correct prediction on these examples does not prove multilingual support.")


if __name__ == "__main__":
    main()
