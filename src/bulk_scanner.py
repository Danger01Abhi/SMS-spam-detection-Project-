import pandas as pd

from src.url_scanner import scan_sms
from src.risk_assessment import assess_security


def readable_label(label):
    """Convert model labels into readable HAM/SPAM labels."""

    value = str(label).strip().lower()

    if value == "spam" or value in ("1", "1.0"):
        return "SPAM"

    if value == "ham" or value in ("0", "0.0"):
        return "HAM"

    return f"UNKNOWN ({label})"


def analyze_bulk_messages(dataframe, model):
    """
    Classify SMS messages and assess URL security
    for every row in a CSV DataFrame.
    """

    if "message" not in dataframe.columns:
        raise ValueError(
            "CSV must contain a column named 'message'."
        )

    results = []

    for index, row in dataframe.iterrows():

        message = str(row["message"]).strip()

        if not message or message.lower() == "nan":
            results.append({
                "message": message,
                "sms_prediction": "NOT ANALYSED",
                "confidence": 0.0,
                "url_count": 0,
                "url_risk": "NOT ANALYSED",
                "review_priority": "REVIEW REQUIRED",
                "detected_urls": ""
            })
            continue

        # SMS classification
        probabilities = model.predict_proba([message])[0]

        predicted_index = int(probabilities.argmax())

        prediction = model.classes_[predicted_index]

        sms_result = readable_label(prediction)

        confidence = float(
            probabilities[predicted_index]
        ) * 100

        # URL security analysis
        url_results = scan_sms(message)

        # Combined assessment
        assessment = assess_security(
            sms_result,
            url_results
        )

        detected_urls = ", ".join(
            result["url"] for result in url_results
        )

        results.append({
            "message": message,
            "sms_prediction": sms_result,
            "confidence": round(confidence, 2),
            "url_count": len(url_results),
            "url_risk": assessment["highest_url_risk"],
            "review_priority": assessment["review_priority"],
            "detected_urls": detected_urls
        })

    return pd.DataFrame(results)
