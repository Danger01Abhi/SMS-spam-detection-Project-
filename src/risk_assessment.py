
def assess_security(sms_prediction, url_results):
    """
    Combine SMS classification and URL findings
    into a rule-based security review priority.

    This is not a calibrated probability of a threat.
    """

    sms_label = str(sms_prediction).strip().upper()

    url_levels = [
        str(result.get("risk", "")).strip().upper()
        for result in url_results
    ]

    reasons = []

    # Identify the highest URL risk
    if "HIGH" in url_levels:
        highest_url_risk = "HIGH"
    elif "MODERATE" in url_levels or "UNABLE TO ANALYSE" in url_levels:
        highest_url_risk = "MODERATE"
    elif url_levels:
        highest_url_risk = "LOW"
    else:
        highest_url_risk = "NO URL"

    # Record SMS classification
    if sms_label == "SPAM":
        reasons.append(
            "The SMS classification model predicted SPAM."
        )
    elif sms_label == "HAM":
        reasons.append(
            "The SMS classification model predicted HAM. "
            "This does not guarantee that the message is safe."
        )
    else:
        reasons.append(
            "SMS classification is unknown. Manual review is recommended."
        )

    # Record URL findings
    if highest_url_risk == "HIGH":
        reasons.append(
            "At least one URL has a HIGH heuristic risk level."
        )
    elif highest_url_risk == "MODERATE":
        reasons.append(
            "At least one URL has MODERATE or unverified risk indicators."
        )
    elif highest_url_risk == "LOW":
        reasons.append(
            "No high or moderate URL indicators were found."
        )
    else:
        reasons.append(
            "No URL was detected, so URL-based risk was not assessed."
        )

    # Determine combined review priority
    if highest_url_risk == "HIGH":
        priority = "HIGH"

    elif sms_label == "SPAM" and highest_url_risk == "MODERATE":
        priority = "HIGH"

    elif highest_url_risk == "MODERATE":
        priority = "MODERATE"

    elif sms_label == "SPAM":
        priority = "MODERATE"

    elif sms_label == "HAM" and highest_url_risk in ("LOW", "NO URL"):
        priority = "LOW"

    else:
        priority = "REVIEW REQUIRED"

    return {
        "sms_prediction": sms_label,
        "highest_url_risk": highest_url_risk,
        "review_priority": priority,
        "reasons": reasons
    }


def display_assessment(result):
    print("\n========== COMBINED SECURITY ASSESSMENT ==========")

    print(
        f"SMS Classification: {result['sms_prediction']}"
    )

    print(
        f"Highest URL Risk: {result['highest_url_risk']}"
    )

    print(
        f"Combined Review Priority: {result['review_priority']}"
    )

    print("\nReasons:")

    for reason in result["reasons"]:
        print(f"- {reason}")

    print("==================================================")
