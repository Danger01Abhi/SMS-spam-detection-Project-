from src.risk_assessment import assess_security, display_assessment

from pathlib import Path
import joblib
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "advanced"
    / "tuned_logistic_regression.joblib"
)
import re
import ipaddress
from urllib.parse import urlsplit


# Find URLs inside SMS messages

URL_PATTERN = re.compile(
    r"""(?ix)
    (?<![@\w])
    (?:
        https?://
        |
        www\.
    )
    [a-z0-9.-]+
    (?::\d{1,5})?
    (?:/[^\s<>"']*)?
    |
    (?<![@\w])
    (?:[a-z0-9-]+\.)+
    [a-z]{2,}
    (?::\d{1,5})?
    (?:/[^\s<>"']*)?
    """
)


# Common URL-shortening services
SHORTENERS = {
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "goo.gl",
    "is.gd",
    "buff.ly",
    "ow.ly",
    "rebrand.ly",
    "cutt.ly",
    "shorturl.at",
}

# Words sometimes found in suspicious URL paths
SUSPICIOUS_WORDS = {
    "login",
    "verify",
    "password",
    "account",
    "wallet",
    "banking",
    "update",
    "gift",
    "prize",
    "claim",
    "free",
}



def extract_urls(message):
    """Extract URLs from SMS text without visiting them."""

    matches = URL_PATTERN.findall(message)

    urls = []

    for url in matches:
        url = url.strip().rstrip(".,!?;:)]}")

        if url and url not in urls:
            urls.append(url)

    return urls



def analyse_url(url):
    """Check basic suspicious characteristics of one URL."""

    # Add a temporary scheme for parsing URLs without http/https
    if "://" in url:
        parsed = urlsplit(url)
    else:
        parsed = urlsplit("http://" + url)

    hostname = parsed.hostname

    if not hostname:
        return {
            "url": url,
            "score": 0,
            "risk": "UNABLE TO ANALYSE",
            "indicators": ["Could not identify the hostname."],
        }

    hostname = hostname.lower()

    score = 0
    indicators = []

    # 1. Check whether the hostname is an IP address
    try:
        ipaddress.ip_address(hostname)

        score += 3
        indicators.append(
            "URL uses an IP address instead of a domain name."
        )

    except ValueError:
        pass

    # 2. Check for known URL shorteners
    is_shortener = any(
        hostname == domain or hostname.endswith("." + domain)
        for domain in SHORTENERS
    )

    if is_shortener:
        score += 2
        indicators.append(
            "URL uses a known shortening service."
        )

    # 3. Check HTTPS presence
    explicitly_uses_https = url.lower().startswith("https://")

    if not explicitly_uses_https:
        score += 2
        indicators.append(
            "HTTPS is not explicitly specified in the URL."
        )

    # 4. Check URL length
    if len(url) > 100:
        score += 2
        indicators.append(
            "URL is unusually long (over 100 characters)."
        )

    # 5. Check for credentials or @ symbol in the URL authority
    if "@" in parsed.netloc:
        score += 3
        indicators.append(
            "URL contains an @ symbol in its authority section."
        )

    # 6. Check for punycode
    if "xn--" in hostname:
        score += 2
        indicators.append(
            "Hostname contains punycode encoding."
        )

    # 7. Check suspicious words in the path or query
    path_and_query = (
        parsed.path + " " + parsed.query
    ).lower()

    found_words = [
        word for word in SUSPICIOUS_WORDS
        if re.search(
            r"\b" + re.escape(word) + r"\b",
            path_and_query
        )
    ]

    if found_words:
        score += 1
        indicators.append(
            "Suspicious URL keywords: "
            + ", ".join(sorted(found_words))
        )

    # Assign a heuristic risk level
    if score >= 6:
        risk = "HIGH"
    elif score >= 3:
        risk = "MODERATE"
    else:
        risk = "LOW"

    if not indicators:
        indicators.append(
            "No suspicious indicators were found by these checks."
        )

    return {
        "url": url,
        "hostname": hostname,
        "score": score,
        "risk": risk,
        "indicators": indicators,
    }


def scan_sms(message):
    """Scan all URLs found in an SMS message."""

    urls = extract_urls(message)

    if not urls:
        return []

    return [
        analyse_url(url)
        for url in urls
    ]



def main():
    print("\n====================================")
    print("       AI SMS SECURITY ANALYSER")
    print("====================================")

    message = input("\nEnter an SMS message: ").strip()

    if not message:
        print("Please enter a message.")
        return

    # PART 1: SMS SPAM/HAM CLASSIFICATION
    if not MODEL_PATH.exists():
        print(f"\nERROR: Trained model not found: {MODEL_PATH}")
        return

    model = joblib.load(MODEL_PATH)

    prediction = model.predict([message])[0]
    probabilities = model.predict_proba([message])[0]

    raw_label = str(prediction).strip().lower()

    if raw_label == "spam":
        sms_result = "SPAM"
    elif raw_label == "ham":
        sms_result = "HAM"
    elif raw_label in ("1", "1.0"):
        sms_result = "SPAM"
    elif raw_label in ("0", "0.0"):
        sms_result = "HAM"
    else:
        sms_result = f"UNKNOWN CLASS ({prediction})"

    confidence = float(max(probabilities)) * 100

    print("\n========== SMS CLASSIFICATION ==========")
    print(f"SMS RESULT: {sms_result}")
    print(f"MODEL CONFIDENCE: {confidence:.2f}%")
    print("========================================")

    # PART 2: URL SECURITY ANALYSIS
    results = scan_sms(message)

    print("\n========== URL SECURITY ANALYSIS ==========")

    if not results:
        print("No URLs detected in this SMS.")
        print("URL security risk could not be assessed.")
    else:
        print(f"URLs detected: {len(results)}")

        for number, result in enumerate(results, start=1):
            print(f"\nURL {number}: {result['url']}")
            print(f"Hostname: {result.get('hostname', 'Unknown')}")
            print(f"Risk score: {result['score']}")
            print(f"Risk level: {result['risk']}")

            print("Indicators:")

            for indicator in result["indicators"]:
                print(f" - {indicator}")

    # PART 3: COMBINED SECURITY ASSESSMENT
    assessment = assess_security(sms_result, results)

    display_assessment(assessment)

if __name__ == "__main__":
    main()


