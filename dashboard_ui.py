import streamlit as st
import pandas as pd

from src.bulk_scanner import analyze_bulk_messages


import pandas as pd
from src.url_scanner import scan_sms
from src.risk_assessment import assess_security
from pathlib import Path

import joblib
import streamlit as st

from src.url_scanner import scan_sms
from src.risk_assessment import assess_security


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "advanced"
    / "tuned_logistic_regression.joblib"
)

REPORTS_DIR = PROJECT_ROOT / "reports" / "advanced"


# Load model once and reuse it
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def readable_label(label):
    """Convert model labels into HAM or SPAM."""

    value = str(label).strip().lower()

    if value == "spam" or value in ("1", "1.0"):
        return "SPAM"

    if value == "ham" or value in ("0", "0.0"):
        return "HAM"

    return f"UNKNOWN ({label})"


# Streamlit page configuration
st.set_page_config(
    page_title="AI SMS Security Analyzer",
    page_icon="🛡️",
    layout="wide"
)


# Dashboard heading
st.title("🛡️ AI SMS Security Analyzer")

st.caption(
    "Machine Learning SMS Classification and URL Risk Assessment"
)

st.divider()


# Dashboard tabs
tab1, tab_bulk, tab2, tab3 = st.tabs([
    "SMS Analyzer",
    "Bulk SMS Scanner",
    "Model Comparison",
    "Project Information"
])


# TAB 1: SMS ANALYZER
with tab1:

    st.header("Analyze an SMS Message")

    message = st.text_area(
        "Enter your SMS message:",
        placeholder=(
            "Example: Congratulations! You won a prize. "
            "Click here to claim..."
        ),
        height=150
    )

    analyze_button = st.button(
        "Analyze SMS",
        type="primary"
    )

    if analyze_button:

        if not message.strip():
            st.warning("Please enter an SMS message.")

        else:

            try:
                model = load_model()

                # SMS classification
                probabilities = model.predict_proba([message])[0]

                classes = model.classes_

                predicted_index = int(
                    probabilities.argmax()
                )

                prediction = classes[predicted_index]

                sms_result = readable_label(prediction)

                confidence = float(
                    probabilities[predicted_index]
                ) * 100

                # Display classification
                st.subheader("SMS Classification")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Predicted Result",
                        sms_result
                    )

                with col2:
                    st.metric(
                        "Model Confidence",
                        f"{confidence:.2f}%"
                    )

                if sms_result == "SPAM":
                    st.error("The model classified this message as SPAM.")

                elif sms_result == "HAM":
                    st.success("The model classified this message as HAM.")

                else:
                    st.warning(
                        "The model returned an unrecognized class label."
                    )

                st.divider()

                # URL security analysis
                st.subheader("URL Security Analysis")

                url_results = scan_sms(message)

                if not url_results:
                    st.info(
                        "No URL detected. URL-based security "
                        "assessment is unavailable."
                    )

                else:
                    for index, result in enumerate(
                        url_results,
                        start=1
                    ):

                        with st.expander(
                            f"URL {index}: {result['url']}",
                            expanded=True
                        ):

                            st.write(
                                f"**Hostname:** "
                                f"{result.get('hostname', 'Unknown')}"
                            )

                            st.write(
                                f"**Risk score:** {result['score']}"
                            )

                            st.write(
                                f"**Risk level:** {result['risk']}"
                            )

                            st.write("**Detected indicators:**")

                            for indicator in result["indicators"]:
                                st.write(f"- {indicator}")

                st.divider()

                # Combined risk assessment
                st.subheader("Combined Security Assessment")

                assessment = assess_security(
                    sms_result,
                    url_results
                )

                st.write(
                    f"**SMS Classification:** "
                    f"{assessment['sms_prediction']}"
                )

                st.write(
                    f"**Highest URL Risk:** "
                    f"{assessment['highest_url_risk']}"
                )

                priority = assessment["review_priority"]

                st.write(
                    f"**Combined Review Priority:** {priority}"
                )

                if priority == "HIGH":
                    st.error(f"Review Priority: {priority}")

                elif priority == "MODERATE":
                    st.warning(f"Review Priority: {priority}")

                elif priority == "LOW":
                    st.success(f"Review Priority: {priority}")

                else:
                    st.warning(f"Review Priority: {priority}")

                st.write("**Assessment reasons:**")

                for reason in assessment["reasons"]:
                    st.write(f"- {reason}")

                st.caption(
                    "Model confidence and URL risk are separate "
                    "measurements. Combined review priority is "
                    "rule-based, not a calibrated threat probability."
                )

            except Exception as error:
                st.error(f"Analysis failed: {error}")

with tab_bulk:
    st.header("Bulk SMS Scanner")

    st.write(
        "Upload a CSV file containing a column named 'message'. "
        "The system will classify each SMS and check detected URLs."
    )

    st.info(
        "Use sample or test messages. Uploaded messages are not "
        "automatically saved by this dashboard."
    )

    if "bulk_results" not in st.session_state:
        st.session_state["bulk_results"] = None

    uploaded_file = st.file_uploader(
        "Upload your SMS CSV file",
        type=["csv"],
        key="bulk_sms_csv"
    )

    if uploaded_file is not None:

        file_signature = (uploaded_file.name, uploaded_file.size)

        if st.session_state.get("bulk_file_signature") != file_signature:
            st.session_state["bulk_file_signature"] = file_signature
            st.session_state["bulk_results"] = None

        try:
            input_df = pd.read_csv(uploaded_file)

            st.subheader("CSV Preview")
            st.dataframe(input_df.head(5), use_container_width=True)

            st.write(f"Total messages in file: {len(input_df)}")

            if "message" not in input_df.columns:
                st.error(
                    "Your CSV must contain a column named 'message'."
                )

            elif st.button(
                "Analyze All Messages",
                type="primary",
                key="analyze_bulk"
            ):
                with st.spinner("Analyzing messages. Please wait..."):
                    model = load_model()
                    results_df = analyze_bulk_messages(input_df, model)

                    st.session_state["bulk_results"] = results_df

                st.success("Bulk analysis completed!")

            if st.session_state["bulk_results"] is not None:
                results_df = st.session_state["bulk_results"]

                st.subheader("Bulk Analysis Results")

                st.metric("Messages processed", len(results_df))

                st.dataframe(
                    results_df,
                    use_container_width=True
                )

                csv_bytes = results_df.to_csv(
                    index=False
                ).encode("utf-8-sig")

                st.download_button(
                    label="Download Results CSV",
                    data=csv_bytes,
                    file_name="sms_bulk_analysis_results.csv",
                    mime="text/csv",
                    key="download_bulk_results"
                )

        except Exception as error:
            st.error(f"Could not process the CSV file: {error}")


# TAB 2: MODEL COMPARISON
with tab2:

    st.header("Machine Learning Model Comparison")

    comparison_image = (
        REPORTS_DIR / "model_comparison.png"
    )

    if comparison_image.exists():

        st.image(
            str(comparison_image),
            caption="Comparison of trained classification models",
            use_container_width=True
        )

    else:
        st.warning(
            "Model comparison graph was not found."
        )

    st.subheader("Additional Evaluation Graphs")

    graph_names = [
        "tuned_confusion_matrix.png",
        "roc_curve.png",
        "precision_recall_curve.png"
    ]

    for graph_name in graph_names:

        graph_path = REPORTS_DIR / graph_name

        if graph_path.exists():

            st.write(graph_name)

            st.image(
                str(graph_path),
                use_container_width=True
            )


# TAB 3: PROJECT INFORMATION
with tab3:

    st.header("About This Project")

    st.write(
        "AI SMS Security Analyzer is a machine learning "
        "project that classifies SMS messages and checks "
        "URLs for suspicious characteristics."
    )

    st.subheader("Main Features")

    st.write("- SMS Spam and Ham Classification")
    st.write("- Logistic Regression Prediction")
    st.write("- Model Confidence Display")
    st.write("- LIME Explainable AI (separate script)")
    st.write("- URL Security Scanner")
    st.write("- Combined Security Risk Assessment")
    st.write("- Machine Learning Model Comparison")

    st.subheader("Important Security Information")

    st.write(
        "This application does not open or visit URLs. "
        "URL risk indicators are heuristic checks and "
        "cannot establish whether a website is safe."
    )

    st.write(
        "A HAM prediction does not guarantee that a message "
        "is harmless."
    )

st.divider()

st.caption("AI SMS Security Analyzer | Student Machine Learning Project")
