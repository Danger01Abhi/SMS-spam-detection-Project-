from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

from src.bulk_scanner import analyze_bulk_messages
from src.url_scanner import scan_sms
from src.risk_assessment import assess_security

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "advanced" / "tuned_logistic_regression.joblib"
REPORTS_DIR = PROJECT_ROOT / "reports" / "advanced"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def readable_label(label):
    value = str(label).strip().lower()
    if value in ("spam", "1", "1.0"):
        return "SPAM"
    if value in ("ham", "0", "0.0"):
        return "HAM"
    return f"UNKNOWN ({label})"


st.set_page_config(
    page_title="AI SMS Security Analyzer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp {
  background: radial-gradient(circle at 8% 0%, rgba(59,130,246,.12), transparent 28%),
              radial-gradient(circle at 90% 5%, rgba(20,184,166,.10), transparent 25%), #0b1220;
  color: #e5edf8;
}
[data-testid="stHeader"] { background: rgba(11,18,32,.75); }
[data-testid="stSidebar"] {
  background: linear-gradient(180deg,#101b2e,#0b1424);
  border-right: 1px solid rgba(148,163,184,.15);
}
.hero {
  padding: 30px; border-radius: 22px;
  background: linear-gradient(115deg,#172b4d,#123b50 58%,#164e63);
  border: 1px solid rgba(148,207,255,.22);
  box-shadow: 0 18px 50px rgba(0,0,0,.22); margin: 8px 0 24px;
}
.kicker { color:#7dd3fc; font-size:12px; font-weight:800; letter-spacing:2px; }
.hero h1 { color:#f8fafc; font-size:clamp(28px,4vw,42px); margin:8px 0; font-weight:800; }
.hero p { color:#c7d9eb; margin:0; font-size:15px; }
.section-heading { color:#f1f5f9; font-size:22px; font-weight:800; margin:6px 0 4px; }
.muted { color:#94a3b8; font-size:13px; margin-bottom:18px; }
.mini-card {
  background:linear-gradient(145deg,rgba(30,41,59,.95),rgba(15,30,49,.95));
  border:1px solid rgba(148,163,184,.17); border-radius:16px;
  padding:18px; min-height:105px;
}
.mini-label { color:#94a3b8; font-size:12px; font-weight:700; letter-spacing:.8px; text-transform:uppercase; }
.mini-value { color:#f8fafc; font-size:23px; font-weight:800; margin-top:8px; }
div[data-testid="stMetric"] {
  background:linear-gradient(145deg,rgba(30,41,59,.95),rgba(15,30,49,.95));
  border:1px solid rgba(148,163,184,.18); padding:16px 18px; border-radius:16px;
}
div[data-testid="stMetricLabel"] { color:#a8b8cc; }
div[data-testid="stMetricValue"] { color:#f8fafc; }
.stTabs [data-baseweb="tab-list"] {
  gap:8px; background:rgba(15,23,42,.75); padding:8px;
  border-radius:14px; border:1px solid rgba(148,163,184,.13);
}
.stTabs [data-baseweb="tab"] { height:45px; border-radius:10px; color:#b8c7d9; font-weight:700; padding:0 15px; }
.stTabs [aria-selected="true"] { background:linear-gradient(90deg,#2563eb,#0f9f9a); color:white !important; }
.stButton > button, .stDownloadButton > button {
  border:0; border-radius:11px; font-weight:800; padding:10px 20px;
  background:linear-gradient(90deg,#2563eb,#0891b2); color:white;
}
.stButton > button:hover, .stDownloadButton > button:hover { color:white; filter:brightness(1.12); border:0; }
.stTextArea textarea { background:#111d30; color:#f8fafc; border:1px solid #334155; border-radius:12px; }
[data-testid="stDataFrame"] { border:1px solid rgba(148,163,184,.18); border-radius:12px; overflow:hidden; }
hr { border-color:rgba(148,163,184,.18); }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🛡️ SMS Shield")
    st.caption("AI-powered message security")
    st.divider()
    st.markdown("**DASHBOARD MODULES**")
    st.markdown("🔎 SMS classification")
    st.markdown("📂 Bulk message analysis")
    st.markdown("📊 Model evaluation")
    st.markdown("ℹ️ Project details")
    st.divider()
    st.markdown("**ACTIVE CLASSIFIER**")
    st.success("Tuned Logistic Regression")
    st.caption("Advanced model · Local analysis")
    st.divider()
    st.caption("Educational project")

st.markdown("""
<div class="hero">
  <div class="kicker">AI • MACHINE LEARNING • CYBERSECURITY</div>
  <h1>SMS Security Analyzer</h1>
  <p>Detect suspicious messages, inspect embedded URLs, and review combined security signals.</p>
</div>
""", unsafe_allow_html=True)

tab_sms, tab_bulk, tab_models, tab_info = st.tabs([
    "🔎  SMS Analyzer", "📂  Bulk Scanner", "📊  Model Insights", "ℹ️  Project"
])

with tab_sms:
    st.markdown('<div class="section-heading">Analyze a message</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">Paste an SMS to classify it and inspect any URLs it contains.</div>', unsafe_allow_html=True)
    left, right = st.columns([1.55, 1], gap="large")
    with left:
        message = st.text_area(
            "SMS message",
            placeholder="Example: Congratulations! You won a prize. Click here to claim...",
            height=180,
            label_visibility="collapsed",
        )
        analyze_button = st.button("🔍  Analyze message", type="primary", use_container_width=True)
    with right:
        st.markdown("""
        <div class="mini-card">
          <div class="mini-label">What this tool checks</div>
          <div class="mini-value">3 security layers</div>
          <p style="color:#a8b8cc;font-size:13px;margin:8px 0 0">
            Message classification · URL indicators · Combined review priority
          </p>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        st.info("Never click a suspicious link just to test it. This dashboard uses local URL checks.")

    if analyze_button:
        if not message.strip():
            st.warning("Please enter an SMS message.")
        else:
            try:
                model = load_model()
                probabilities = model.predict_proba([message])[0]
                classes = model.classes_
                predicted_index = int(probabilities.argmax())
                sms_result = readable_label(classes[predicted_index])
                confidence = float(probabilities[predicted_index]) * 100
                url_results = scan_sms(message)

                st.divider()
                st.markdown('<div class="section-heading">Analysis results</div>', unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                c1.metric("SMS prediction", sms_result)
                c2.metric("Model confidence", f"{confidence:.2f}%")
                c3.metric("URLs detected", len(url_results))

                if sms_result == "SPAM":
                    st.error("⚠️ The model classified this message as SPAM. Treat it cautiously.")
                elif sms_result == "HAM":
                    st.success("✓ The model classified this message as HAM. This does not guarantee it is harmless.")
                else:
                    st.warning("The model returned an unrecognized class label.")

                st.markdown("#### 🌐 URL security inspection")
                if not url_results:
                    st.info("No URL detected. URL-based security assessment is unavailable.")
                else:
                    for i, result in enumerate(url_results, start=1):
                        with st.expander(f"URL {i} · {result['url']}", expanded=True):
                            a, b = st.columns(2)
                            a.write(f"**Hostname:** {result.get('hostname', 'Unknown')}")
                            b.write(f"**Risk score:** {result['score']}")
                            st.write(f"**Risk level:** {result['risk']}")
                            st.write("**Detected indicators**")
                            for indicator in result["indicators"]:
                                st.write(f"• {indicator}")

                st.markdown("#### 🧭 Combined security assessment")
                assessment = assess_security(sms_result, url_results)
                priority = assessment["review_priority"]
                p1, p2, p3 = st.columns(3)
                p1.metric("SMS classification", assessment["sms_prediction"])
                p2.metric("Highest URL risk", assessment["highest_url_risk"])
                p3.metric("Review priority", priority)
                if priority == "HIGH":
                    st.error(f"🚨 High review priority — {priority}")
                elif priority == "MODERATE":
                    st.warning(f"⚠️ Moderate review priority — {priority}")
                elif priority == "LOW":
                    st.success(f"✓ Low review priority — {priority}")
                else:
                    st.info(f"Review priority: {priority}")
                with st.expander("View assessment reasons"):
                    for reason in assessment["reasons"]:
                        st.write(f"• {reason}")
                st.caption(
                    "Model confidence and URL risk are separate measurements. Combined review priority "
                    "is rule-based, not a calibrated threat probability."
                )
            except Exception as error:
                st.error(f"Analysis failed: {error}")

with tab_bulk:
    st.markdown('<div class="section-heading">Bulk SMS scanner</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">Upload a CSV containing a column named message to analyze multiple messages.</div>', unsafe_allow_html=True)
    st.info("Uploaded messages are not automatically saved by this dashboard. Use sample or test messages.")
    if "bulk_results" not in st.session_state:
        st.session_state["bulk_results"] = None
    uploaded_file = st.file_uploader("Choose SMS CSV file", type=["csv"], key="bulk_sms_csv")
    if uploaded_file is not None:
        signature = (uploaded_file.name, uploaded_file.size)
        if st.session_state.get("bulk_file_signature") != signature:
            st.session_state["bulk_file_signature"] = signature
            st.session_state["bulk_results"] = None
        try:
            input_df = pd.read_csv(uploaded_file)
            st.markdown("#### File preview")
            st.dataframe(input_df.head(5), use_container_width=True)
            st.caption(f"{len(input_df):,} messages found in the uploaded file.")
            if "message" not in input_df.columns:
                st.error("Your CSV must contain a column named 'message'.")
            elif st.button("⚡  Analyze all messages", type="primary", key="analyze_bulk"):
                with st.spinner("Analyzing messages. Please wait..."):
                    results_df = analyze_bulk_messages(input_df, load_model())
                    st.session_state["bulk_results"] = results_df
                st.success("Bulk analysis completed!")
            if st.session_state["bulk_results"] is not None:
                results_df = st.session_state["bulk_results"]
                st.markdown("#### Bulk analysis summary")
                st.metric("Messages processed", f"{len(results_df):,}")
                st.dataframe(results_df, use_container_width=True)
                csv_bytes = results_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "⬇️  Download results CSV", data=csv_bytes,
                    file_name="sms_bulk_analysis_results.csv", mime="text/csv",
                    key="download_bulk_results",
                )
        except Exception as error:
            st.error(f"Could not process the CSV file: {error}")
    else:
        st.markdown("""
        <div class="mini-card">
          <div class="mini-label">Supported input</div>
          <div class="mini-value">.CSV files</div>
          <p style="color:#a8b8cc;font-size:13px;margin:8px 0 0">Required column: <code>message</code></p>
        </div>
        """, unsafe_allow_html=True)

with tab_models:
    st.markdown('<div class="section-heading">Model performance & evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">Visual reports generated during model training and evaluation.</div>', unsafe_allow_html=True)
    comparison_image = REPORTS_DIR / "model_comparison.png"
    if comparison_image.exists():
        st.image(str(comparison_image), caption="Comparison of trained classification models", use_container_width=True)
    else:
        st.warning(f"Model comparison graph was not found at: {comparison_image}")
    graphs = [
        ("tuned_confusion_matrix.png", "Confusion matrix"),
        ("roc_curve.png", "ROC curve"),
        ("precision_recall_curve.png", "Precision–recall curve"),
    ]
    available = [(filename, title) for filename, title in graphs if (REPORTS_DIR / filename).exists()]
    if available:
        st.markdown("#### Additional evaluation graphs")
        for filename, title in available:
            with st.expander(title, expanded=True):
                st.image(str(REPORTS_DIR / filename), use_container_width=True)
    else:
        st.info("No additional evaluation graphs were found in reports/advanced.")

with tab_info:
    st.markdown('<div class="section-heading">About this project</div>', unsafe_allow_html=True)
    st.markdown('<div class="muted">An educational machine-learning project combining SMS classification and URL risk indicators.</div>', unsafe_allow_html=True)
    features = [
        ("💬", "SMS classification", "Predicts whether a message is HAM or SPAM."),
        ("🔗", "URL inspection", "Checks URLs for suspicious characteristics."),
        ("🧠", "Model evaluation", "Compares trained machine-learning models."),
        ("📦", "Bulk scanning", "Processes messages from a CSV file."),
        ("🔍", "Explainable AI", "LIME explanation is available through a separate script."),
        ("🛡️", "Risk assessment", "Combines SMS and URL signals into a review priority."),
    ]
    cols = st.columns(3)
    for i, (emoji, title, description) in enumerate(features):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="mini-card" style="margin-bottom:14px">
              <div style="font-size:25px">{emoji}</div>
              <div style="color:#f8fafc;font-weight:800;margin-top:8px">{title}</div>
              <div style="color:#a8b8cc;font-size:13px;margin-top:5px">{description}</div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("#### Security notes")
    st.warning(
        "This application does not open or visit URLs. URL risk indicators are heuristic checks "
        "and cannot establish whether a website is safe."
    )
    st.info(
        "A HAM prediction does not guarantee that a message is harmless. Always verify unexpected "
        "requests, payment demands, and links independently."
    )

st.divider()
footer_left, footer_right = st.columns([2, 1])
with footer_left:
    st.caption("🛡️ AI SMS Security Analyzer · Student Machine Learning Project")
with footer_right:
    st.caption("Local analysis · Educational use")
