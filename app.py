from pathlib import Path
from datetime import datetime

import streamlit as st
import tensorflow as tf
import pandas as pd


# ============================================================
# SMS SPAM DETECTION — POLISHED STREAMLIT DASHBOARD
# ============================================================

st.set_page_config(
    page_title="SMS Shield | Spam Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "sms_spam_bilstm.keras"


# ------------------------- Styling ---------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --ink: #eaf1ff;
        --muted: #a7b6d2;
        --panel: rgba(18, 31, 55, 0.78);
        --line: rgba(157, 181, 222, 0.18);
        --blue: #76a9ff;
        --cyan: #62e6d5;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        color: var(--ink);
        background:
            radial-gradient(ellipse at 5% 0%, rgba(38, 91, 180, .28), transparent 35%),
            radial-gradient(ellipse at 95% 8%, rgba(20, 184, 166, .13), transparent 28%),
            linear-gradient(145deg, #07101f 0%, #0b1425 50%, #101b30 100%);
    }

    [data-testid="stHeader"] {
        background: rgba(7, 16, 31, .45);
    }

    [data-testid="stSidebar"] {
        background: rgba(8, 17, 32, .96);
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] * {
        color: var(--ink);
    }

    .block-container {
        max-width: 1220px;
        padding-top: 1.8rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: #f3f7ff !important;
        letter-spacing: -0.025em;
    }

    p, label, [data-testid="stMarkdownContainer"] {
        color: var(--ink);
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 30px 32px;
        border: 1px solid rgba(142, 177, 240, .24);
        border-radius: 24px;
        background: linear-gradient(115deg, rgba(26, 50, 91, .88), rgba(15, 39, 62, .78));
        box-shadow: 0 18px 50px rgba(0, 0, 0, .2);
        margin-bottom: 24px;
    }

    .hero:after {
        content: " ";
        position: absolute;
        width: 250px;
        height: 250px;
        border-radius: 50%;
        right: -75px;
        top: -115px;
        border: 1px solid rgba(98, 230, 213, .22);
        box-shadow: 0 0 0 24px rgba(98, 230, 213, .035),
                    0 0 0 50px rgba(98, 230, 213, .025);
    }

    .eyebrow {
        text-transform: uppercase;
        letter-spacing: .16em;
        font-size: .72rem;
        font-weight: 700;
        color: var(--cyan);
        margin-bottom: 10px;
    }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(1.9rem, 4vw, 3rem);
        line-height: 1.1;
        letter-spacing: -.045em;
        font-weight: 700;
        color: #f5f8ff;
        margin: 0 0 12px 0;
    }

    .hero-copy {
        max-width: 720px;
        line-height: 1.65;
        color: #c0cee7;
        font-size: 1rem;
        margin: 0;
    }

    .hero-chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 20px;
    }

    .chip {
        display: inline-block;
        font-size: .78rem;
        padding: 6px 10px;
        border-radius: 100px;
        color: #dbeaff;
        background: rgba(210, 226, 255, .08);
        border: 1px solid rgba(190, 213, 255, .16);
    }

    .section-kicker {
        text-transform: uppercase;
        letter-spacing: .13em;
        font-weight: 700;
        font-size: .73rem;
        color: #91a9ce;
        margin: 4px 0 8px;
    }

    .panel {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 20px;
        padding: 20px 22px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, .12);
        height: 100%;
    }

    .panel-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.14rem;
        font-weight: 700;
        color: #f4f7ff;
        margin-bottom: 8px;
    }

    .panel-copy {
        color: var(--muted);
        font-size: .91rem;
        line-height: 1.6;
    }

    .step {
        display: flex;
        gap: 12px;
        margin: 15px 0;
        align-items: flex-start;
    }

    .step-num {
        min-width: 30px;
        height: 30px;
        border-radius: 10px;
        display: flex;
        justify-content: center;
        align-items: center;
        color: #061322;
        font-weight: 800;
        background: linear-gradient(135deg, var(--cyan), var(--blue));
    }

    .step-text strong {
        color: #eaf1ff;
        display: block;
        font-size: .9rem;
        margin-bottom: 2px;
    }

    .step-text span {
        color: var(--muted);
        font-size: .82rem;
        line-height: 1.45;
    }

    .status-card {
        padding: 21px 23px;
        border-radius: 20px;
        border: 1px solid var(--line);
        margin: 14px 0 10px;
        animation: rise-in .32s ease-out;
    }

    .status-spam {
        background: linear-gradient(120deg, rgba(111, 35, 57, .42), rgba(43, 24, 47, .78));
        border-color: rgba(255, 112, 141, .42);
    }

    .status-ham {
        background: linear-gradient(120deg, rgba(17, 88, 81, .38), rgba(20, 43, 60, .78));
        border-color: rgba(98, 230, 213, .35);
    }

    .status-label {
        font-size: .75rem;
        text-transform: uppercase;
        letter-spacing: .13em;
        font-weight: 700;
        color: #b6c6e0;
        margin-bottom: 6px;
    }

    .status-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.8rem;
        line-height: 1.2;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 7px;
    }

    .status-description {
        font-size: .9rem;
        line-height: 1.55;
        color: #c7d4e9;
    }

    @keyframes rise-in {
        from { opacity: .15; transform: translateY(7px); }
        to { opacity: 1; transform: translateY(0); }
    }

    [data-testid="stMetric"] {
        padding: 15px 16px;
        background: rgba(18, 31, 55, .72);
        border: 1px solid var(--line);
        border-radius: 16px;
    }

    [data-testid="stMetricLabel"] {
        color: #a7b6d2;
    }

    [data-testid="stMetricValue"] {
        color: #f3f7ff;
    }

    .stTextArea textarea {
        background: rgba(9, 19, 35, .9) !important;
        color: #f3f7ff !important;
        border: 1px solid rgba(143, 174, 228, .32) !important;
        border-radius: 14px !important;
        line-height: 1.6 !important;
    }

    .stTextArea textarea:focus {
        border-color: #62e6d5 !important;
        box-shadow: 0 0 0 1px #62e6d5 !important;
    }

    .stButton > button {
        border-radius: 12px;
        border: 1px solid rgba(135, 174, 242, .34);
        min-height: 2.7rem;
        font-weight: 600;
        transition: all .16s ease;
    }

    .stButton > button[kind="primary"] {
        color: #061322;
        background: linear-gradient(100deg, #62e6d5, #88b7ff);
        border: none;
    }

    .stButton > button:hover {
        border-color: #62e6d5;
        transform: translateY(-1px);
    }

    [data-testid="stProgressBar"] > div > div {
        background: linear-gradient(90deg, #62e6d5, #88b7ff);
    }

    div[data-testid="stAlert"] {
        border-radius: 13px;
    }

    .footer {
        color: #8296b8;
        text-align: center;
        font-size: .79rem;
        margin-top: 28px;
        padding-top: 18px;
        border-top: 1px solid var(--line);
    }

    @media (max-width: 700px) {
        .hero { padding: 24px 20px; }
        .panel { padding: 17px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------- Model loading -----------------------

@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model was not found at: {MODEL_PATH}"
        )
    return tf.keras.models.load_model(MODEL_PATH)


# ----------------------- Session state ----------------------

if "sms_input" not in st.session_state:
    st.session_state.sms_input = ""

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "scan_history" not in st.session_state:
    st.session_state.scan_history = []


def load_example(text):
    st.session_state.sms_input = text


def clear_input():
    st.session_state.sms_input = ""
    st.session_state.last_result = None


# --------------------------- Sidebar ------------------------

with st.sidebar:
    st.markdown("## 🛡️ SMS Shield")
    st.caption("Deep learning spam detection")
    st.markdown("---")

    st.markdown("### ⚙️ Detection settings")
    threshold = st.slider(
        "Spam decision threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help=(
            "Messages scoring at or above this threshold are labelled spam. "
            "Changing this threshold changes the decision rule, not the model."
        ),
    )

    st.markdown("---")
    st.markdown("### 🧠 Model information")
    st.markdown(
        """
        <div class="panel" style="padding:15px;">
          <div class="panel-title">BiLSTM Neural Network</div>
          <div class="panel-copy">
            Learns patterns from message text and returns a spam score.
          </div>
          <div class="hero-chip-row">
            <span class="chip">NLP</span>
            <span class="chip">TensorFlow</span>
            <span class="chip">Binary classifier</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")
    st.caption("Prototype for educational use. Predictions can be incorrect.")


# ---------------------------- Header ------------------------

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">✦ Intelligent message screening</div>
      <div class="hero-title">Know what's in your inbox.</div>
      <p class="hero-copy">
        Inspect an SMS with your trained deep learning model.
        Get a clear spam prediction and a score in seconds.
      </p>
      <div class="hero-chip-row">
        <span class="chip">🛡️ Spam screening</span>
        <span class="chip">⚡ Fast prediction</span>
        <span class="chip">🔎 Text analysis</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Top summary cards
m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Model", "BiLSTM", help="Bidirectional LSTM text classifier")
with m2:
    st.metric("Task", "Text classification")
with m3:
    st.metric("Output", "Ham / Spam")

st.markdown("")


# ----------------------- Main workspace ---------------------

left, right = st.columns([1.35, 0.85], gap="large")

with left:
    st.markdown('<div class="section-kicker">01 / Message input</div>', unsafe_allow_html=True)
    st.markdown("### Analyze an SMS")
    st.markdown(
        '<p class="panel-copy">Paste a message or load one of the samples below to try the detector.</p>',
        unsafe_allow_html=True,
    )

    ex1, ex2 = st.columns(2)
    with ex1:
        st.button(
            "✨ Try a spam example",
            use_container_width=True,
            on_click=load_example,
            args=("Congratulations! You have won a free prize. Call now to claim your reward!",),
        )
    with ex2:
        st.button(
            "💬 Try a normal example",
            use_container_width=True,
            on_click=load_example,
            args=("Hi, I will reach college in 10 minutes. Please wait for me.",),
        )

    message = st.text_area(
        "SMS message",
        key="sms_input",
        height=180,
        max_chars=5000,
        placeholder="Type or paste the SMS message you want to check...",
        label_visibility="collapsed",
    )

    run_col, clear_col = st.columns([2, 1])
    with run_col:
        detect_clicked = st.button(
            "🛡️  Detect spam",
            type="primary",
            use_container_width=True,
        )
    with clear_col:
        st.button(
            "Clear message",
            use_container_width=True,
            on_click=clear_input,
        )

    if detect_clicked:
        if not message.strip():
            st.warning("Please enter a message before running detection.")
        else:
            try:
                with st.spinner("Analyzing the message with the BiLSTM model…"):
                    model = load_model()
                    input_tensor = tf.constant([message], dtype=tf.string)
                    score = float(model.predict(input_tensor, verbose=0).reshape(-1)[0])

                label = "SPAM" if score >= threshold else "NORMAL"
                result = {
                    "message": message.strip(),
                    "score": score,
                    "label": label,
                    "threshold": threshold,
                    "time": datetime.now().strftime("%H:%M:%S"),
                }
                st.session_state.last_result = result
                st.session_state.scan_history.insert(0, result)
                st.session_state.scan_history = st.session_state.scan_history[:5]

            except Exception as exc:
                st.error("The trained model could not complete the prediction.")
                st.exception(exc)

    result = st.session_state.last_result

    if result:
        # Reapply current threshold if the user changes it after a scan.
        score = result["score"]
        label = "SPAM" if score >= threshold else "NORMAL"

        st.markdown('<div class="section-kicker">02 / Detection result</div>', unsafe_allow_html=True)

        if label == "SPAM":
            st.markdown(
                f"""
                <div class="status-card status-spam">
                  <div class="status-label">⚠ Detection outcome</div>
                  <div class="status-value">🚨 SPAM DETECTED</div>
                  <div class="status-description">
                    The model's spam score is above the selected threshold.
                    Treat unexpected links, prize claims, and requests for personal information carefully.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="status-card status-ham">
                  <div class="status-label">✓ Detection outcome</div>
                  <div class="status-value">✅ NORMAL MESSAGE</div>
                  <div class="status-description">
                    The model's spam score is below the selected threshold.
                    This is a prediction, not a guarantee that the message is safe.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        score_col, threshold_col = st.columns(2)
        with score_col:
            st.metric("Estimated spam score", f"{score:.1%}")
            st.progress(min(max(score, 0.0), 1.0), text="Spam score")
        with threshold_col:
            st.metric("Current threshold", f"{threshold:.0%}")
            st.caption(
                "A score at or above the threshold is labelled spam."
            )

        with st.expander("View analyzed message"):
            st.write(result["message"])


with right:
    st.markdown('<div class="section-kicker">How it works</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="panel">
          <div class="panel-title">From message to prediction</div>
          <div class="panel-copy">
            The application sends your text to the trained model. The model outputs a spam score,
            then the selected threshold determines the displayed category.
          </div>
          <div class="step">
            <div class="step-num">1</div>
            <div class="step-text">
              <strong>Enter a message</strong>
              <span>Paste the SMS you want to inspect.</span>
            </div>
          </div>
          <div class="step">
            <div class="step-num">2</div>
            <div class="step-text">
              <strong>Analyze text patterns</strong>
              <span>The saved BiLSTM model processes the message.</span>
            </div>
          </div>
          <div class="step">
            <div class="step-num">3</div>
            <div class="step-text">
              <strong>Review the result</strong>
              <span>See the predicted category and estimated spam score.</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")
    st.markdown(
        """
        <div class="panel">
          <div class="panel-title">💡 Remember</div>
          <div class="panel-copy">
            A high score does not prove fraud, and a low score does not prove safety.
            Avoid opening unexpected links or sharing passwords and verification codes.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------- Recent scans ---------------------

if st.session_state.scan_history:
    st.markdown("")
    st.markdown('<div class="section-kicker">03 / Recent activity</div>', unsafe_allow_html=True)
    st.markdown("### Recent scans")

    rows = []
    for item in st.session_state.scan_history:
        rows.append(
            {
                "Time": item["time"],
                "Prediction": item["label"],
                "Spam score": f'{item["score"]:.1%}',
                "Message": (
                    item["message"][:75] + "…"
                    if len(item["message"]) > 75
                    else item["message"]
                ),
            }
        )
    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )


# ----------------------------- Footer -----------------------

st.markdown(
    """
    <div class="footer">
      SMS Shield · SMS Spam Detection Using Deep Learning<br>
      Built with Streamlit and TensorFlow · College final-year project
    </div>
    """,
    unsafe_allow_html=True,
)
