from pathlib import Path
import sys
import numpy as np
import streamlit as st
import joblib

# Setup project paths
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import project modules
from utils.text_cleaning import clean_text
from utils.verifier import verify_all, compute_confidence

# Load the trained model and vectorizer
MODEL_PATH = ROOT / "models" / "fake_job_model.pkl"
VECTORIZER_PATH = ROOT / "models" / "tfidf_vectorizer.pkl"
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# Configure the Streamlit page
st.set_page_config(
    page_title="Fake Job Posting Detector",
    page_icon="🕵️‍♂️",
    layout="wide"
)

# Add custom CSS styling
st.markdown("""
<style>
.block-container {
    max-width: 1100px !important;
}

:root {
    --brand: #3b5bdb;
    --brand2: #7c3aed;
    --muted: #6b7280;
}

.hero-wrap {
    margin: -3rem -4rem 2rem -4rem;
    padding: 64px 0;
    background: linear-gradient(135deg, var(--brand), var(--brand2));
    color: #fff;
    text-align: center;
}

.hero-title {
    font-size: 56px;
    line-height: 1.1;
    font-weight: 800;
    margin: 0;
}

.hero-sub {
    font-size: 16px;
    opacity: .95;
    margin-top: 10px;
}

.hero-bullets {
    margin-top: 10px;
    font-size: 14px;
    opacity: .9;
}

.card {
    margin: 24px auto 24px auto;
    max-width: 700px;
    background: #fff;
    border-radius: 16px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 8px 28px rgba(2,6,23,.08);
    padding: 22px;
}

@media (prefers-color-scheme: dark) {
    .card {
        background: #0f172a;
        border-color: #334155;
    }
}

div[data-testid="stTextInput"],
div[data-testid="stTextArea"] {
    background: transparent !important;
    box-shadow: none !important;
    border-radius: 0 !important;
    padding: 0 !important;
}

div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stTextInput"]),
div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stTextArea"]) {
    background: transparent !important;
    box-shadow: none !important;
}

.stTextInput > div > div > input,
.stTextArea > div > textarea {
    border-radius: 12px;
}

.stButton > button {
    width: 100%;
    height: 46px;
    border-radius: 12px;
    background: #334155;
    border: 1px solid #334155;
    color: #fff;
    font-weight: 600;
}

.stButton > button:hover {
    background: #1f2937;
    border-color: #1f2937;
}

.section-title {
    text-align: center;
    margin: 24px 0 8px;
    font-size: 32px;
    font-weight: 800;
}

.section-sub {
    text-align: center;
    color: var(--muted);
    margin-bottom: 18px;
}

.info-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
}

.info-card {
    background: #fff;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 16px;
    box-shadow: 0 6px 22px rgba(2,6,23,.05);
}

.info-card h4 {
    margin: .25rem 0 .25rem;
}

.info-card p {
    color: #6b7280;
    font-size: 14px;
    margin: 0;
}

@media (prefers-color-scheme: dark) {
    .info-card {
        background: #0f172a;
        border-color: #334155;
    }
    .info-card p {
        color: #94a3b8;
    }
}

.metrics {
    display: flex;
    gap: 14px;
    margin-top: 8px;
}

.metric {
    flex: 1;
    text-align: center;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 12px;
}

.metric small {
    color: #6b7280;
}

.metric h3 {
    margin: .25rem 0;
    font-size: 28px;
}

@media (prefers-color-scheme: dark) {
    .metric {
        border-color: #334155;
    }
}

.conf-bar {
    width: 100%;
    height: 10px;
    border-radius: 999px;
    margin-top: 12px;
    background: linear-gradient(90deg, #22c55e 0%, #22c55e var(--real), #ef4444 var(--real), #ef4444 100%);
    border: 1px solid #e5e7eb;
}

@media (prefers-color-scheme: dark) {
    .conf-bar {
        border-color: #334155;
    }
}

.result {
    margin-top: 12px;
    padding: 14px;
    border-radius: 12px;
    font-weight: 600;
    border: 1px solid;
}

.result.real {
    background: rgba(34,197,94,.12);
    color: #065f46;
    border-color: rgba(34,197,94,.4);
}

.result.fake {
    background: rgba(239,68,68,.12);
    color: #7f1d1d;
    border-color: rgba(239,68,68,.4);
}

section[data-testid="stHeader"] div:has(> div:empty),
div[data-testid="stVerticalBlock"] > div:empty {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}
</style>
""", unsafe_allow_html=True)

# Display hero section
st.markdown("""
<div class="hero-wrap">
    <div class="hero-title">Detect Fake Jobs<br/>Protect Your Career</div>
    <div class="hero-sub">AI-powered analysis to identify fraudulent job postings in seconds. Stay safe from scams and focus on real opportunities.</div>
    <div class="hero-bullets">• Instant Analysis &nbsp;&nbsp;• AI-Powered Detection &nbsp;&nbsp;• Free to Use</div>
</div>
""", unsafe_allow_html=True)

# Create input form
st.markdown('<div class="card">', unsafe_allow_html=True)
job_title = st.text_input("Job Title", placeholder="e.g., Senior Software Engineer")
job_description = st.text_area(
    "Job Description",
    placeholder="Paste the complete job posting here...",
    height=140
)
company_name = st.text_input("Company (optional)")
job_location = st.text_input("Location (optional)")
analyze_button = st.button("🔎 Analyze Job Posting")
st.markdown('</div>', unsafe_allow_html=True)


@st.cache_data(ttl=600)
def get_verification_signals(title, description, company, location):
    """
    Fetch verification signals from external sources.
    Results are cached for 10 minutes to reduce API calls.
    """
    return verify_all(title, description, company=company, location=location)


def get_model_probabilities(vectorized_input):
    """
    Calculate probability scores from the trained model.
    Returns (fake_probability, real_probability) tuple.
    """
    # Try predict_proba first (most models support this)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(vectorized_input)[0]
        fake_prob = float(probabilities[1])
        real_prob = 1.0 - fake_prob
        return fake_prob, real_prob

    # Fall back to decision_function for SVM-like models
    if hasattr(model, "decision_function"):
        decision_score = float(model.decision_function(vectorized_input)[0])
        fake_prob = 1.0 / (1.0 + np.exp(-decision_score))
        real_prob = 1.0 - fake_prob
        return fake_prob, real_prob

    # Last resort: use predict for binary classification
    prediction = int(model.predict(vectorized_input)[0])
    if prediction == 1:
        return 1.0, 0.0
    else:
        return 0.0, 1.0


def build_api_verification_card(api_data):
    """Build HTML card showing Adzuna API verification results"""
    if api_data.get("found"):
        sample_job = api_data.get("sample") or {}
        job_url = sample_job.get('url', '')
        url_link = (
            f'<a href="{job_url}" target="_blank" '
            f'style="font-size:13px;color:#3b82f6;">Open ↗</a>'
            if job_url else ''
        )

        return f"""
        <div class="info-card" style="border-left:6px solid #22c55e;">
            <h4>🌐 Public Index (Adzuna)</h4>
            <p>✅ Found {api_data.get('matches', 1)} similar job(s).</p>
            <p style="font-size:13px;color:#555;">
                {sample_job.get('title', '')} — {sample_job.get('company', '')}
            </p>
            {url_link}
        </div>
        """
    else:
        return """
        <div class="info-card" style="border-left:6px solid #facc15;">
            <h4>🌐 Public Index (Adzuna)</h4>
            <p>No matching results found.</p>
        </div>
        """


def build_email_verification_card(email_checks):
    """Build HTML card showing email verification results"""
    if email_checks:
        email_results = []
        for email_data in email_checks:
            if email_data["signals"]:
                warning = f"❗ {email_data['email']} → {', '.join(email_data['signals'])}"
                email_results.append(warning)
            else:
                success = f"✅ {email_data['email']} looks okay."
                email_results.append(success)

        return f"""
        <div class="info-card" style="border-left:6px solid #3b82f6;">
            <h4>📧 Emails</h4>
            <p>{'<br>'.join(email_results)}</p>
        </div>
        """
    else:
        return """
        <div class="info-card" style="border-left:6px solid #3b82f6;">
            <h4>📧 Emails</h4>
            <p>No email found in the text.</p>
        </div>
        """


def build_keyword_verification_card(keyword_hits):
    """Build HTML card showing risky keyword detection results"""
    if keyword_hits:
        keywords_display = ', '.join(keyword_hits[:8])
        return f"""
        <div class="info-card" style="border-left:6px solid #ef4444;">
            <h4>🚨 Keywords</h4>
            <p>{keywords_display}</p>
        </div>
        """
    else:
        return """
        <div class="info-card" style="border-left:6px solid #22c55e;">
            <h4>🚨 Keywords</h4>
            <p>No risky phrases detected.</p>
        </div>
        """


# Handle the analysis when button is clicked
if analyze_button:
    # Validate inputs
    if not job_title.strip() or not job_description.strip():
        st.warning("Please enter both job title and description.")
    else:
        # Prepare text for model
        combined_text = f"{job_title} {job_description}".strip()
        cleaned_text = clean_text(combined_text)
        vectorized_text = vectorizer.transform([cleaned_text])

        # Get model predictions
        fake_prob_model, real_prob_model = get_model_probabilities(vectorized_text)

        # Get verification signals from external sources
        verification_signals = get_verification_signals(
            job_title,
            job_description,
            company_name,
            job_location,
        )

        # Combine model prediction with verification signals
        combined_confidence = compute_confidence(
            model_fake_prob=fake_prob_model,
            api_found=verification_signals["api"]["found"],
            email_checks=verification_signals["emails"],
            kw_hits=verification_signals["kw_hits"],
        )

        # Extract final results
        real_percentage = combined_confidence["real_pct"]
        fake_percentage = combined_confidence["fake_pct"]
        explanation_reasons = combined_confidence["reasons"]
        is_likely_fake = fake_percentage >= 50.0

        # Display results card
        st.markdown('<div class="card">', unsafe_allow_html=True)

        # Show confidence metrics
        st.markdown(f"""
        <div class="metrics">
            <div class="metric">
                <small>Real (final)</small>
                <h3>{real_percentage}%</h3>
            </div>
            <div class="metric">
                <small>Fake (final)</small>
                <h3>{fake_percentage}%</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Show confidence bar
        st.markdown(
            f'<div class="conf-bar" style="--real:{real_percentage}%;"></div>',
            unsafe_allow_html=True
        )

        # Show verdict
        if is_likely_fake:
            st.markdown(
                '<div class="result fake">❌ This job looks FAKE.</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="result real">✅ This job appears REAL.</div>',
                unsafe_allow_html=True
            )

        # Show raw model score for transparency
        real_model_pct = round(real_prob_model * 100, 1)
        fake_model_pct = round(fake_prob_model * 100, 1)
        st.caption(
            f"Model-only score (before verification): "
            f"Real {real_model_pct}% · Fake {fake_model_pct}%"
        )

        # Verification insights section
        st.markdown("""
        <hr style="margin:2rem 0;border:none;border-top:1px solid #e5e7eb;">
        <div style="text-align:center;margin-bottom:1rem;">
            <h2 style="margin-bottom:0;">🔍 Verification Insights</h2>
            <p style="color:#6b7280;font-size:15px;">
                Cross-check using public job index, email domains, and risky phrases.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Build verification cards
        st.markdown('<div class="info-grid">', unsafe_allow_html=True)

        api_card = build_api_verification_card(verification_signals["api"])
        email_card = build_email_verification_card(verification_signals["emails"])
        keyword_card = build_keyword_verification_card(verification_signals["kw_hits"])

        st.markdown(api_card + email_card + keyword_card, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Show explanation if available
        if explanation_reasons:
            reasons_text = ' · '.join(explanation_reasons)
            st.markdown(
                f"<p style='text-align:center;color:#6b7280;font-size:14px;margin-top:10px;'>"
                f"Why this score: {reasons_text}</p>",
                unsafe_allow_html=True,
            )

        st.markdown('</div>', unsafe_allow_html=True)

# Information section at the bottom
st.markdown('<div class="section-title">How It Works</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-sub">Cutting-edge AI technology to keep you safe from job scams</div>',
    unsafe_allow_html=True
)
st.markdown("""
<div class="info-grid">
    <div class="info-card">
        <h4>🤖 AI-Powered Analysis</h4>
        <p>We clean the text and use TF-IDF plus a trained model to spot red flags and suspicious patterns.</p>
    </div>
    <div class="info-card">
        <h4>⚡ Instant Results</h4>
        <p>Paste a job post and get a real vs fake score in seconds — no forms or signup.</p>
    </div>
    <div class="info-card">
        <h4>🛡️ Stay Protected</h4>
        <p>We cross-check with Adzuna, email/domain checks, and scam keyword detection.</p>
    </div>
</div>
""", unsafe_allow_html=True)