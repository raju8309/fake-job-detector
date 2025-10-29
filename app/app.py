# app/app.py
from pathlib import Path
import sys
import numpy as np
import streamlit as st
import joblib

# --- paths and imports ---
ROOT = Path(__file__).resolve().parents[1]  # project root: .../fake-job-detector
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.text_cleaning import clean_text  

# --- load artifacts ---
MODEL_FILE = ROOT / "models" / "fake_job_model.pkl"
VEC_FILE   = ROOT / "models" / "tfidf_vectorizer.pkl"

model = joblib.load(MODEL_FILE)
vectorizer = joblib.load(VEC_FILE)

# --- page setup ---
st.set_page_config(
    page_title="Fake Job Posting Detector",
    page_icon="🕵️‍♂️",
    layout="wide",
)

# --- styles ---
st.markdown(
    """
<style>
/* layout */
.block-container { max-width: 1100px !important; }
:root { --brand:#3b5bdb; --brand2:#7c3aed; --muted:#6b7280; }

/* hero */
.hero-wrap {
  margin: -3rem -4rem 2rem -4rem;
  padding: 64px 0;
  background: linear-gradient(135deg, var(--brand), var(--brand2));
  color: #fff;
  text-align: center;
}
.hero-title { font-size: 56px; line-height: 1.1; font-weight: 800; margin: 0; }
.hero-sub { font-size: 16px; opacity:.95; margin-top: 10px; }
.hero-bullets { margin-top: 10px; font-size: 14px; opacity: .9; }

/* card */
.card {
  margin: -40px auto 24px auto;
  max-width: 700px;
  background: #fff;
  border-radius: 16px;
  border: 1px solid #e5e7eb;
  box-shadow: 0 8px 28px rgba(2,6,23,.08);
  padding: 22px;
}
@media (prefers-color-scheme: dark){
  .card{ background: #0f172a; border-color: #334155; }
}

/* inputs */
.stTextInput > div > div > input,
.stTextArea > div > textarea { border-radius: 12px; }
.stButton > button {
  width: 100%;
  height: 46px;
  border-radius: 12px;
  background: #334155;
  border: 1px solid #334155;
  color: #fff;
  font-weight: 600;
}
.stButton > button:hover { background: #1f2937; border-color: #1f2937; }

/* section titles */
.section-title { text-align:center; margin: 24px 0 8px; font-size: 32px; font-weight: 800; }
.section-sub { text-align:center; color: var(--muted); margin-bottom: 18px; }

/* info cards */
.info-grid { display:grid; grid-template-columns: repeat(3, 1fr); gap:16px; }
.info-card {
  background: #fff; border:1px solid #e5e7eb; border-radius:14px;
  padding:16px; box-shadow: 0 6px 22px rgba(2,6,23,.05);
}
.info-card h4 { margin:.25rem 0 .25rem; }
.info-card p { color: #6b7280; font-size: 14px; margin:0; }
@media (prefers-color-scheme: dark){
  .info-card{ background:#0f172a; border-color:#334155; }
  .info-card p{ color:#94a3b8; }
}

/* metrics */
.metrics { display:flex; gap:14px; margin-top:8px; }
.metric {
  flex:1; text-align:center; border:1px solid #e5e7eb;
  border-radius:12px; padding:12px;
}
.metric small{ color:#6b7280; }
.metric h3{ margin:.25rem 0; font-size: 28px; }
@media (prefers-color-scheme: dark){ .metric{ border-color:#334155; } }

/* confidence bar */
.conf-bar {
  width:100%; height:10px; border-radius:999px; margin-top:12px;
  background: linear-gradient(90deg, #22c55e 0%, #22c55e var(--real), #ef4444 var(--real), #ef4444 100%);
  border:1px solid #e5e7eb;
}
@media (prefers-color-scheme: dark){ .conf-bar{ border-color:#334155; } }

/* result banner */
.result {
  margin-top: 12px; padding: 14px; border-radius: 12px; font-weight: 600; border:1px solid;
}
.result.real { background: rgba(34,197,94,.12); color:#065f46; border-color: rgba(34,197,94,.4); }
.result.fake { background: rgba(239,68,68,.12); color:#7f1d1d; border-color: rgba(239,68,68,.4); }

/* remove the stray white rounded box under the hero */
section[data-testid="stHeader"] div:has(> div:empty),
div[data-testid="stVerticalBlock"] > div:empty {
  display: none !important;
  visibility: hidden !important;
  height: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# --- hero ---
st.markdown(
    """
<div class="hero-wrap">
  <div class="hero-title">Detect Fake Jobs<br/>Protect Your Career</div>
  <div class="hero-sub">AI-powered analysis to identify fraudulent job postings in seconds. Stay safe from scams and focus on real opportunities.</div>
  <div class="hero-bullets">• Instant Analysis &nbsp;&nbsp;• AI-Powered Detection &nbsp;&nbsp;• Free to Use</div>
</div>
""",
    unsafe_allow_html=True,
)

# --- input card ---
st.markdown('<div class="card">', unsafe_allow_html=True)
job_title = st.text_input("Job Title", placeholder="e.g., Senior Software Engineer")
job_desc  = st.text_area("Job Description", placeholder="Paste the complete job posting here...", height=140)
run_btn   = st.button("🔎 Analyze Job Posting")
st.markdown('</div>', unsafe_allow_html=True)

def predict_probs(X_vec) -> tuple[float, float]:
    """Return (p_fake, p_real). Uses true probabilities when available; otherwise a sigmoid on decision score."""
    if hasattr(model, "predict_proba"):
        p_fake = float(model.predict_proba(X_vec)[0][1])
        return p_fake, 1.0 - p_fake
    if hasattr(model, "decision_function"):
        score = float(model.decision_function(X_vec)[0])
        p_fake = 1.0 / (1.0 + np.exp(-score))  # approximate prob
        return p_fake, 1.0 - p_fake
    # last resort: hard label -> 0/1
    label = int(model.predict(X_vec)[0])
    return (1.0, 0.0) if label == 1 else (0.0, 1.0)

# --- prediction ---
if run_btn:
    if not job_title.strip() or not job_desc.strip():
        st.warning("Please enter both job title and description.")
    else:
        text = f"{job_title} {job_desc}".strip()
        cleaned = clean_text(text)
        X = vectorizer.transform([cleaned])

        p_fake, p_real = predict_probs(X)
        fake_pct = round(p_fake * 100, 1)
        real_pct = round(p_real * 100, 1)
        is_fake = p_fake >= 0.5

        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="metrics">
              <div class="metric"><small>Real</small><h3>{real_pct}%</h3></div>
              <div class="metric"><small>Fake</small><h3>{fake_pct}%</h3></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(f'<div class="conf-bar" style="--real:{real_pct}%;"></div>', unsafe_allow_html=True)

        if is_fake:
            st.markdown('<div class="result fake">❌ This job looks FAKE.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result real">✅ This job appears REAL.</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

# --- info section ---
st.markdown('<div class="section-title">How It Works</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Cutting-edge AI technology to keep you safe from job scams</div>', unsafe_allow_html=True)
st.markdown(
    """
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
    <p>Avoid scams and focus on verified opportunities by checking each posting quickly.</p>
  </div>
</div>
""",
    unsafe_allow_html=True,
)