from pathlib import Path
import sys
from typing import List, Optional

import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]  # backend/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import project utilities
from app.utils.text_cleaning import clean_text
from app.utils.verifier import verify_all, compute_confidence


# Load Model & Vectorizer
MODEL_PATH = ROOT / "models" / "fake_job_model.pkl"
VECTORIZER_PATH = ROOT / "models" / "tfidf_vectorizer.pkl"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

# Pydantic Schemas
class JobRequest(BaseModel):
    title: str
    description: str
    company: Optional[str] = ""
    location: Optional[str] = ""

class VerificationSignals(BaseModel):
    api: dict
    emails: List[dict]
    kw_hits: List[str]

class JobResponse(BaseModel):
    real_pct: float
    fake_pct: float
    verdict: str
    reasons: List[str]
    model_real_pct: float
    model_fake_pct: float
    verification: VerificationSignals

# FastAPI App
app = FastAPI(title="Fake Job Detector API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # update later for domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper - Convert model score

def get_model_probabilities(vectorized_input):
    """Return (fake_prob, real_prob)"""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(vectorized_input)[0]
        fake_prob = float(probabilities[1])
        real_prob = 1.0 - fake_prob
        return fake_prob, real_prob

    if hasattr(model, "decision_function"):
        decision_score = float(model.decision_function(vectorized_input)[0])
        fake_prob = 1.0 / (1.0 + np.exp(-decision_score))
        real_prob = 1.0 - fake_prob
        return fake_prob, real_prob

    prediction = int(model.predict(vectorized_input)[0])
    return (1.0, 0.0) if prediction == 1 else (0.0, 1.0)

# Routes
@app.get("/")
def health_check():
    return {"status": "ok", "message": "Fake Job Detector API running"}

@app.post("/analyze-job", response_model=JobResponse)
def analyze_job(payload: JobRequest):
    # Clean + vectorize
    combined_text = f"{payload.title} {payload.description}".strip()
    cleaned = clean_text(combined_text)
    vectorized = vectorizer.transform([cleaned])

    # Model score
    fake_prob_model, real_prob_model = get_model_probabilities(vectorized)

    # Verification checks
    signals = verify_all(
        payload.title,
        payload.description,
        company=payload.company or "",
        location=payload.location or "",
    )

    # Combine scores
    combined = compute_confidence(
        model_fake_prob=fake_prob_model,
        api_found=signals["api"]["found"],
        email_checks=signals["emails"],
        kw_hits=signals["kw_hits"],
    )

    # Final output
    real_pct = float(round(combined["real_pct"], 1))
    fake_pct = float(round(combined["fake_pct"], 1))
    verdict = "fake" if fake_pct >= 50 else "real"

    return JobResponse(
        real_pct=real_pct,
        fake_pct=fake_pct,
        verdict=verdict,
        reasons=combined["reasons"],
        model_real_pct=float(round(real_prob_model * 100, 1)),
        model_fake_pct=float(round(fake_prob_model * 100, 1)),
        verification=VerificationSignals(
            api=signals["api"],
            emails=signals["emails"],
            kw_hits=signals["kw_hits"],
        ),
    )

# Run the app using: python main.py

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )