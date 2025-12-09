from pathlib import Path
import sys
from typing import List, Optional
import joblib
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import project utilities
from app.utils.text_cleaning import clean_text
from app.utils.verifier import verify_all_async, compute_confidence

# Model paths
MODEL_PATH = ROOT / "models" / "fake_job_model.pkl"
VECTORIZER_PATH = ROOT / "models" / "tfidf_vectorizer.pkl"

# Global variables for lazy loading
model = None
vectorizer = None

def load_models():
    """Lazy load models only when needed"""
    global model, vectorizer
    if model is None or vectorizer is None:
        logger.info("Loading models...")
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        logger.info("Models loaded successfully")
    return model, vectorizer

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

# CORS - Update with your Vercel domain for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "https://fake-job-detector-iota.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup event to preload models
@app.on_event("startup")
async def startup_event():
    """Preload models on startup"""
    logger.info("Starting up...")
    load_models()
    logger.info("Ready to serve requests")

# Helper function
def get_model_probabilities(vectorized_input, model_instance):
    """Return (fake_prob, real_prob)"""
    if hasattr(model_instance, "predict_proba"):
        probabilities = model_instance.predict_proba(vectorized_input)[0]
        fake_prob = float(probabilities[1])
        real_prob = 1.0 - fake_prob
        return fake_prob, real_prob
    
    if hasattr(model_instance, "decision_function"):
        decision_score = float(model_instance.decision_function(vectorized_input)[0])
        fake_prob = 1.0 / (1.0 + np.exp(-decision_score))
        real_prob = 1.0 - fake_prob
        return fake_prob, real_prob
    
    prediction = int(model_instance.predict(vectorized_input)[0])
    return (1.0, 0.0) if prediction == 1 else (0.0, 1.0)

# Routes
@app.get("/")
def health_check():
    """Simple health check for uptime monitoring"""
    return {"status": "ok", "message": "Fake Job Detector API running"}

@app.get("/health")
def detailed_health():
    """Detailed health check - use this for UptimeRobot"""
    models_loaded = model is not None and vectorizer is not None
    return {
        "status": "healthy" if models_loaded else "warming_up",
        "models_loaded": models_loaded,
        "version": "1.0.0"
    }

@app.post("/analyze-job", response_model=JobResponse)
async def analyze_job(payload: JobRequest):
    """
    Analyze a job posting for authenticity
    Now using ASYNC verification for better performance!
    """
    try:
        # Ensure models are loaded
        current_model, current_vectorizer = load_models()
        
        # Clean and vectorize text
        combined_text = f"{payload.title} {payload.description}".strip()
        cleaned = clean_text(combined_text)
        vectorized = current_vectorizer.transform([cleaned])
        
        # Get model prediction
        fake_prob_model, real_prob_model = get_model_probabilities(
            vectorized, current_model
        )
        
        # Run verification checks ASYNC - THIS IS THE KEY OPTIMIZATION!
        # Multiple API calls now happen concurrently instead of sequentially
        signals = await verify_all_async(
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
        
        # Prepare response
        real_pct = float(round(combined["real_pct"], 1))
        fake_pct = float(round(combined["fake_pct"], 1))
        verdict = "fake" if fake_pct >= 50 else "real"
        
        logger.info(f"Analysis complete: {verdict} (fake: {fake_pct}%, real: {real_pct}%)")
        
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
    
    except Exception as e:
        logger.error(f"Error analyzing job: {str(e)}", exc_info=True)
        raise

# For local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )