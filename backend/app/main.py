from pathlib import Path
import sys
from typing import List, Optional
import joblib
import numpy as np
import shap
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
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
SCAM_DATA_PATH = ROOT / "data" / "fake_job_postings.csv"

# Global variables for lazy loading
model = None
vectorizer = None
shap_explainer = None
rag_index = None

def load_models():
    """Lazy load models only when needed"""
    global model, vectorizer
    if model is None or vectorizer is None:
        logger.info("Loading models...")
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        logger.info("Models loaded successfully")
    return model, vectorizer


def load_shap_explainer():
    global shap_explainer
    if shap_explainer is not None:
        return shap_explainer

    current_model, current_vectorizer = load_models()

    def predict_fake_proba(texts: List[str]):
        cleaned_texts = [clean_text(t) for t in texts]
        vectorized_input = current_vectorizer.transform(cleaned_texts)
        if hasattr(current_model, "predict_proba"):
            proba = current_model.predict_proba(vectorized_input)
            return proba[:, 1]
        if hasattr(current_model, "decision_function"):
            decision_scores = current_model.decision_function(vectorized_input)
            return 1.0 / (1.0 + np.exp(-decision_scores))
        preds = current_model.predict(vectorized_input)
        return np.array([1.0 if int(p) == 1 else 0.0 for p in preds])

    masker = shap.maskers.Text()
    shap_explainer = shap.Explainer(predict_fake_proba, masker)
    return shap_explainer


def load_rag_index():
    global rag_index
    if rag_index is not None:
        return rag_index

    current_model, current_vectorizer = load_models()

    if not SCAM_DATA_PATH.exists():
        rag_index = {
            "ready": False,
            "note": "missing scam dataset",
        }
        return rag_index

    df = pd.read_csv(SCAM_DATA_PATH)
    if "fraudulent" not in df.columns:
        rag_index = {
            "ready": False,
            "note": "dataset missing fraudulent column",
        }
        return rag_index

    scam_df = df[df["fraudulent"] == 1].copy()

    def safe_str(x):
        if x is None:
            return ""
        return str(x)

    texts = (
        scam_df["title"].apply(safe_str)
        + " "
        + scam_df["description"].apply(safe_str)
    ).tolist()
    cleaned_texts = [clean_text(t) for t in texts]
    matrix = current_vectorizer.transform(cleaned_texts)

    meta = []
    for _, row in scam_df.iterrows():
        desc = safe_str(row.get("description", ""))
        meta.append(
            {
                "job_id": int(row.get("job_id")) if not pd.isna(row.get("job_id")) else None,
                "title": safe_str(row.get("title", ""))[:160],
                "location": safe_str(row.get("location", ""))[:120],
                "snippet": desc.replace("\n", " ")[:240],
            }
        )

    rag_index = {
        "ready": True,
        "matrix": matrix,
        "meta": meta,
        "note": f"indexed {len(meta)} known scams",
    }
    return rag_index

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
    agentic: Optional[dict] = None


class ExplainabilityToken(BaseModel):
    token: str
    impact: float
    impact_abs: float


class Explainability(BaseModel):
    tokens: List[ExplainabilityToken]


class RagMatch(BaseModel):
    similarity: float
    item: dict


class RagResult(BaseModel):
    max_similarity: float
    matches: List[RagMatch]

class JobResponse(BaseModel):
    real_pct: float
    fake_pct: float
    verdict: str
    reasons: List[str]
    model_real_pct: float
    model_fake_pct: float
    verification: VerificationSignals
    explainability: Optional[Explainability] = None
    rag: Optional[RagResult] = None

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
    load_shap_explainer()
    try:
        load_rag_index()
    except Exception as e:
        logger.warning(f"RAG index load failed: {str(e)}")
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

        rag = None
        try:
            idx = load_rag_index()
            if idx.get("ready"):
                sims = cosine_similarity(vectorized, idx["matrix"])[0]
                top_k = 5
                top_idx = np.argsort(-sims)[:top_k]
                max_sim = float(sims[top_idx[0]]) if len(top_idx) > 0 else 0.0
                matches = []
                for j in top_idx:
                    matches.append(
                        RagMatch(
                            similarity=float(sims[j]),
                            item=idx["meta"][int(j)],
                        )
                    )
                rag = RagResult(max_similarity=max_sim, matches=matches)
        except Exception as e:
            logger.warning(f"RAG retrieval failed: {str(e)}")
        
        # Run verification checks ASYNC - THIS IS THE KEY OPTIMIZATION!
        # Multiple API calls now happen concurrently instead of sequentially
        signals = await verify_all_async(
            payload.title,
            payload.description,
            company=payload.company or "",
            location=payload.location or "",
        )

        explainability = None
        try:
            explainer = load_shap_explainer()
            combined_text_raw = f"{payload.title} {payload.description}".strip()
            explanation = explainer([combined_text_raw])
            tokens = explanation.data[0]
            values = explanation.values[0]
            if hasattr(values, "ndim") and values.ndim == 2:
                values = values[:, -1]

            token_impacts = []
            for tok, val in zip(tokens, values):
                if tok is None:
                    continue
                token_str = str(tok).strip()
                if token_str == "":
                    continue
                token_impacts.append((token_str, float(val)))

            token_impacts = [t for t in token_impacts if t[1] > 0]
            token_impacts.sort(key=lambda x: x[1], reverse=True)
            token_impacts = token_impacts[:20]

            max_abs = max((abs(v) for _, v in token_impacts), default=0.0)
            explainability = Explainability(
                tokens=[
                    ExplainabilityToken(
                        token=t,
                        impact=v,
                        impact_abs=(abs(v) / max_abs if max_abs > 0 else 0.0),
                    )
                    for t, v in token_impacts
                ]
            )
        except Exception as e:
            logger.warning(f"SHAP explainability failed: {str(e)}")
        
        # Combine scores
        combined = compute_confidence(
            model_fake_prob=fake_prob_model,
            api_found=signals["api"]["found"],
            email_checks=signals["emails"],
            kw_hits=signals["kw_hits"],
        )

        # Apply an additional bump when the posting is highly similar to known scams
        if rag is not None and rag.max_similarity >= 0.80:
            combined["fake_pct"] = min(100.0, float(combined["fake_pct"]) + 10.0)
            combined["real_pct"] = max(0.0, 100.0 - float(combined["fake_pct"]))
            combined["reasons"].append(
                f"Highly similar to previously seen scam postings ({rag.max_similarity:.2f})"
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
                agentic=signals.get("agentic"),
            ),
            explainability=explainability,
            rag=rag,
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