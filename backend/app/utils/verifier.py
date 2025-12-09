import os
import re
import asyncio
from typing import List, Dict, Optional
import httpx
from rapidfuzz import fuzz

HEADERS = {"User-Agent": "Mozilla/5.0 (FakeJobVerifier/1.0)"}
ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID", "d0dac5ab")
ADZUNA_APP_KEY = os.getenv("ADZUNA_APP_KEY", "8ad5b4c10f639bd4dda3e5d4649cd9b1")

# Constants
FREE_EMAIL_PROVIDERS = {
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "live.com",
    "icloud.com", "aol.com", "proton.me", "protonmail.com", "zoho.com", "mail.com",
}

DISPOSABLE_EMAIL_PATTERNS = {
    "tempmail", "10minutemail", "mailinator", "guerrillamail",
}

SUSPICIOUS_PHRASES = [
    "no interview", "quick money", "wire transfer", "urgent hiring",
    "send your bank", "gift card", "training fee", "application fee",
    "crypto", "whatsapp only", "telegram only", "20 minutes onboarding",
    "immediate joining no experience", "ssn", "pay to start",
]

# Helper Functions
def normalize_text(text: str) -> str:
    """Normalize text for comparison"""
    if not text:
        return ""
    cleaned = text.strip().lower()
    return re.sub(r"\s+", " ", cleaned)

def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate fuzzy similarity between two texts"""
    n1 = normalize_text(text1)
    n2 = normalize_text(text2)
    if not n1 or not n2:
        return 0.0
    return float(fuzz.token_set_ratio(n1, n2))

def extract_company_name(job_title: str, job_description: str, company_input: str = "") -> str:
    """Extract or infer company name from job posting"""
    if company_input:
        return company_input.strip()
    
    # Try to extract from email domain
    email_pattern = r"[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+)\.[A-Za-z]{2,}"
    email_match = re.search(email_pattern, job_description or "")
    if email_match:
        domain = email_match.group(1).lower()
        parts = re.split(r"[.\-]", domain)
        return parts[0]
    
    # Try to find capitalized words (potential company names)
    capitalized_word = re.search(r"\b([A-Z][a-zA-Z]+)\b", job_description or "")
    if capitalized_word:
        return capitalized_word.group(1)
    
    return ""

# Async Adzuna API Search
async def search_adzuna_jobs_async(
    job_title: str,
    company_name: str = "",
    location: str = "",
    country: str = "us",
    num_pages: int = 2
) -> Dict:
    """
    Async version of Adzuna job search - MUCH FASTER
    Searches multiple pages concurrently instead of sequentially
    """
    if not ADZUNA_APP_ID or not ADZUNA_APP_KEY:
        return {
            "found": False,
            "matches": 0,
            "sample": None,
            "note": "missing adzuna keys",
        }
    
    normalized_title = normalize_text(job_title)
    normalized_company = normalize_text(company_name)
    total_matches = 0
    best_match = None
    
    async def fetch_page(client: httpx.AsyncClient, page_num: int):
        """Fetch a single page of results"""
        api_url = f"https://api.adzuna.com/v1/api/jobs/{country}/search/{page_num}"
        params = {
            "app_id": ADZUNA_APP_ID,
            "app_key": ADZUNA_APP_KEY,
            "what": job_title,
            "where": location or "",
            "results_per_page": 50,
            "content-type": "application/json",
        }
        
        try:
            resp = await client.get(api_url, params=params, timeout=5.0)  # Reduced timeout
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return None
    
    # Fetch all pages concurrently
    async with httpx.AsyncClient(headers=HEADERS) as client:
        tasks = [fetch_page(client, page) for page in range(1, num_pages + 1)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process all results
    for job_data in results:
        if not job_data or isinstance(job_data, Exception):
            continue
        
        for job in job_data.get("results", []):
            job_title_api = normalize_text(job.get("title", ""))
            company_info = job.get("company") or {}
            job_company_api = normalize_text(company_info.get("display_name", ""))
            
            title_sim = calculate_similarity(normalized_title, job_title_api)
            
            if normalized_company:
                company_sim = calculate_similarity(normalized_company, job_company_api)
            else:
                company_sim = 100.0
            
            if title_sim >= 75 and company_sim >= 70:
                total_matches += 1
                if not best_match:
                    best_match = {
                        "title": job.get("title"),
                        "company": company_info.get("display_name"),
                        "url": job.get("redirect_url"),
                        "source": "adzuna",
                    }
    
    return {
        "found": total_matches > 0,
        "matches": total_matches,
        "sample": best_match,
    }

# Synchronous wrapper for backwards compatibility
def search_adzuna_jobs(
    job_title: str,
    company_name: str = "",
    location: str = "",
    country: str = "us",
    num_pages: int = 2
) -> Dict:
    """
    Synchronous wrapper - creates event loop if needed
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(
        search_adzuna_jobs_async(job_title, company_name, location, country, num_pages)
    )

# Email Analysis Functions
def find_email_addresses(text: str) -> List[str]:
    """Extract all email addresses from text"""
    pattern = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    return re.findall(pattern, text or "")

def analyze_email_domain(email_address: str, company_name: Optional[str] = None) -> Dict:
    """Analyze an email address for suspicious patterns"""
    _username, domain = email_address.split("@", 1)
    domain_lower = domain.lower()
    signals = []
    
    # Check for free email providers
    if domain_lower in FREE_EMAIL_PROVIDERS:
        signals.append("free_domain")
    
    # Check for disposable email patterns
    for pattern in DISPOSABLE_EMAIL_PATTERNS:
        if pattern in domain_lower:
            signals.append("disposable_like")
            break
    
    # Check company-domain mismatch
    if company_name:
        company_clean = re.sub(r"[^a-z0-9]", "", company_name.lower())
        domain_clean = re.sub(r"[^a-z0-9]", "", domain_lower.split(".")[0])
        if company_clean and company_clean not in domain_clean:
            signals.append("company_domain_mismatch")
    
    return {"email": email_address, "domain": domain_lower, "signals": signals}

def find_suspicious_keywords(text: str) -> List[str]:
    """Find suspicious phrases in job posting text"""
    norm = normalize_text(text)
    hits = []
    for phrase in SUSPICIOUS_PHRASES:
        if phrase in norm:
            hits.append(phrase)
    return hits

def calculate_fraud_probability(
    model_prediction: float,
    found_on_adzuna: bool,
    email_analysis: List[Dict],
    suspicious_keywords: List[str]
) -> Dict:
    """
    Calculate final fraud probability by combining ML model output
    with verification signals
    """
    fraud_probability = float(model_prediction)
    reasons = []
    
    # Adjust based on Adzuna verification
    if found_on_adzuna:
        fraud_probability *= 0.8
        reasons.append("Found on public job index (Adzuna)")
    
    # Adjust based on email analysis
    for email_check in email_analysis:
        for warning in email_check["signals"]:
            if warning == "free_domain":
                fraud_probability = min(1.0, fraud_probability + 0.10)
                reasons.append(f"Free email domain: {email_check['domain']}")
            elif warning == "disposable_like":
                fraud_probability = min(1.0, fraud_probability + 0.20)
                reasons.append(f"Disposable-like email: {email_check['domain']}")
            elif warning == "company_domain_mismatch":
                fraud_probability = min(1.0, fraud_probability + 0.15)
                reasons.append(f"Email domain does not match company: {email_check['domain']}")
    
    # Adjust based on suspicious keywords
    if suspicious_keywords:
        penalty = min(0.25, 0.05 * len(suspicious_keywords))
        fraud_probability = min(1.0, fraud_probability + penalty)
        reasons.append("Suspicious phrases: " + ", ".join(suspicious_keywords[:5]))
    
    fake_pct = round(fraud_probability * 100, 1)
    real_pct = round((1.0 - fraud_probability) * 100, 1)
    
    return {"real_pct": real_pct, "fake_pct": fake_pct, "reasons": reasons}

# Main Verification Functions
async def run_full_verification_async(
    job_title: str,
    job_description: str,
    company_name: str = "",
    job_location: str = ""
) -> Dict:
    """
    ASYNC version - runs all verification checks
    This is what you should use in FastAPI
    """
    # Extract company name
    actual_company = extract_company_name(job_title, job_description, company_name)
    
    # Search Adzuna (async)
    adzuna_results = await search_adzuna_jobs_async(
        job_title,
        company_name=actual_company,
        location=job_location,
    )
    
    # Email analysis (synchronous, fast)
    emails = find_email_addresses(job_description)
    email_checks = [analyze_email_domain(email, company_name=actual_company) for email in emails]
    
    # Keyword analysis (synchronous, fast)
    full_text = f"{job_title} {job_description}"
    suspicious_words = find_suspicious_keywords(full_text)
    
    return {
        "api": adzuna_results,
        "emails": email_checks,
        "kw_hits": suspicious_words,
    }

def run_full_verification(
    job_title: str,
    job_description: str,
    company_name: str = "",
    job_location: str = ""
) -> Dict:
    """
    Synchronous wrapper for backwards compatibility
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(
        run_full_verification_async(job_title, job_description, company_name, job_location)
    )

# Public API (keep for backwards compatibility)
def verify_all(title: str, description: str, company: str = "", location: str = "") -> Dict:
    """Main verification function - synchronous"""
    return run_full_verification(title, description, company, location)

async def verify_all_async(title: str, description: str, company: str = "", location: str = "") -> Dict:
    """Main verification function - ASYNC (use this in FastAPI!)"""
    return await run_full_verification_async(title, description, company, location)

def compute_confidence(
    model_fake_prob: float,
    api_found: bool,
    email_checks: List[Dict],
    kw_hits: List[str]
) -> Dict:
    """Compute final confidence score"""
    return calculate_fraud_probability(model_fake_prob, api_found, email_checks, kw_hits)