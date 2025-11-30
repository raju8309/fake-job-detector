import os
import re
import requests
from rapidfuzz import fuzz

HEADERS = {"User-Agent": "Mozilla/5.0 (FakeJobVerifier/1.0)"}

ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID", "")
ADZUNA_APP_KEY = os.getenv("ADZUNA_APP_KEY", "")


def normalize_text(text: str) -> str:
    if not text:
        return ""
    cleaned = text.strip().lower()
    return re.sub(r"\s+", " ", cleaned)


def calculate_similarity(text1: str, text2: str) -> float:
    n1 = normalize_text(text1)
    n2 = normalize_text(text2)
    if not n1 or not n2:
        return 0.0
    return float(fuzz.token_set_ratio(n1, n2))


def extract_company_name(job_title, job_description, company_input=""):
    if company_input:
        return company_input.strip()

    email_pattern = r"[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+)\.[A-Za-z]{2,}"
    email_match = re.search(email_pattern, job_description or "")
    if email_match:
        domain = email_match.group(1).lower()
        parts = re.split(r"[.\-]", domain)
        return parts[0]

    capitalized_word = re.search(r"\b([A-Z][a-zA-Z]+)\b", job_description or "")
    if capitalized_word:
        return capitalized_word.group(1)

    return ""


def search_adzuna_jobs(job_title, company_name="", location="", country="us", num_pages=2):
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

    for page_num in range(1, num_pages + 1):
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
            resp = requests.get(api_url, params=params, headers=HEADERS, timeout=8)
            resp.raise_for_status()
            job_data = resp.json()
        except Exception:
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


FREE_EMAIL_PROVIDERS = {
    "gmail.com",
    "yahoo.com",
    "outlook.com",
    "hotmail.com",
    "live.com",
    "icloud.com",
    "aol.com",
    "proton.me",
    "protonmail.com",
    "zoho.com",
    "mail.com",
}

DISPOSABLE_EMAIL_PATTERNS = {
    "tempmail",
    "10minutemail",
    "mailinator",
    "guerrillamail",
}

SUSPICIOUS_PHRASES = [
    "no interview",
    "quick money",
    "wire transfer",
    "urgent hiring",
    "send your bank",
    "gift card",
    "training fee",
    "application fee",
    "crypto",
    "whatsapp only",
    "telegram only",
    "20 minutes onboarding",
    "immediate joining no experience",
    "ssn",
    "pay to start",
]


def find_email_addresses(text: str):
    pattern = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    return re.findall(pattern, text or "")


def analyze_email_domain(email_address: str, company_name=None):
    _username, domain = email_address.split("@", 1)
    domain_lower = domain.lower()
    signals = []

    if domain_lower in FREE_EMAIL_PROVIDERS:
        signals.append("free_domain")

    for pattern in DISPOSABLE_EMAIL_PATTERNS:
        if pattern in domain_lower:
            signals.append("disposable_like")
            break

    if company_name:
        company_clean = re.sub(r"[^a-z0-9]", "", company_name.lower())
        domain_clean = re.sub(r"[^a-z0-9]", "", domain_lower.split(".")[0])
        if company_clean and company_clean not in domain_clean:
            signals.append("company_domain_mismatch")

    return {"email": email_address, "domain": domain_lower, "signals": signals}


def find_suspicious_keywords(text: str):
    norm = normalize_text(text)
    hits = []
    for phrase in SUSPICIOUS_PHRASES:
        if phrase in norm:
            hits.append(phrase)
    return hits


def calculate_fraud_probability(model_prediction, found_on_adzuna, email_analysis, suspicious_keywords):
    fraud_probability = float(model_prediction)
    reasons = []

    if found_on_adzuna:
        fraud_probability *= 0.8
        reasons.append("Found on public job index (Adzuna)")

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

    if suspicious_keywords:
        penalty = min(0.25, 0.05 * len(suspicious_keywords))
        fraud_probability = min(1.0, fraud_probability + penalty)
        reasons.append("Suspicious phrases: " + ", ".join(suspicious_keywords[:5]))

    fake_pct = round(fraud_probability * 100, 1)
    real_pct = round((1.0 - fraud_probability) * 100, 1)

    return {"real_pct": real_pct, "fake_pct": fake_pct, "reasons": reasons}


def run_full_verification(job_title, job_description, company_name="", job_location=""):
    actual_company = extract_company_name(job_title, job_description, company_name)

    adzuna_results = search_adzuna_jobs(
        job_title,
        company_name=actual_company,
        location=job_location,
    )

    emails = find_email_addresses(job_description)
    email_checks = [analyze_email_domain(email, company_name=actual_company) for email in emails]

    full_text = f"{job_title} {job_description}"
    suspicious_words = find_suspicious_keywords(full_text)

    return {
        "api": adzuna_results,
        "emails": email_checks,
        "kw_hits": suspicious_words,
    }


def verify_all(title, description, company="", location=""):
    return run_full_verification(title, description, company, location)


def compute_confidence(model_fake_prob, api_found, email_checks, kw_hits):
    return calculate_fraud_probability(model_fake_prob, api_found, email_checks, kw_hits)