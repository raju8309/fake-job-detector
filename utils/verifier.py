# utils/verifier.py

import os
import re
import requests
from rapidfuzz import fuzz

# Standard browser header to avoid being blocked
HEADERS = {"User-Agent": "Mozilla/5.0 (FakeJobVerifier/1.0)"}

# Adzuna API credentials - pull from environment variables
ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID", "d0dac5ab")
ADZUNA_APP_KEY = os.getenv("ADZUNA_APP_KEY", "8ad5b4c10f639bd4dda3e5d4649cd9b1")


def normalize_text(text):
    """Clean up text by converting to lowercase and removing extra whitespace"""
    if not text:
        return ""
    cleaned = text.strip().lower()
    return re.sub(r"\s+", " ", cleaned)


def calculate_similarity(text1, text2):
    """Compare two strings and return a similarity score (0-100)"""
    normalized1 = normalize_text(text1)
    normalized2 = normalize_text(text2)
    return fuzz.token_set_ratio(normalized1, normalized2)


def extract_company_name(job_title, job_description, company_input=""):
    """
    Try to figure out the company name if the user didn't provide one.
    First checks if they already gave us a company name.
    Then looks for email addresses and extracts the domain.
    Falls back to finding the first capitalized word.
    """
    if company_input:
        return company_input.strip()

    # Look for email addresses like hr@company.com
    email_pattern = r"[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+)\.[A-Za-z]{2,}"
    email_match = re.search(email_pattern, job_description or "")

    if email_match:
        domain = email_match.group(1).lower()
        # Get the first part of the domain (before any dots or hyphens)
        company_parts = re.split(r"[.\-]", domain)
        return company_parts[0]

    # Last resort: find first capitalized word in the description
    capitalized_word = re.search(r"\b([A-Z][a-zA-Z]+)\b", job_description or "")
    if capitalized_word:
        return capitalized_word.group(1)

    return ""


def search_adzuna_jobs(job_title, company_name="", location="", country="us", num_pages=2):
    """
    Search Adzuna's job database to see if similar jobs exist.
    Returns a dictionary with info about whether we found matching jobs.
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

    # Search through multiple pages of results
    for page_num in range(1, num_pages + 1):
        api_url = f"https://api.adzuna.com/v1/api/jobs/{country}/search/{page_num}"

        query_params = {
            "app_id": ADZUNA_APP_ID,
            "app_key": ADZUNA_APP_KEY,
            "what": job_title,
            "where": location or "",
            "results_per_page": 50,
            "content-type": "application/json",
        }

        try:
            response = requests.get(api_url, params=query_params, headers=HEADERS, timeout=8)
            response.raise_for_status()
            job_data = response.json()
        except Exception:
            # If the request fails, just skip to the next page
            continue

        # Check each job posting for matches
        for job in job_data.get("results", []):
            job_title_from_api = normalize_text(job.get("title"))

            company_info = job.get("company") or {}
            job_company_from_api = normalize_text(company_info.get("display_name"))

            # Calculate how similar the titles are
            title_similarity = calculate_similarity(normalized_title, job_title_from_api)

            # Only check company similarity if we have a company name to compare
            if normalized_company:
                company_similarity = calculate_similarity(normalized_company, job_company_from_api)
            else:
                company_similarity = 100  # Don't penalize if no company provided

            # Count as a match if both title and company are similar enough
            if title_similarity >= 75 and company_similarity >= 70:
                total_matches += 1

                # Save the first good match as an example
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


# Common free email providers that legitimate companies usually don't use
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

# Patterns often found in disposable email addresses
DISPOSABLE_EMAIL_PATTERNS = {
    "tempmail",
    "10minutemail",
    "mailinator",
    "guerrillamail",
}

# Red flag phrases commonly found in job scams
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


def find_email_addresses(text):
    """Pull out all email addresses from a block of text"""
    email_pattern = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    return re.findall(email_pattern, text or "")


def analyze_email_domain(email_address, company_name=None):
    """
    Check an email address for suspicious characteristics.
    Compares the domain to known free/disposable providers and checks
    if it matches the claimed company name.
    """
    username, domain = email_address.split("@", 1)
    domain_lower = domain.lower()
    red_flags = []

    # Check if it's a free email provider
    if domain_lower in FREE_EMAIL_PROVIDERS:
        red_flags.append("free_domain")

    # Check if it looks like a disposable email service
    for pattern in DISPOSABLE_EMAIL_PATTERNS:
        if pattern in domain_lower:
            red_flags.append("disposable_like")
            break

    # Check if the domain matches the company name
    if company_name:
        # Strip out non-alphanumeric characters for comparison
        company_cleaned = re.sub(r"[^a-z0-9]", "", company_name.lower())
        domain_cleaned = re.sub(r"[^a-z0-9]", "", domain_lower.split(".")[0])

        if company_cleaned and company_cleaned not in domain_cleaned:
            red_flags.append("company_domain_mismatch")

    return {
        "email": email_address,
        "domain": domain_lower,
        "signals": red_flags,
    }


def find_suspicious_keywords(text):
    """Look for common scam phrases in the job posting text"""
    normalized = normalize_text(text)
    found_keywords = []

    for phrase in SUSPICIOUS_PHRASES:
        if phrase in normalized:
            found_keywords.append(phrase)

    return found_keywords


def calculate_fraud_probability(model_prediction, found_on_adzuna, email_analysis, suspicious_keywords):
    """
    Combine all our checks into a final probability that the job is fake.
    Takes the ML model's prediction and adjusts it based on other evidence.
    """
    fraud_probability = model_prediction
    explanation = []

    # If we found the job on Adzuna, it's more likely to be real
    if found_on_adzuna:
        fraud_probability *= 0.8
        explanation.append("Found on public job index (Adzuna)")

    # Check all the email addresses for red flags
    for email_check in email_analysis:
        for warning in email_check["signals"]:
            if warning == "free_domain":
                fraud_probability = min(1.0, fraud_probability + 0.10)
                explanation.append(f"Free email domain: {email_check['domain']}")

            elif warning == "disposable_like":
                fraud_probability = min(1.0, fraud_probability + 0.20)
                explanation.append(f"Disposable-like email: {email_check['domain']}")

            elif warning == "company_domain_mismatch":
                fraud_probability = min(1.0, fraud_probability + 0.15)
                explanation.append(f"Email domain does not match company: {email_check['domain']}")

    # Add points for each suspicious keyword found (up to a limit)
    if suspicious_keywords:
        keyword_penalty = min(0.25, 0.05 * len(suspicious_keywords))
        fraud_probability = min(1.0, fraud_probability + keyword_penalty)
        explanation.append("Suspicious phrases: " + ", ".join(suspicious_keywords[:5]))

    # Convert to percentages
    fake_percentage = round(fraud_probability * 100, 1)
    real_percentage = round((1 - fraud_probability) * 100, 1)

    return {
        "real_pct": real_percentage,
        "fake_pct": fake_percentage,
        "reasons": explanation,
    }


def run_full_verification(job_title, job_description, company_name="", job_location=""):
    """
    Main function that runs all verification checks on a job posting.
    This is what the Streamlit app calls to check if a job is legitimate.
    """
    # Try to figure out the company name if not provided
    actual_company = extract_company_name(job_title, job_description, company_name)

    # Search Adzuna to see if this job exists on legitimate job boards
    adzuna_results = search_adzuna_jobs(
        job_title,
        company_name=actual_company,
        location=job_location,   # <-- FIXED: use location keyword
    )

    # Extract and analyze any email addresses in the posting
    email_addresses = find_email_addresses(job_description)
    email_checks = [
        analyze_email_domain(email, company_name=actual_company)  # <-- FIXED keyword
        for email in email_addresses
    ]

    # Look for suspicious keywords in the job posting
    full_text = f"{job_title} {job_description}"
    suspicious_words = find_suspicious_keywords(full_text)

    return {
        "api": adzuna_results,
        "emails": email_checks,
        "kw_hits": suspicious_words,
    }


# Backward compatibility aliases - these allow existing code to keep using the old function names
def verify_all(title, description, company="", location=""):
    """
    Legacy function name - calls run_full_verification.
    Kept for backward compatibility with existing code.
    """
    return run_full_verification(title, description, company, location)


def compute_confidence(model_fake_prob, api_found, email_checks, kw_hits):
    """
    Legacy function name - calls calculate_fraud_probability.
    Kept for backward compatibility with existing code.
    """
    return calculate_fraud_probability(model_fake_prob, api_found, email_checks, kw_hits)