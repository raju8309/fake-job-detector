# utils/verifier.py

import os
import re
import requests
from rapidfuzz import fuzz

# Basic HTTP header so sites don’t block us
UA = {"User-Agent": "Mozilla/5.0 (FakeJobVerifier/1.0)"}

# Adzuna API keys (read from env; fall back to defaults for local dev)
ADZUNA_APP_ID  = os.getenv("ADZUNA_APP_ID", "d0dac5ab")
ADZUNA_APP_KEY = os.getenv("ADZUNA_APP_KEY", "8ad5b4c10f639bd4dda3e5d4649cd9b1")


# ------------ small helpers ------------

def _n(s: str) -> str:
    """Normalize text: lowercase + collapse spaces."""
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _sim(a: str, b: str) -> int:
    """Fuzzy similarity between two strings."""
    return fuzz.token_set_ratio(_n(a), _n(b))


def _guess_company(title: str, description: str, provided: str = "") -> str:
    """Try to guess company name from email/domain if user didn’t type it."""
    if provided:
        return provided.strip()

    # look for something like hr@company.com
    m = re.search(r"[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+)\.[A-Za-z]{2,}", description or "")
    if m:
        dom = m.group(1).lower()
        return re.split(r"[.\-]", dom)[0]

    # simple fallback: first capitalized word in description
    m = re.search(r"\b([A-Z][a-zA-Z]+)\b", description or "")
    return m.group(1) if m else ""


# ------------ Adzuna search ------------

def adzuna_search(
    title: str,
    company: str = "",
    where: str = "",
    country: str = "us",
    pages: int = 2,
) -> dict:
    """
    Look up similar jobs on Adzuna and return:
    {
        "found": bool,
        "matches": int,
        "sample": {"title", "company", "url", "source"} | None,
        "note": optional reason string
    }
    """
    if not ADZUNA_APP_ID or not ADZUNA_APP_KEY:
        return {
            "found": False,
            "matches": 0,
            "sample": None,
            "note": "missing adzuna keys",
        }

    t_norm = _n(title)
    c_norm = _n(company)
    matches = 0
    best = None

    for page in range(1, pages + 1):
        url = f"https://api.adzuna.com/v1/api/jobs/{country}/search/{page}"
        params = {
            "app_id": ADZUNA_APP_ID,
            "app_key": ADZUNA_APP_KEY,
            "what": title,
            "where": where or "",
            "results_per_page": 50,
            "content-type": "application/json",
        }

        try:
            r = requests.get(url, params=params, headers=UA, timeout=8)
            r.raise_for_status()
            data = r.json()
        except Exception:
            # on error, just skip this page
            continue

        for j in data.get("results", []):
            jt = _n(j.get("title"))
            jc = _n((j.get("company") or {}).get("display_name"))

            st = _sim(t_norm, jt)
            sc = _sim(c_norm, jc) if c_norm else 100

            if st >= 75 and sc >= 70:
                matches += 1
                if not best:
                    best = {
                        "title": j.get("title"),
                        "company": (j.get("company") or {}).get("display_name"),
                        "url": j.get("redirect_url"),
                        "source": "adzuna",
                    }

    return {"found": matches > 0, "matches": matches, "sample": best}


# ------------ emails + keywords ------------

FREE_EMAILS = {
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

DISPOSABLE_HINTS = {
    "tempmail",
    "10minutemail",
    "mailinator",
    "guerrillamail",
}

SCAM_KEYWORDS = [
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


def extract_emails(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text or "")


def check_email_domain(email: str, claimed_company: str | None = None) -> dict:
    user, domain = email.split("@", 1)
    d = domain.lower()
    signals = []

    if d in FREE_EMAILS:
        signals.append("free_domain")

    if any(h in d for h in DISPOSABLE_HINTS):
        signals.append("disposable_like")

    if claimed_company:
        ck = re.sub(r"[^a-z0-9]", "", (claimed_company or "").lower())
        dk = re.sub(r"[^a-z0-9]", "", d.split(".")[0])
        if ck and ck not in dk:
            signals.append("company_domain_mismatch")

    return {"email": email, "domain": d, "signals": signals}


def detect_keywords(text: str) -> list[str]:
    t = _n(text)
    return [kw for kw in SCAM_KEYWORDS if kw in t]


# ------------ scoring / final outputs ------------

def compute_confidence(
    model_fake_prob: float,
    api_found: bool,
    email_checks: list[dict],
    kw_hits: list[str],
) -> dict:
    """
    Combine model score + Adzuna + emails + keywords
    into final real/fake percentages.
    """
    fake_score = model_fake_prob
    reasons: list[str] = []

    if api_found:
        fake_score *= 0.8
        reasons.append("Found on public job index (Adzuna)")

    for e in email_checks:
        for s in e["signals"]:
            if s == "free_domain":
                fake_score = min(1.0, fake_score + 0.10)
                reasons.append(f"Free email domain: {e['domain']}")
            if s == "disposable_like":
                fake_score = min(1.0, fake_score + 0.20)
                reasons.append(f"Disposable-like email: {e['domain']}")
            if s == "company_domain_mismatch":
                fake_score = min(1.0, fake_score + 0.15)
                reasons.append(f"Email domain does not match company: {e['domain']}")

    if kw_hits:
        bump = min(0.25, 0.05 * len(kw_hits))
        fake_score = min(1.0, fake_score + bump)
        reasons.append("Suspicious phrases: " + ", ".join(kw_hits[:5]))

    fake_pct = round(fake_score * 100, 1)
    real_pct = round((1 - fake_score) * 100, 1)

    return {
        "real_pct": real_pct,
        "fake_pct": fake_pct,
        "reasons": reasons,
    }


# ------------ one-call helper for the app ------------

def verify_all(
    title: str,
    description: str,
    company: str = "",
    location: str = "",
) -> dict:
    """
    Run all checks in one call so the Streamlit app
    can just call verify_all(...) once.
    """
    guessed_company = _guess_company(title, description, company)
    api = adzuna_search(title, company=guessed_company, where=location)

    emails = extract_emails(description)
    email_checks = [check_email_domain(e, claimed_company=guessed_company) for e in emails]

    kw_hits = detect_keywords(f"{title} {description}")

    return {
        "api": api,
        "emails": email_checks,
        "kw_hits": kw_hits,
    }