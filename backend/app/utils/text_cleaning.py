import re
import string


def clean_text(text: str) -> str:
    """
    Basic text cleaning for job postings.
    - Lowercase
    - Remove URLs, emails
    - Remove punctuation and extra spaces
    """
    if not text:
        return ""

    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove emails
    text = re.sub(r"\S+@\S+\.\S+", " ", text)

    # Remove numbers
    text = re.sub(r"\d+", " ", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text