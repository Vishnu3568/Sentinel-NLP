from pathlib import Path
import json
import re

import joblib


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"


# Load configuration
with open(MODEL_DIR / "feature_config.json", "r") as f:
    FEATURE_CONFIG = json.load(f)


# Load trained artifacts
VECTORIZER = joblib.load(
    MODEL_DIR / "digit_count_tfidf_vectorizer.pkl"
)

SCALER = joblib.load(
    MODEL_DIR / "digit_count_scaler.pkl"
)

MODEL = joblib.load(
    MODEL_DIR / "digit_count_logistic_regression_model.pkl"
)


# Selected classification threshold from our experiments
THRESHOLD = 0.30


def extract_digit_count(text: str) -> int:
    """Return the number of digits present in the message."""
    return sum(character.isdigit() for character in text)


def predict_spam(text: str) -> dict:
    """
    Predict whether an SMS message is spam.

    Returns:
        Dictionary containing the predicted label,
        spam probability, and threshold used.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    if not text.strip():
        raise ValueError("text must not be empty")

    # TF-IDF features
    text_features = VECTORIZER.transform([text])

    # Numeric feature
    digit_count = extract_digit_count(text)

    numeric_features = SCALER.transform([[digit_count]])

    # Combine sparse TF-IDF features with numeric feature
    from scipy.sparse import hstack

    combined_features = hstack(
        [text_features, numeric_features]
    )

    # Spam probability
    spam_probability = MODEL.predict_proba(
        combined_features
    )[0][1]

    # Apply selected threshold
    label = (
        "spam"
        if spam_probability >= THRESHOLD
        else "ham"
    )

    return {
        "label": label,
        "spam_probability": float(spam_probability),
        "threshold": THRESHOLD,
    }