from pathlib import Path
import json
import re

import joblib
from scipy.sparse import hstack


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

with open(MODEL_DIR / "feature_config.json", "r") as f:
    FEATURE_CONFIG = json.load(f)

THRESHOLD = 0.30


def load_artifacts():
    """Load the trained vectorizer, scaler, and classifier."""
    vectorizer = joblib.load(
        MODEL_DIR / "digit_count_tfidf_vectorizer.pkl"
    )

    scaler = joblib.load(
        MODEL_DIR / "digit_count_scaler.pkl"
    )

    model = joblib.load(
        MODEL_DIR / "digit_count_logistic_regression_model.pkl"
    )

    return vectorizer, scaler, model


def extract_digit_count(text: str) -> int:
    """Return the number of digits present in the message."""
    return sum(character.isdigit() for character in text)


def predict_spam(text: str) -> dict:
    """
    Predict whether an SMS message is spam.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    if not text.strip():
        raise ValueError("text must not be empty")

    vectorizer, scaler, model = load_artifacts()

    text_features = vectorizer.transform([text])

    digit_count = extract_digit_count(text)

    numeric_features = scaler.transform([[digit_count]])

    combined_features = hstack(
        [text_features, numeric_features]
    )

    spam_probability = model.predict_proba(
        combined_features
    )[0][1]

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