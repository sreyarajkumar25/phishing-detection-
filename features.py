"""Shared feature extraction functions for phishing detection."""

import re

import numpy as np


def _iter_texts(texts) -> list:
    if isinstance(texts, str):
        return [texts]
    return list(texts)


def extract_url_count(texts) -> np.ndarray:
    results = []
    for text in _iter_texts(texts):
        urls = re.findall(r"https?://\S+|www\.\S+", str(text), re.IGNORECASE)
        results.append([len(urls)])
    return np.array(results)


def extract_keyword_score(texts) -> np.ndarray:
    phishing_keywords = [
        "urgent", "verify", "account", "suspended", "click here", "password",
        "bank", "login", "winner", "prize", "free", "limited time", "act now",
        "confirm", "update", "security alert", "unusual activity", "wire transfer",
        "invoice", "paypal", "bitcoin", "credential", "expire",
    ]
    results = []
    for text in _iter_texts(texts):
        text_lower = str(text).lower()
        score = sum(1 for kw in phishing_keywords if kw in text_lower)
        results.append([score])
    return np.array(results)


def extract_special_char_ratio(texts) -> np.ndarray:
    results = []
    for text in _iter_texts(texts):
        text = str(text)
        if not text:
            results.append([0.0])
        else:
            special = sum(1 for c in text if not c.isalnum() and not c.isspace())
            results.append([special / len(text)])
    return np.array(results)
