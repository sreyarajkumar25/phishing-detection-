#!/usr/bin/env python3
"""Classify emails as Phishing or Safe using the trained model."""

import argparse
import sys
from pathlib import Path

import features  # noqa: F401 - required for joblib unpickling
import joblib
import numpy as np

MODEL_DIR = Path(__file__).parent / "models"


def load_model():
    features_path = MODEL_DIR / "features.joblib"
    classifier_path = MODEL_DIR / "classifier.joblib"
    if not features_path.exists() or not classifier_path.exists():
        print("Model not found. Run: python train_model.py")
        sys.exit(1)
    return joblib.load(features_path), joblib.load(classifier_path)


def predict_email(text: str):
    features, classifier = load_model()
    X = features.transform([text])
    prediction = classifier.predict(X)[0]
    label = "Phishing" if prediction == 1 else "Safe"

    if hasattr(classifier, "decision_function"):
        score = classifier.decision_function(X)[0]
        confidence = 1 / (1 + np.exp(-abs(score)))
    else:
        confidence = 1.0

    return label, confidence


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict if an email is phishing")
    parser.add_argument("--text", help="Email text to classify")
    parser.add_argument("--file", type=Path, help="Read email from file")
    args = parser.parse_args()

    if args.file:
        text = args.file.read_text(encoding="utf-8")
    elif args.text:
        text = args.text
    else:
        print("Enter email content (end with Ctrl+D or empty line twice):")
        lines = []
        try:
            while True:
                line = input()
                lines.append(line)
        except EOFError:
            text = "\n".join(lines)

    label, confidence = predict_email(text)
    print(f"\nClassification: {label}")
    print(f"Confidence:     {confidence:.2%}")


if __name__ == "__main__":
    main()
