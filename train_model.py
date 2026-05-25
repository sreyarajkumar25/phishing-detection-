#!/usr/bin/env python3
"""
Phishing Email Detection Model
Trains a Scikit-learn classifier on email text and URL features.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from features import extract_keyword_score, extract_special_char_ratio, extract_url_count
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC

DATA_PATH = Path(__file__).parent / "data" / "emails.csv"
MODEL_DIR = Path(__file__).parent / "models"


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"text", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {required}")
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].str.strip().str.lower()
    df = df[df["label"].isin(["phishing", "safe"])]
    return df


def build_pipeline() -> tuple[FeatureUnion, LinearSVC]:
    text_vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )

    feature_union = FeatureUnion([
        ("tfidf", text_vectorizer),
        ("url_count", FunctionTransformer(extract_url_count, validate=False)),
        ("keyword_score", FunctionTransformer(extract_keyword_score, validate=False)),
        ("special_ratio", FunctionTransformer(extract_special_char_ratio, validate=False)),
    ])

    classifier = LinearSVC(class_weight="balanced", max_iter=3000, random_state=42)
    return feature_union, classifier


def plot_confusion_matrix(cm: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    classes = ["Safe", "Phishing"]
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def train(data_path: Path = DATA_PATH) -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    df = load_dataset(data_path)

    X = df["text"].values
    y = (df["label"] == "phishing").astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    features, classifier = build_pipeline()
    X_train_features = features.fit_transform(X_train)
    classifier.fit(X_train_features, y_train)

    X_test_features = features.transform(X_test)
    y_pred = classifier.predict(X_test_features)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 50)
    print("  PHISHING EMAIL DETECTION - TRAINING RESULTS")
    print("=" * 50)
    print(f"\nDataset size:     {len(df)} emails")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples:     {len(X_test)}")
    print(f"\nAccuracy:  {accuracy:.2%}")
    print(f"F1 Score:  {f1:.2%}")
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"                 Safe  Phishing")
    print(f"  Actual Safe    {cm[0][0]:4d}  {cm[0][1]:6d}")
    print(f"  Actual Phishing {cm[1][0]:4d}  {cm[1][1]:6d}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Safe', 'Phishing'])}")

    joblib.dump(features, MODEL_DIR / "features.joblib")
    joblib.dump(classifier, MODEL_DIR / "classifier.joblib")
    plot_confusion_matrix(cm, MODEL_DIR / "confusion_matrix.png")

    print(f"\nModel saved to: {MODEL_DIR}")
    print(f"Confusion matrix plot: {MODEL_DIR / 'confusion_matrix.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train phishing detection model")
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Path to CSV dataset")
    args = parser.parse_args()
    train(args.data)


if __name__ == "__main__":
    main()
