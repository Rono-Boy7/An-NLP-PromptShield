"""
src/promptshield/detection/tfidf_classifier.py

Purpose
-------
TF-IDF + Logistic Regression classifier for prompt injection detection.

Why this file matters
---------------------
Before training heavier transformer models, PromptShield needs a strong classical
ML baseline.

A TF-IDF classifier is useful because it is:

- fast to train
- easy to evaluate
- cheap to run
- explainable compared to deep models
- a strong baseline for text classification problems

This baseline gives us a serious reference point. Later, the transformer model
must prove that it performs better than this simpler model.

What this module does
---------------------
1. Build a TF-IDF + Logistic Regression pipeline
2. Train on labeled prompt injection examples
3. Predict suspicious probabilities
4. Convert probabilities into labels
5. Evaluate predictions with security-focused metrics
6. Save and load trained model artifacts

Design scope
------------
This file defines the reusable classifier.

It does not download data, prepare datasets, or run full training experiments.
Those responsibilities belong to scripts such as scripts/train_tfidf.py.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from promptshield.core.labels import InjectionLabel
from promptshield.core.types import ModelPrediction
from promptshield.evaluation.metrics import (
    BinaryMetrics,
    binary_classification_metrics,
    labels_from_probabilities,
    validate_binary_labels,
)
from promptshield.utils.config import ModelConfig


@dataclass(frozen=True, slots=True)
class TfidfClassifierConfig:
    """Config for the TF-IDF baseline classifier."""

    model_name: str = "tfidf_logistic_regression"
    max_features: int = 30000
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 2
    max_df: float = 0.95
    lowercase: bool = True
    class_weight: str | None = "balanced"
    random_seed: int = 42
    max_iter: int = 1000

    def __post_init__(self) -> None:
        if self.max_features < 1:
            raise ValueError("max_features must be at least 1.")

        if self.ngram_range[0] < 1:
            raise ValueError("ngram_range minimum must be at least 1.")

        if self.ngram_range[1] < self.ngram_range[0]:
            raise ValueError("ngram_range maximum must be >= minimum.")

        if self.min_df < 1:
            raise ValueError("min_df must be at least 1.")

        if self.max_df <= 0.0 or self.max_df > 1.0:
            raise ValueError("max_df must be in the range (0.0, 1.0].")

        if self.max_iter < 1:
            raise ValueError("max_iter must be at least 1.")


class TfidfPromptInjectionClassifier:
    """Trainable TF-IDF baseline for prompt injection detection."""

    def __init__(
        self,
        config: TfidfClassifierConfig | None = None,
        *,
        pipeline: Pipeline | None = None,
        is_fitted: bool = False,
    ) -> None:
        self.config = config or TfidfClassifierConfig()
        self.pipeline = pipeline
        self.is_fitted = is_fitted

    @classmethod
    def from_model_config(cls, config: ModelConfig) -> TfidfPromptInjectionClassifier:
        """Create classifier settings from configs/model.yaml."""

        baseline = config.baseline

        return cls(
            config=TfidfClassifierConfig(
                model_name=baseline.name,
                max_features=baseline.max_features,
                ngram_range=baseline.ngram_range,
                min_df=baseline.min_df,
                max_df=baseline.max_df,
                lowercase=baseline.lowercase,
                class_weight=baseline.class_weight,
                random_seed=baseline.random_seed,
            )
        )

    def fit(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
    ) -> TfidfPromptInjectionClassifier:
        """Train the classifier."""

        train_texts = validate_texts(texts, name="texts")
        train_labels = validate_labels(labels)

        if len(train_texts) != len(train_labels):
            raise ValueError(
                f"texts and labels must have the same length. "
                f"Got {len(train_texts)} and {len(train_labels)}."
            )

        pipeline = build_tfidf_pipeline(self.config)
        pipeline.fit(train_texts, train_labels)

        self.pipeline = pipeline
        self.is_fitted = True

        return self

    def predict_suspicious_probabilities(self, texts: Sequence[str]) -> list[float]:
        """Predict probability that each text is suspicious."""

        self._ensure_fitted()
        prediction_texts = validate_texts(texts, name="texts")

        probabilities = self.pipeline.predict_proba(prediction_texts)
        suspicious_index = self._class_index(InjectionLabel.SUSPICIOUS.value)

        return [float(row[suspicious_index]) for row in probabilities]

    def predict_labels(
        self,
        texts: Sequence[str],
        *,
        threshold: float = 0.5,
    ) -> list[int]:
        """Predict binary labels using a suspicious probability threshold."""

        probabilities = self.predict_suspicious_probabilities(texts)

        return labels_from_probabilities(probabilities, threshold=threshold)

    def predict(
        self,
        texts: Sequence[str],
        *,
        threshold: float = 0.5,
    ) -> list[ModelPrediction]:
        """Return structured model predictions."""

        suspicious_probabilities = self.predict_suspicious_probabilities(texts)
        predicted_labels = labels_from_probabilities(
            suspicious_probabilities,
            threshold=threshold,
        )

        predictions = []

        for label, suspicious_probability in zip(
            predicted_labels,
            suspicious_probabilities,
            strict=True,
        ):
            predictions.append(
                ModelPrediction(
                    label=InjectionLabel(label),
                    suspicious_probability=suspicious_probability,
                    benign_probability=1.0 - suspicious_probability,
                    model_name=self.config.model_name,
                )
            )

        return predictions

    def evaluate(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        *,
        threshold: float = 0.5,
    ) -> BinaryMetrics:
        """Evaluate classifier predictions against true labels."""

        true_labels = validate_labels(labels)
        predicted_labels = self.predict_labels(texts, threshold=threshold)

        return binary_classification_metrics(true_labels, predicted_labels)

    def save(self, path: str | Path) -> Path:
        """Save trained classifier artifact."""

        self._ensure_fitted()

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "config": asdict(self.config),
            "pipeline": self.pipeline,
            "model_name": self.config.model_name,
        }

        joblib.dump(payload, output_path)

        return output_path

    @classmethod
    def load(cls, path: str | Path) -> TfidfPromptInjectionClassifier:
        """Load trained classifier artifact."""

        model_path = Path(path)

        if not model_path.is_file():
            raise FileNotFoundError(f"Model artifact not found: {model_path}")

        payload = joblib.load(model_path)

        if not isinstance(payload, dict):
            raise TypeError("Invalid TF-IDF model artifact. Expected dictionary payload.")

        config_data = payload.get("config")
        pipeline = payload.get("pipeline")

        if not isinstance(config_data, dict):
            raise TypeError("Invalid TF-IDF model artifact. Missing config dictionary.")

        if not isinstance(pipeline, Pipeline):
            raise TypeError("Invalid TF-IDF model artifact. Missing sklearn Pipeline.")

        return cls(
            config=TfidfClassifierConfig(**config_data),
            pipeline=pipeline,
            is_fitted=True,
        )

    def _ensure_fitted(self) -> None:
        """Raise a clear error if the classifier has not been trained."""

        if not self.is_fitted or self.pipeline is None:
            raise RuntimeError("TF-IDF classifier is not fitted yet.")

    def _class_index(self, label: int) -> int:
        """Return the predict_proba column index for a class label."""

        self._ensure_fitted()

        classifier = self.pipeline.named_steps["classifier"]
        classes = list(classifier.classes_)

        if label not in classes:
            raise ValueError(f"Trained classifier does not contain class label {label}.")

        return classes.index(label)


def build_tfidf_pipeline(config: TfidfClassifierConfig) -> Pipeline:
    """Build the sklearn TF-IDF + Logistic Regression pipeline."""

    vectorizer = TfidfVectorizer(
        max_features=config.max_features,
        ngram_range=config.ngram_range,
        min_df=config.min_df,
        max_df=config.max_df,
        lowercase=config.lowercase,
        strip_accents="unicode",
        sublinear_tf=True,
    )

    classifier = LogisticRegression(
        class_weight=config.class_weight,
        max_iter=config.max_iter,
        random_state=config.random_seed,
        solver="liblinear",
    )

    return Pipeline(
        steps=[
            ("vectorizer", vectorizer),
            ("classifier", classifier),
        ]
    )


def validate_texts(texts: Sequence[str], *, name: str) -> list[str]:
    """Validate a text sequence."""

    if isinstance(texts, str):
        raise TypeError(f"{name} must be a sequence of strings, not a single string.")

    cleaned_texts = []

    for index, text in enumerate(texts):
        if not isinstance(text, str):
            raise TypeError(f"{name}[{index}] must be a string.")

        cleaned_texts.append(text)

    if not cleaned_texts:
        raise ValueError(f"{name} cannot be empty.")

    return cleaned_texts


def validate_labels(labels: Sequence[int]) -> list[int]:
    """Validate binary labels."""

    label_list = [int(label) for label in labels]

    if not label_list:
        raise ValueError("labels cannot be empty.")

    validate_binary_labels(label_list, name="labels")

    return label_list


def top_weighted_features(
    classifier: TfidfPromptInjectionClassifier,
    *,
    top_n: int = 20,
) -> dict[str, list[tuple[str, float]]]:
    """Return top positive and negative TF-IDF features."""

    classifier._ensure_fitted()

    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    vectorizer = classifier.pipeline.named_steps["vectorizer"]
    logistic_model = classifier.pipeline.named_steps["classifier"]

    feature_names = np.array(vectorizer.get_feature_names_out())
    coefficients = logistic_model.coef_[0]

    positive_indices = np.argsort(coefficients)[-top_n:][::-1]
    negative_indices = np.argsort(coefficients)[:top_n]

    return {
        "suspicious": [
            (str(feature_names[index]), float(coefficients[index])) for index in positive_indices
        ],
        "benign": [
            (str(feature_names[index]), float(coefficients[index])) for index in negative_indices
        ],
    }


def prediction_to_dict(prediction: ModelPrediction) -> dict[str, Any]:
    """Convert a ModelPrediction into a plain dictionary."""

    return {
        "label": int(prediction.label),
        "label_name": prediction.label.name.lower(),
        "suspicious_probability": prediction.suspicious_probability,
        "benign_probability": prediction.benign_probability,
        "model_name": prediction.model_name,
        "model_version": prediction.model_version,
    }


__all__ = [
    "TfidfClassifierConfig",
    "TfidfPromptInjectionClassifier",
    "build_tfidf_pipeline",
    "prediction_to_dict",
    "top_weighted_features",
    "validate_labels",
    "validate_texts",
]
