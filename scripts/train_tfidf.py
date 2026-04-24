"""
scripts/train_tfidf.py

Purpose
-------
Train the TF-IDF + Logistic Regression prompt injection classifier.

Why this file matters
---------------------
This is PromptShield's first trainable model.

Before we add transformer models, we need a strong classical ML baseline that is:

- fast
- explainable
- cheap to run
- easy to evaluate
- useful as a production fallback

This script trains the baseline on the processed Deepset dataset and saves the
model plus evaluation artifacts.

What this script does
---------------------
1. Load processed train, validation, and test JSONL files
2. Train the TF-IDF classifier
3. Evaluate at the default threshold
4. Find the best validation threshold by F1
5. Re-evaluate validation and test at the best threshold
6. Save model, metrics, feature weights, and test predictions

Design scope
------------
This script trains only the TF-IDF baseline.

It does not train transformers, run the API, or evaluate full agent behavior.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from promptshield.detection.tfidf_classifier import (
    TfidfPromptInjectionClassifier,
    top_weighted_features,
)
from promptshield.evaluation.metrics import (
    best_threshold_by_f1,
    metrics_summary,
    threshold_sweep,
)
from promptshield.utils.config import load_app_config, load_model_config
from promptshield.utils.logging import configure_logging, get_logger
from promptshield.utils.paths import ensure_project_dir, project_path, require_project_file


@dataclass(frozen=True, slots=True)
class TfidfTrainingArtifacts:
    """Paths created by the TF-IDF training script."""

    model_path: str
    metrics_path: str
    features_path: str
    predictions_path: str


def main() -> None:
    """Train and evaluate the TF-IDF baseline."""

    configure_logging()
    log = get_logger(component="train_tfidf")

    app_config = load_app_config()
    model_config = load_model_config()

    processed_dir = project_path(app_config.paths.processed_data_dir)
    metrics_dir = ensure_project_dir(model_config.artifacts.metrics_dir)
    predictions_dir = ensure_project_dir(model_config.artifacts.predictions_dir)
    model_dir = ensure_project_dir(model_config.artifacts.baseline_model_dir)

    train_path = require_project_file(processed_dir, "deepset_train.jsonl")
    val_path = require_project_file(processed_dir, "deepset_val.jsonl")
    test_path = require_project_file(processed_dir, "deepset_test.jsonl")

    log.info("Loading processed datasets.")
    train_df = read_jsonl(train_path)
    val_df = read_jsonl(val_path)
    test_df = read_jsonl(test_path)

    train_texts, train_labels = frame_to_texts_labels(train_df)
    val_texts, val_labels = frame_to_texts_labels(val_df)
    test_texts, test_labels = frame_to_texts_labels(test_df)

    log.info("Training TF-IDF baseline.")
    classifier = TfidfPromptInjectionClassifier.from_model_config(model_config)
    classifier.fit(train_texts, train_labels)

    default_threshold = model_config.thresholds.suspicious_probability

    train_metrics = classifier.evaluate(
        train_texts,
        train_labels,
        threshold=default_threshold,
    )
    val_metrics = classifier.evaluate(
        val_texts,
        val_labels,
        threshold=default_threshold,
    )
    test_metrics = classifier.evaluate(
        test_texts,
        test_labels,
        threshold=default_threshold,
    )

    val_probabilities = classifier.predict_suspicious_probabilities(val_texts)
    test_probabilities = classifier.predict_suspicious_probabilities(test_texts)

    threshold_results = threshold_sweep(val_labels, val_probabilities)
    best_threshold_result = best_threshold_by_f1(val_labels, val_probabilities)
    best_threshold = best_threshold_result.threshold

    val_best_metrics = classifier.evaluate(
        val_texts,
        val_labels,
        threshold=best_threshold,
    )
    test_best_metrics = classifier.evaluate(
        test_texts,
        test_labels,
        threshold=best_threshold,
    )

    model_path = model_dir / "model.joblib"
    metrics_path = metrics_dir / "tfidf_metrics.json"
    features_path = metrics_dir / "tfidf_top_features.json"
    predictions_path = predictions_dir / "tfidf_test_predictions.csv"

    classifier.save(model_path)

    metrics_payload = build_metrics_payload(
        model_name=model_config.baseline.name,
        default_threshold=default_threshold,
        best_threshold=best_threshold,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        val_best_metrics=val_best_metrics,
        test_best_metrics=test_best_metrics,
        threshold_results=threshold_results,
    )

    write_json(metrics_payload, metrics_path)
    write_json(top_weighted_features(classifier, top_n=30), features_path)
    write_test_predictions(
        test_df=test_df,
        probabilities=test_probabilities,
        threshold=best_threshold,
        output_path=predictions_path,
    )

    artifacts = TfidfTrainingArtifacts(
        model_path=relative_path(model_path),
        metrics_path=relative_path(metrics_path),
        features_path=relative_path(features_path),
        predictions_path=relative_path(predictions_path),
    )

    log.success("TF-IDF baseline training complete.")
    log.info("Model: {}", artifacts.model_path)
    log.info("Metrics: {}", artifacts.metrics_path)
    log.info("Feature weights: {}", artifacts.features_path)
    log.info("Predictions: {}", artifacts.predictions_path)
    log.info("Default threshold: {:.2f}", default_threshold)
    log.info("Best validation threshold: {:.2f}", best_threshold)
    log.info("Validation F1 at best threshold: {:.4f}", val_best_metrics.f1)
    log.info("Test F1 at best threshold: {:.4f}", test_best_metrics.f1)
    log.info("Test recall at best threshold: {:.4f}", test_best_metrics.recall)
    log.info(
        "Test false negative rate at best threshold: {:.4f}",
        test_best_metrics.false_negative_rate,
    )


def read_jsonl(path: Path) -> pd.DataFrame:
    """Read a processed JSONL dataset file."""

    df = pd.read_json(path, lines=True)

    required_columns = {"id", "text", "label"}
    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {sorted(missing_columns)}")

    return df


def frame_to_texts_labels(df: pd.DataFrame) -> tuple[list[str], list[int]]:
    """Convert a dataframe into model-ready texts and labels."""

    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()

    return texts, labels


def build_metrics_payload(
    *,
    model_name: str,
    default_threshold: float,
    best_threshold: float,
    train_metrics: Any,
    val_metrics: Any,
    test_metrics: Any,
    val_best_metrics: Any,
    test_best_metrics: Any,
    threshold_results: Any,
) -> dict[str, Any]:
    """Build the metrics JSON payload."""

    return {
        "model_name": model_name,
        "default_threshold": default_threshold,
        "best_validation_threshold": best_threshold,
        "default_threshold_metrics": {
            "train": metrics_summary(train_metrics),
            "validation": metrics_summary(val_metrics),
            "test": metrics_summary(test_metrics),
        },
        "best_threshold_metrics": {
            "validation": metrics_summary(val_best_metrics),
            "test": metrics_summary(test_best_metrics),
        },
        "validation_threshold_sweep": [
            {
                "threshold": result.threshold,
                "metrics": metrics_summary(result.metrics),
            }
            for result in threshold_results
        ],
        "notes": [
            "False negatives are security-sensitive because risky content was missed.",
            "False positives affect utility because safe content may be over-flagged.",
            "Best threshold is selected on validation F1 only, then reported on test.",
        ],
    }


def write_test_predictions(
    *,
    test_df: pd.DataFrame,
    probabilities: list[float],
    threshold: float,
    output_path: Path,
) -> None:
    """Save test predictions without storing raw text."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_df = pd.DataFrame(
        {
            "id": test_df["id"].astype(str),
            "label": test_df["label"].astype(int),
            "suspicious_probability": probabilities,
        }
    )

    output_df["prediction"] = (
        output_df["suspicious_probability"] >= threshold
    ).astype(int)
    output_df["threshold"] = threshold

    output_df.to_csv(output_path, index=False)


def write_json(payload: dict[str, Any], path: Path) -> None:
    """Write a JSON file with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(make_json_safe(payload), file, indent=2, sort_keys=True)


def make_json_safe(value: Any) -> Any:
    """Convert dataclasses and nested objects into JSON-safe values."""

    if hasattr(value, "__dataclass_fields__"):
        return make_json_safe(asdict(value))

    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}

    if isinstance(value, list | tuple):
        return [make_json_safe(item) for item in value]

    return value


def relative_path(path: Path) -> str:
    """Return a project-relative path for cleaner logs."""

    return str(path.relative_to(project_path()))


if __name__ == "__main__":
    main()