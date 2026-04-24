"""
src/promptshield/evaluation/metrics.py

Purpose
-------
Evaluation metrics for PromptShield classifiers and scanners.

Why this file matters
---------------------
PromptShield is a security-focused NLP project, so accuracy alone is not enough.

For prompt injection defense, we care about:

- false negatives: risky content incorrectly allowed
- false positives: safe content incorrectly blocked
- precision: how trustworthy positive detections are
- recall: how many risky samples we catch
- F1: balance between precision and recall
- threshold behavior: how changing the risk threshold affects safety

This module gives us clean reusable metrics for baseline models, transformer
models, rule scanners, and later API/report outputs.

What this module does
---------------------
1. Validate binary labels
2. Compute confusion matrix counts
3. Compute precision, recall, F1, specificity, and error rates
4. Convert model probabilities into labels
5. Sweep thresholds to compare safety/utility tradeoffs

Design scope
------------
This file does not train models or scan text.

It only evaluates predictions that were already produced elsewhere.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ConfusionCounts:
    """Binary confusion matrix counts."""

    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int

    @property
    def total(self) -> int:
        """Return total number of evaluated samples."""

        return self.true_positive + self.true_negative + self.false_positive + self.false_negative


@dataclass(frozen=True, slots=True)
class BinaryMetrics:
    """Core binary classification metrics."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    specificity: float
    false_positive_rate: float
    false_negative_rate: float
    balanced_accuracy: float
    positive_rate: float
    predicted_positive_rate: float
    support: int
    counts: ConfusionCounts

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to a plain dictionary."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class ThresholdMetrics:
    """Metrics computed at a specific probability threshold."""

    threshold: float
    metrics: BinaryMetrics

    def to_dict(self) -> dict[str, Any]:
        """Convert threshold metrics to a plain dictionary."""

        return {
            "threshold": self.threshold,
            **self.metrics.to_dict(),
        }


def safe_divide(numerator: float, denominator: float) -> float:
    """Divide safely and return 0.0 when denominator is zero."""

    if denominator == 0:
        return 0.0

    return numerator / denominator


def validate_binary_labels(labels: list[int], *, name: str) -> None:
    """Validate that labels only contain 0 or 1."""

    invalid_values = sorted({label for label in labels if label not in {0, 1}})

    if invalid_values:
        raise ValueError(f"{name} contains non-binary labels: {invalid_values}")


def validate_same_length(y_true: list[int], y_pred: list[int]) -> None:
    """Validate that true and predicted labels have the same length."""

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have the same length. Got {len(y_true)} and {len(y_pred)}."
        )

    if not y_true:
        raise ValueError("y_true and y_pred cannot be empty.")


def confusion_counts(y_true: list[int], y_pred: list[int]) -> ConfusionCounts:
    """Compute binary confusion matrix counts."""

    validate_same_length(y_true, y_pred)
    validate_binary_labels(y_true, name="y_true")
    validate_binary_labels(y_pred, name="y_pred")

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for actual, predicted in zip(y_true, y_pred, strict=True):
        if actual == 1 and predicted == 1:
            true_positive += 1
        elif actual == 0 and predicted == 0:
            true_negative += 1
        elif actual == 0 and predicted == 1:
            false_positive += 1
        elif actual == 1 and predicted == 0:
            false_negative += 1

    return ConfusionCounts(
        true_positive=true_positive,
        true_negative=true_negative,
        false_positive=false_positive,
        false_negative=false_negative,
    )


def binary_classification_metrics(y_true: list[int], y_pred: list[int]) -> BinaryMetrics:
    """Compute binary classification metrics."""

    counts = confusion_counts(y_true, y_pred)

    tp = counts.true_positive
    tn = counts.true_negative
    fp = counts.false_positive
    fn = counts.false_negative
    total = counts.total

    accuracy = safe_divide(tp + tn, total)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    f1 = safe_divide(2 * precision * recall, precision + recall)

    false_positive_rate = safe_divide(fp, fp + tn)
    false_negative_rate = safe_divide(fn, fn + tp)
    balanced_accuracy = (recall + specificity) / 2

    positive_rate = safe_divide(sum(y_true), total)
    predicted_positive_rate = safe_divide(sum(y_pred), total)

    return BinaryMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        specificity=specificity,
        false_positive_rate=false_positive_rate,
        false_negative_rate=false_negative_rate,
        balanced_accuracy=balanced_accuracy,
        positive_rate=positive_rate,
        predicted_positive_rate=predicted_positive_rate,
        support=total,
        counts=counts,
    )


def labels_from_probabilities(
    probabilities: list[float],
    *,
    threshold: float = 0.5,
) -> list[int]:
    """Convert suspicious-class probabilities into binary labels."""

    validate_threshold(threshold)

    return [1 if probability >= threshold else 0 for probability in probabilities]


def validate_threshold(threshold: float) -> None:
    """Validate a probability threshold."""

    if threshold < 0.0 or threshold > 1.0:
        raise ValueError(f"threshold must be between 0.0 and 1.0. Got {threshold}.")


def threshold_sweep(
    y_true: list[int],
    probabilities: list[float],
    *,
    thresholds: list[float] | None = None,
) -> list[ThresholdMetrics]:
    """Evaluate model performance across multiple thresholds."""

    if len(y_true) != len(probabilities):
        raise ValueError(
            f"y_true and probabilities must have the same length. "
            f"Got {len(y_true)} and {len(probabilities)}."
        )

    validate_binary_labels(y_true, name="y_true")

    active_thresholds = thresholds or [
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
    ]

    results = []

    for threshold in active_thresholds:
        validate_threshold(threshold)
        predictions = labels_from_probabilities(probabilities, threshold=threshold)
        metrics = binary_classification_metrics(y_true, predictions)

        results.append(
            ThresholdMetrics(
                threshold=threshold,
                metrics=metrics,
            )
        )

    return results


def best_threshold_by_f1(
    y_true: list[int],
    probabilities: list[float],
    *,
    thresholds: list[float] | None = None,
) -> ThresholdMetrics:
    """Return the threshold result with the highest F1 score."""

    results = threshold_sweep(
        y_true,
        probabilities,
        thresholds=thresholds,
    )

    return max(
        results,
        key=lambda result: (
            result.metrics.f1,
            result.metrics.recall,
            -result.metrics.false_positive_rate,
        ),
    )


def metrics_summary(metrics: BinaryMetrics) -> dict[str, float | int]:
    """Return a compact metrics summary for logs and reports."""

    return {
        "accuracy": round(metrics.accuracy, 4),
        "precision": round(metrics.precision, 4),
        "recall": round(metrics.recall, 4),
        "f1": round(metrics.f1, 4),
        "false_positive_rate": round(metrics.false_positive_rate, 4),
        "false_negative_rate": round(metrics.false_negative_rate, 4),
        "balanced_accuracy": round(metrics.balanced_accuracy, 4),
        "support": metrics.support,
    }


__all__ = [
    "BinaryMetrics",
    "ConfusionCounts",
    "ThresholdMetrics",
    "best_threshold_by_f1",
    "binary_classification_metrics",
    "confusion_counts",
    "labels_from_probabilities",
    "metrics_summary",
    "safe_divide",
    "threshold_sweep",
    "validate_binary_labels",
    "validate_same_length",
    "validate_threshold",
]
