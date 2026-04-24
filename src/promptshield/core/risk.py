# PromptShield/src/promptshield/core/risk.py
#
# Risk scoring helpers used by scanners, classifiers, and policy logic.

from __future__ import annotations

from dataclasses import dataclass

from promptshield.core.labels import RiskDecision, RiskLevel


@dataclass(frozen=True, slots=True)
class RiskBand:
    """Score range mapped to a risk level and policy decision."""

    name: str
    min_score: float
    max_score: float
    level: RiskLevel
    decision: RiskDecision


DEFAULT_RISK_BANDS: tuple[RiskBand, ...] = (
    RiskBand(
        name="allow",
        min_score=0.0,
        max_score=0.29,
        level=RiskLevel.LOW,
        decision=RiskDecision.ALLOW,
    ),
    RiskBand(
        name="monitor",
        min_score=0.30,
        max_score=0.59,
        level=RiskLevel.MEDIUM,
        decision=RiskDecision.ALLOW_WITH_WARNING,
    ),
    RiskBand(
        name="sanitize",
        min_score=0.60,
        max_score=0.84,
        level=RiskLevel.HIGH,
        decision=RiskDecision.SANITIZE,
    ),
    RiskBand(
        name="block",
        min_score=0.85,
        max_score=1.0,
        level=RiskLevel.CRITICAL,
        decision=RiskDecision.BLOCK,
    ),
)


def clamp_score(score: float) -> float:
    """Keep risk scores inside the 0.0 to 1.0 range."""

    return max(0.0, min(1.0, float(score)))


def risk_band_for_score(
    score: float,
    bands: tuple[RiskBand, ...] = DEFAULT_RISK_BANDS,
) -> RiskBand:
    """Return the matching risk band for a score."""

    normalized_score = clamp_score(score)

    for band in bands:
        if band.min_score <= normalized_score <= band.max_score:
            return band

    return bands[-1]


def risk_level_for_score(score: float) -> RiskLevel:
    """Return a human-readable risk level for a score."""

    return risk_band_for_score(score).level


def decision_for_score(score: float) -> RiskDecision:
    """Return the default policy decision for a score."""

    return risk_band_for_score(score).decision


def weighted_risk_score(
    category_scores: dict[str, float],
    category_weights: dict[str, float],
    model_score: float | None = None,
    model_weight: float = 0.55,
) -> float:
    """Combine model and category scores into one risk score."""

    if not category_scores and model_score is None:
        return 0.0

    normalized_category_score = _weighted_category_score(category_scores, category_weights)

    if model_score is None:
        return normalized_category_score

    safe_model_weight = clamp_score(model_weight)
    safe_model_score = clamp_score(model_score)
    category_weight = 1.0 - safe_model_weight

    final_score = (safe_model_score * safe_model_weight) + (
        normalized_category_score * category_weight
    )

    return clamp_score(final_score)


def _weighted_category_score(
    category_scores: dict[str, float],
    category_weights: dict[str, float],
) -> float:
    """Calculate weighted score for rule/category detections."""

    if not category_scores:
        return 0.0

    weighted_sum = 0.0
    total_weight = 0.0

    for category, score in category_scores.items():
        weight = max(0.0, float(category_weights.get(category, 0.1)))
        weighted_sum += clamp_score(score) * weight
        total_weight += weight

    if total_weight == 0.0:
        return 0.0

    return clamp_score(weighted_sum / total_weight)


__all__ = [
    "DEFAULT_RISK_BANDS",
    "RiskBand",
    "clamp_score",
    "decision_for_score",
    "risk_band_for_score",
    "risk_level_for_score",
    "weighted_risk_score",
]
