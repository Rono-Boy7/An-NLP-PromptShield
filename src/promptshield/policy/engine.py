"""
src/promptshield/policy/engine.py

Purpose
-------
Policy engine for turning PromptShield detection results into security decisions.

Why this file matters
---------------------
Detection alone is not enough for a security system.

A scanner can say:

- this text looks suspicious
- this category was detected
- this model probability is high
- these spans were flagged

But the product still needs to decide what to do next.

This module converts risk scores into practical decisions:

- ALLOW
- ALLOW_WITH_WARNING
- SANITIZE
- BLOCK

What this module does
---------------------
1. Load policy thresholds from config
2. Map risk scores to risk levels and decisions
3. Expose category weights for scoring
4. Return structured PolicyResult objects
5. Keep policy logic separate from scanners and models

Design scope
------------
This file does not scan text.

It only decides how PromptShield should respond after detection has already
produced a risk score.
"""

from __future__ import annotations

from dataclasses import dataclass

from promptshield.core.labels import RiskDecision, RiskLevel
from promptshield.core.risk import clamp_score
from promptshield.core.types import CategoryScore, ModelPrediction, PolicyResult
from promptshield.utils.config import PolicyConfig, load_policy_config


@dataclass(frozen=True, slots=True)
class PolicyBand:
    """One risk score range mapped to a decision."""

    name: str
    min_score: float
    max_score: float
    decision: RiskDecision
    risk_level: RiskLevel

    def contains(self, score: float) -> bool:
        """Return whether a score belongs to this band."""

        normalized_score = clamp_score(score)
        return self.min_score <= normalized_score <= self.max_score


class PolicyEngine:
    """Config-driven engine for final risk decisions."""

    def __init__(self, bands: tuple[PolicyBand, ...], category_weights: dict[str, float]) -> None:
        if not bands:
            raise ValueError("PolicyEngine requires at least one policy band.")

        self.bands = tuple(sorted(bands, key=lambda band: band.min_score))
        self.category_weights = dict(category_weights)

    @classmethod
    def from_config(cls, config: PolicyConfig) -> PolicyEngine:
        """Create a policy engine from validated policy config."""

        bands = []

        for name, risk_policy in config.risk_levels.items():
            bands.append(
                PolicyBand(
                    name=name,
                    min_score=risk_policy.min_score,
                    max_score=risk_policy.max_score,
                    decision=RiskDecision(risk_policy.decision),
                    risk_level=risk_level_from_band_name(name),
                )
            )

        category_weights = {
            category_name: category_policy.weight
            for category_name, category_policy in config.categories.items()
        }

        return cls(
            bands=tuple(bands),
            category_weights=category_weights,
        )

    def decide(
        self,
        risk_score: float,
        *,
        category_scores: tuple[CategoryScore, ...] = (),
        model_prediction: ModelPrediction | None = None,
    ) -> PolicyResult:
        """Return the final policy decision for a risk score."""

        normalized_score = clamp_score(risk_score)
        band = self.band_for_score(normalized_score)
        message = build_policy_message(
            band=band,
            risk_score=normalized_score,
            category_scores=category_scores,
            model_prediction=model_prediction,
        )

        return PolicyResult(
            decision=band.decision,
            risk_level=band.risk_level,
            risk_score=normalized_score,
            should_sanitize=band.decision == RiskDecision.SANITIZE,
            should_block=band.decision == RiskDecision.BLOCK,
            message=message,
        )

    def band_for_score(self, risk_score: float) -> PolicyBand:
        """Return the policy band for a score."""

        normalized_score = clamp_score(risk_score)

        for band in self.bands:
            if band.contains(normalized_score):
                return band

        return self.bands[-1]


def load_policy_engine() -> PolicyEngine:
    """Load the default policy engine from configs/policy.yaml."""

    return PolicyEngine.from_config(load_policy_config())


def risk_level_from_band_name(name: str) -> RiskLevel:
    """Map policy band names to readable risk levels."""

    normalized = name.strip().lower()

    if normalized == "allow":
        return RiskLevel.LOW

    if normalized == "monitor":
        return RiskLevel.MEDIUM

    if normalized == "sanitize":
        return RiskLevel.HIGH

    if normalized == "block":
        return RiskLevel.CRITICAL

    raise ValueError(f"Unknown policy band name: {name}")


def build_policy_message(
    *,
    band: PolicyBand,
    risk_score: float,
    category_scores: tuple[CategoryScore, ...],
    model_prediction: ModelPrediction | None,
) -> str:
    """Build a short human-readable policy explanation."""

    category_count = len(category_scores)

    if band.decision == RiskDecision.ALLOW:
        return f"Content allowed. Risk score {risk_score:.2f} is within the low-risk range."

    if band.decision == RiskDecision.ALLOW_WITH_WARNING:
        return (
            f"Content allowed with warning. Risk score {risk_score:.2f} "
            f"shows moderate risk across {category_count} categories."
        )

    if band.decision == RiskDecision.SANITIZE:
        return (
            f"Content should be sanitized. Risk score {risk_score:.2f} "
            f"shows high risk across {category_count} categories."
        )

    if band.decision == RiskDecision.BLOCK:
        model_context = ""

        if model_prediction is not None:
            model_context = (
                f" Model suspicious probability: {model_prediction.suspicious_probability:.2f}."
            )

        return (
            f"Content should be blocked. Risk score {risk_score:.2f} "
            f"is in the critical range.{model_context}"
        )

    return f"Policy decision {band.decision} selected for risk score {risk_score:.2f}."


__all__ = [
    "PolicyBand",
    "PolicyEngine",
    "build_policy_message",
    "load_policy_engine",
    "risk_level_from_band_name",
]
