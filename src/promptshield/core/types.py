# PromptShield/src/promptshield/core/types.py
#
# Shared internal data types for documents, chunks, detections, model predictions,
# and final scan results.

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from promptshield.core.labels import InjectionLabel, RiskCategory, RiskDecision, RiskLevel
from promptshield.core.risk import clamp_score


class SourceType(StrEnum):
    """Supported untrusted content sources."""

    WEBPAGE = "webpage"
    EMAIL = "email"
    PDF = "pdf"
    DOCUMENT = "document"
    RAG_CHUNK = "rag_chunk"
    TOOL_OUTPUT = "tool_output"
    USER_TEXT = "user_text"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class TextDocument:
    """Normalized document before chunking or scanning."""

    id: str
    text: str
    source_type: SourceType = SourceType.UNKNOWN
    source_uri: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("TextDocument.id cannot be empty.")

        if not isinstance(self.text, str):
            raise TypeError("TextDocument.text must be a string.")


@dataclass(frozen=True, slots=True)
class TextChunk:
    """A smaller section of a document used for scanning or model input."""

    id: str
    document_id: str
    text: str
    start_char: int
    end_char: int
    source_type: SourceType = SourceType.UNKNOWN
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("TextChunk.id cannot be empty.")

        if not self.document_id.strip():
            raise ValueError("TextChunk.document_id cannot be empty.")

        if self.start_char < 0:
            raise ValueError("TextChunk.start_char cannot be negative.")

        if self.end_char < self.start_char:
            raise ValueError("TextChunk.end_char cannot be smaller than start_char.")


@dataclass(frozen=True, slots=True)
class DetectionSpan:
    """A suspicious text span found by rules or models."""

    start: int
    end: int
    text: str
    category: RiskCategory
    score: float
    reason: str
    detector_name: str

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("DetectionSpan.start cannot be negative.")

        if self.end < self.start:
            raise ValueError("DetectionSpan.end cannot be smaller than start.")

        object.__setattr__(self, "score", clamp_score(self.score))


@dataclass(frozen=True, slots=True)
class CategoryScore:
    """Aggregated risk score for a single category."""

    category: RiskCategory
    score: float
    evidence_count: int = 0

    def __post_init__(self) -> None:
        if self.evidence_count < 0:
            raise ValueError("CategoryScore.evidence_count cannot be negative.")

        object.__setattr__(self, "score", clamp_score(self.score))


@dataclass(frozen=True, slots=True)
class ModelPrediction:
    """Classifier output for one text sample."""

    label: InjectionLabel
    suspicious_probability: float
    benign_probability: float
    model_name: str
    model_version: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "suspicious_probability",
            clamp_score(self.suspicious_probability),
        )
        object.__setattr__(
            self,
            "benign_probability",
            clamp_score(self.benign_probability),
        )

        if not self.model_name.strip():
            raise ValueError("ModelPrediction.model_name cannot be empty.")


@dataclass(frozen=True, slots=True)
class PolicyResult:
    """Final policy decision after risk scoring."""

    decision: RiskDecision
    risk_level: RiskLevel
    risk_score: float
    should_sanitize: bool
    should_block: bool
    message: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "risk_score", clamp_score(self.risk_score))


@dataclass(frozen=True, slots=True)
class ScanResult:
    """Complete result returned by PromptShield after scanning content."""

    input_id: str
    source_type: SourceType
    risk_score: float
    risk_level: RiskLevel
    decision: RiskDecision
    category_scores: tuple[CategoryScore, ...] = ()
    spans: tuple[DetectionSpan, ...] = ()
    model_prediction: ModelPrediction | None = None
    sanitized_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.input_id.strip():
            raise ValueError("ScanResult.input_id cannot be empty.")

        object.__setattr__(self, "risk_score", clamp_score(self.risk_score))

    @property
    def flagged_span_count(self) -> int:
        """Return how many suspicious spans were found."""

        return len(self.spans)

    @property
    def detected_categories(self) -> tuple[RiskCategory, ...]:
        """Return unique detected risk categories."""

        categories = {span.category for span in self.spans}
        categories.update(score.category for score in self.category_scores)

        return tuple(sorted(categories, key=lambda category: category.value))


__all__ = [
    "CategoryScore",
    "DetectionSpan",
    "ModelPrediction",
    "PolicyResult",
    "ScanResult",
    "SourceType",
    "TextChunk",
    "TextDocument",
]
