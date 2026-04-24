"""
src/promptshield/detection/rule_scanner.py

Purpose
-------
Rule-based prompt injection risk scanner.

Why this file matters
---------------------
PromptShield should not rely only on machine learning.

A production-style security system usually combines:

- deterministic rules
- ML classifiers
- policy decisions
- explainable evidence

This module gives us the deterministic layer. It scans untrusted text for
instruction-like, role-changing, tool-manipulating, context-confusing, and
data-exfiltration patterns.

What this module does
---------------------
1. Define safe defensive detection patterns
2. Scan text or chunks for suspicious spans
3. Return explainable DetectionSpan objects
4. Aggregate category-level risk scores
5. Produce one rule-based risk score

Design scope
------------
This file does not call an LLM and does not generate attack content.

It only detects suspicious patterns inside text that PromptShield is asked to
inspect.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from promptshield.core.labels import RiskCategory
from promptshield.core.risk import clamp_score, weighted_risk_score
from promptshield.core.types import CategoryScore, DetectionSpan, TextChunk


@dataclass(frozen=True, slots=True)
class RulePattern:
    """One defensive rule used to flag suspicious text."""

    name: str
    category: RiskCategory
    regex: re.Pattern[str]
    score: float
    reason: str


@dataclass(frozen=True, slots=True)
class RuleScanResult:
    """Output from the rule-based scanner."""

    input_id: str
    risk_score: float
    spans: tuple[DetectionSpan, ...]
    category_scores: tuple[CategoryScore, ...]

    @property
    def flagged_span_count(self) -> int:
        """Return number of suspicious spans."""

        return len(self.spans)

    @property
    def detected_categories(self) -> tuple[RiskCategory, ...]:
        """Return unique detected categories."""

        categories = {span.category for span in self.spans}
        categories.update(score.category for score in self.category_scores)

        return tuple(sorted(categories, key=lambda category: category.value))


DEFAULT_CATEGORY_WEIGHTS: dict[str, float] = {
    RiskCategory.INSTRUCTION_OVERRIDE.value: 0.30,
    RiskCategory.ROLE_MANIPULATION.value: 0.20,
    RiskCategory.TOOL_MANIPULATION.value: 0.25,
    RiskCategory.DATA_EXFILTRATION.value: 0.35,
    RiskCategory.CONTEXT_CONFUSION.value: 0.15,
    RiskCategory.HIDDEN_INSTRUCTION.value: 0.20,
}


def _compile(pattern: str) -> re.Pattern[str]:
    """Compile a case-insensitive multiline regex pattern."""

    return re.compile(pattern, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)


DEFAULT_RULE_PATTERNS: tuple[RulePattern, ...] = (
    RulePattern(
        name="instruction_override_ignore",
        category=RiskCategory.INSTRUCTION_OVERRIDE,
        regex=_compile(
            r"\b(ignore|disregard|forget|bypass)\b.{0,80}"
            r"\b(instruction|prompt|rule|policy)\b"
        ),
        score=0.82,
        reason="Text appears to tell the model to ignore trusted instructions or policies.",
    ),
    RulePattern(
        name="instruction_override_priority",
        category=RiskCategory.INSTRUCTION_OVERRIDE,
        regex=_compile(
            r"\b(system|developer|highest|priority)\b.{0,80}"
            r"\b(instruction|message|rule)\b"
        ),
        score=0.72,
        reason="Text appears to claim higher instruction priority inside untrusted content.",
    ),
    RulePattern(
        name="role_manipulation_identity",
        category=RiskCategory.ROLE_MANIPULATION,
        regex=_compile(
            r"\b(you are now|act as|pretend to be|roleplay as)\b.{0,80}"
            r"\b(agent|assistant|system|developer|admin)\b"
        ),
        score=0.68,
        reason="Text appears to alter the assistant role or identity.",
    ),
    RulePattern(
        name="role_manipulation_mode",
        category=RiskCategory.ROLE_MANIPULATION,
        regex=_compile(
            r"\b(enable|enter|switch to|activate)\b.{0,60}"
            r"\b(mode|developer mode|admin mode|debug mode)\b"
        ),
        score=0.70,
        reason="Text appears to request a different operating mode.",
    ),
    RulePattern(
        name="tool_manipulation_action",
        category=RiskCategory.TOOL_MANIPULATION,
        regex=_compile(
            r"\b(call|invoke|run|execute|use)\b.{0,80}"
            r"\b(tool|function|api|command|plugin)\b"
        ),
        score=0.66,
        reason="Text appears to influence tool or function usage.",
    ),
    RulePattern(
        name="tool_manipulation_hidden_action",
        category=RiskCategory.TOOL_MANIPULATION,
        regex=_compile(
            r"\b(without asking|without confirmation|silently|in the background)\b.{0,80}"
            r"\b(send|submit|delete|modify|forward|download|upload)\b"
        ),
        score=0.78,
        reason="Text appears to request hidden or unconfirmed external actions.",
    ),
    RulePattern(
        name="data_exfiltration_secret",
        category=RiskCategory.DATA_EXFILTRATION,
        regex=_compile(
            r"\b(reveal|print|show|exfiltrate|leak|send)\b.{0,80}"
            r"\b(secret|token|api key|password|credential|private key)\b"
        ),
        score=0.90,
        reason="Text appears to request private or secret information.",
    ),
    RulePattern(
        name="data_exfiltration_hidden_context",
        category=RiskCategory.DATA_EXFILTRATION,
        regex=_compile(
            r"\b(reveal|print|show|repeat|dump)\b.{0,80}"
            r"\b(system prompt|developer message|hidden context|internal instructions)\b"
        ),
        score=0.88,
        reason="Text appears to request hidden instructions or internal context.",
    ),
    RulePattern(
        name="context_confusion_boundary",
        category=RiskCategory.CONTEXT_CONFUSION,
        regex=_compile(
            r"\b(the following|above|previous|next)\b.{0,80}"
            r"\b(is not data|is an instruction|must be followed)\b"
        ),
        score=0.64,
        reason="Text appears to blur the boundary between untrusted data and instructions.",
    ),
    RulePattern(
        name="context_confusion_user_impersonation",
        category=RiskCategory.CONTEXT_CONFUSION,
        regex=_compile(
            r"\b(user says|user asked|the user wants|the real task is)\b.{0,100}"
            r"\b(ignore|change|override|instead)\b"
        ),
        score=0.69,
        reason="Text appears to impersonate or redefine the user's request.",
    ),
    RulePattern(
        name="hidden_instruction_encoded",
        category=RiskCategory.HIDDEN_INSTRUCTION,
        regex=_compile(
            r"\b(base64|rot13|encoded|decode this|hidden message|invisible instruction)\b"
        ),
        score=0.58,
        reason="Text references encoded or hidden instructions.",
    ),
    RulePattern(
        name="hidden_instruction_markdown_html",
        category=RiskCategory.HIDDEN_INSTRUCTION,
        regex=_compile(
            r"(<!--.*?(instruction|ignore|system|prompt).*?-->|"
            r"<[^>]+style\s*=.*?display\s*:\s*none.*?>)"
        ),
        score=0.74,
        reason="Text contains hidden-looking HTML or markup instruction patterns.",
    ),
)


class RuleScanner:
    """Deterministic scanner for suspicious instruction-like text."""

    def __init__(
        self,
        patterns: Iterable[RulePattern] | None = None,
        category_weights: dict[str, float] | None = None,
        detector_name: str = "rule_scanner",
    ) -> None:
        self.patterns = tuple(patterns or DEFAULT_RULE_PATTERNS)
        self.category_weights = dict(category_weights or DEFAULT_CATEGORY_WEIGHTS)
        self.detector_name = detector_name

    def scan_text(self, text: str, *, input_id: str = "text") -> RuleScanResult:
        """Scan raw text and return rule-based detections."""

        if not isinstance(text, str):
            raise TypeError("RuleScanner.scan_text expects a string.")

        spans = self._find_spans(text)
        category_scores = aggregate_category_scores(spans)
        risk_score = weighted_risk_score(
            category_scores={
                category_score.category.value: category_score.score
                for category_score in category_scores
            },
            category_weights=self.category_weights,
        )

        return RuleScanResult(
            input_id=input_id,
            risk_score=risk_score,
            spans=tuple(spans),
            category_scores=tuple(category_scores),
        )

    def scan_chunk(self, chunk: TextChunk) -> RuleScanResult:
        """Scan one TextChunk."""

        result = self.scan_text(chunk.text, input_id=chunk.id)

        return RuleScanResult(
            input_id=chunk.id,
            risk_score=result.risk_score,
            spans=offset_spans(result.spans, offset=chunk.start_char),
            category_scores=result.category_scores,
        )

    def scan_chunks(self, chunks: Iterable[TextChunk]) -> list[RuleScanResult]:
        """Scan multiple chunks."""

        return [self.scan_chunk(chunk) for chunk in chunks]

    def _find_spans(self, text: str) -> list[DetectionSpan]:
        """Find suspicious spans using all configured rules."""

        spans: list[DetectionSpan] = []

        for rule in self.patterns:
            for match in rule.regex.finditer(text):
                matched_text = match.group(0).strip()

                if not matched_text:
                    continue

                spans.append(
                    DetectionSpan(
                        start=match.start(),
                        end=match.end(),
                        text=matched_text,
                        category=rule.category,
                        score=rule.score,
                        reason=rule.reason,
                        detector_name=f"{self.detector_name}:{rule.name}",
                    )
                )

        return deduplicate_spans(spans)


def aggregate_category_scores(spans: Iterable[DetectionSpan]) -> list[CategoryScore]:
    """Aggregate span-level detections into category-level scores."""

    grouped: dict[RiskCategory, list[DetectionSpan]] = {}

    for span in spans:
        grouped.setdefault(span.category, []).append(span)

    category_scores = []

    for category, category_spans in grouped.items():
        max_score = max(span.score for span in category_spans)
        count_bonus = min(0.15, 0.03 * max(0, len(category_spans) - 1))
        final_score = clamp_score(max_score + count_bonus)

        category_scores.append(
            CategoryScore(
                category=category,
                score=final_score,
                evidence_count=len(category_spans),
            )
        )

    return sorted(category_scores, key=lambda score: score.category.value)


def deduplicate_spans(spans: Iterable[DetectionSpan]) -> list[DetectionSpan]:
    """Remove duplicate spans while keeping the highest scoring version."""

    best_by_key: dict[tuple[int, int, RiskCategory], DetectionSpan] = {}

    for span in spans:
        key = (span.start, span.end, span.category)
        current = best_by_key.get(key)

        if current is None or span.score > current.score:
            best_by_key[key] = span

    return sorted(
        best_by_key.values(),
        key=lambda span: (span.start, span.end, span.category.value),
    )


def offset_spans(spans: Iterable[DetectionSpan], *, offset: int) -> tuple[DetectionSpan, ...]:
    """Shift span offsets from chunk-relative to document-relative positions."""

    if offset < 0:
        raise ValueError("offset cannot be negative.")

    shifted_spans = []

    for span in spans:
        shifted_spans.append(
            DetectionSpan(
                start=span.start + offset,
                end=span.end + offset,
                text=span.text,
                category=span.category,
                score=span.score,
                reason=span.reason,
                detector_name=span.detector_name,
            )
        )

    return tuple(shifted_spans)


def scan_text(text: str, *, input_id: str = "text") -> RuleScanResult:
    """Convenience function for one-off rule scans."""

    return RuleScanner().scan_text(text, input_id=input_id)


__all__ = [
    "DEFAULT_CATEGORY_WEIGHTS",
    "DEFAULT_RULE_PATTERNS",
    "RulePattern",
    "RuleScanResult",
    "RuleScanner",
    "aggregate_category_scores",
    "deduplicate_spans",
    "offset_spans",
    "scan_text",
]
