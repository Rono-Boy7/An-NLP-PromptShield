"""
src/promptshield/ingestion/normalizer.py

Purpose
-------
Text normalization utilities for PromptShield.

Why this file matters
---------------------
PromptShield scans untrusted text from webpages, emails, PDFs, RAG chunks, and
tool outputs. These sources can contain messy formatting:

- repeated whitespace
- invisible Unicode characters
- copied HTML spacing
- broken line endings
- null bytes or control characters
- extremely long repeated characters

Before scanning or model inference, we normalize text into a cleaner and more
consistent form.

What this module does
---------------------
1. Normalize Unicode text
2. Remove unsafe control characters
3. Standardize line endings and whitespace
4. Collapse excessive repeated characters
5. Provide one main normalize_text function for ingestion code

Design scope
------------
This file does not classify text as safe or unsafe.

It only prepares text so downstream scanners and models receive cleaner input.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
ZERO_WIDTH_PATTERN = re.compile(r"[\u200B-\u200F\u202A-\u202E\u2060-\u206F\uFEFF]")
REPEATED_SPACE_PATTERN = re.compile(r"[ \t]+")
REPEATED_NEWLINE_PATTERN = re.compile(r"\n{3,}")
REPEATED_CHAR_PATTERN = re.compile(r"(.)\1{20,}", flags=re.DOTALL)


@dataclass(frozen=True, slots=True)
class NormalizationResult:
    """Normalized text plus lightweight cleanup metadata."""

    text: str
    original_length: int
    normalized_length: int
    removed_control_chars: int
    removed_zero_width_chars: int

    @property
    def changed(self) -> bool:
        """Return whether normalization changed the text length."""

        return self.original_length != self.normalized_length


def normalize_text(text: str, *, collapse_repeated_chars: bool = True) -> str:
    """Return normalized text only."""

    return normalize_text_with_metadata(
        text,
        collapse_repeated_chars=collapse_repeated_chars,
    ).text


def normalize_text_with_metadata(
    text: str,
    *,
    collapse_repeated_chars: bool = True,
) -> NormalizationResult:
    """Normalize text and return cleanup metadata."""

    if not isinstance(text, str):
        raise TypeError("normalize_text expects a string.")

    original_length = len(text)

    normalized = normalize_unicode(text)
    normalized, removed_control_chars = remove_control_chars(normalized)
    normalized, removed_zero_width_chars = remove_zero_width_chars(normalized)
    normalized = normalize_line_endings(normalized)
    normalized = normalize_whitespace(normalized)

    if collapse_repeated_chars:
        normalized = collapse_excessive_repeated_chars(normalized)

    normalized = normalized.strip()

    return NormalizationResult(
        text=normalized,
        original_length=original_length,
        normalized_length=len(normalized),
        removed_control_chars=removed_control_chars,
        removed_zero_width_chars=removed_zero_width_chars,
    )


def normalize_unicode(text: str) -> str:
    """Normalize Unicode into a consistent compatibility form."""

    return unicodedata.normalize("NFKC", text)


def remove_control_chars(text: str) -> tuple[str, int]:
    """Remove non-printable control characters while preserving newlines and tabs."""

    matches = CONTROL_CHAR_PATTERN.findall(text)
    cleaned = CONTROL_CHAR_PATTERN.sub("", text)

    return cleaned, len(matches)


def remove_zero_width_chars(text: str) -> tuple[str, int]:
    """Remove invisible zero-width and directional formatting characters."""

    matches = ZERO_WIDTH_PATTERN.findall(text)
    cleaned = ZERO_WIDTH_PATTERN.sub("", text)

    return cleaned, len(matches)


def normalize_line_endings(text: str) -> str:
    """Convert Windows and old Mac line endings to Unix-style newlines."""

    return text.replace("\r\n", "\n").replace("\r", "\n")


def normalize_whitespace(text: str) -> str:
    """Normalize repeated spaces and excessive blank lines."""

    lines = []

    for line in text.split("\n"):
        cleaned_line = REPEATED_SPACE_PATTERN.sub(" ", line).strip()
        lines.append(cleaned_line)

    cleaned = "\n".join(lines)
    cleaned = REPEATED_NEWLINE_PATTERN.sub("\n\n", cleaned)

    return cleaned


def collapse_excessive_repeated_chars(text: str, *, max_repeats: int = 20) -> str:
    """Limit extremely long repeated character runs."""

    if max_repeats < 1:
        raise ValueError("max_repeats must be at least 1.")

    def replace_match(match: re.Match[str]) -> str:
        repeated_char = match.group(1)
        return repeated_char * max_repeats

    return REPEATED_CHAR_PATTERN.sub(replace_match, text)


def is_effectively_empty(text: str) -> bool:
    """Return true when text is empty after normalization."""

    return normalize_text(text) == ""


__all__ = [
    "NormalizationResult",
    "collapse_excessive_repeated_chars",
    "is_effectively_empty",
    "normalize_line_endings",
    "normalize_text",
    "normalize_text_with_metadata",
    "normalize_unicode",
    "normalize_whitespace",
    "remove_control_chars",
    "remove_zero_width_chars",
]
