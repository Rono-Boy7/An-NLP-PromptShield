"""
src/promptshield/ingestion/cleaners.py

Purpose
-------
Source-specific cleaning utilities for untrusted text.

Why this file matters
---------------------
PromptShield will scan text from different sources, and each source has its own
noise patterns.

Examples:
- webpages contain HTML tags, scripts, styles, and navigation text
- emails contain headers, quoted replies, and signatures
- tool outputs may contain JSON-like or markdown-like formatting
- raw text may contain messy spacing or copied formatting

This module gives us reusable cleaners before chunking and scanning.

What this module does
---------------------
1. Clean raw HTML into readable text
2. Clean email-like text while preserving useful content
3. Clean plain text with shared normalization
4. Wrap cleaned content in TextDocument objects
5. Keep source metadata for later explainability

Design scope
------------
This file does not detect prompt injection.

It prepares untrusted content so the scanner, model, and policy engine can work
with cleaner text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from email import policy
from email.parser import Parser
from typing import Any

from bs4 import BeautifulSoup

from promptshield.core.types import SourceType, TextDocument
from promptshield.ingestion.normalizer import normalize_text

SCRIPT_STYLE_TAGS = {
    "script",
    "style",
    "noscript",
    "template",
    "svg",
    "canvas",
}

BLOCK_TAGS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "br",
    "div",
    "footer",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "td",
    "th",
    "tr",
    "ul",
}

EMAIL_HEADER_PATTERN = re.compile(
    r"^(from|to|cc|bcc|subject|date|reply-to|message-id):\s*",
    flags=re.IGNORECASE,
)

QUOTED_REPLY_PATTERN = re.compile(
    r"^\s*(>|on .+ wrote:|from:\s|sent:\s|subject:\s)",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class CleanedText:
    """Cleaned text plus source metadata."""

    text: str
    source_type: SourceType
    original_length: int
    cleaned_length: int
    metadata: dict[str, Any]

    @property
    def changed(self) -> bool:
        """Return whether cleaning changed the text length."""

        return self.original_length != self.cleaned_length


def clean_text(
    text: str,
    *,
    source_type: SourceType = SourceType.UNKNOWN,
    source_uri: str | None = None,
) -> CleanedText:
    """Clean text based on its source type."""

    if not isinstance(text, str):
        raise TypeError("clean_text expects a string.")

    if source_type == SourceType.WEBPAGE:
        return clean_html(text, source_uri=source_uri)

    if source_type == SourceType.EMAIL:
        return clean_email(text, source_uri=source_uri)

    return clean_plain_text(text, source_type=source_type, source_uri=source_uri)


def clean_plain_text(
    text: str,
    *,
    source_type: SourceType = SourceType.UNKNOWN,
    source_uri: str | None = None,
) -> CleanedText:
    """Clean generic plain text."""

    cleaned = normalize_text(text)

    return CleanedText(
        text=cleaned,
        source_type=source_type,
        original_length=len(text),
        cleaned_length=len(cleaned),
        metadata={
            "source_uri": source_uri,
            "cleaner": "plain_text",
        },
    )


def clean_html(html: str, *, source_uri: str | None = None) -> CleanedText:
    """Convert raw HTML into readable normalized text."""

    if not isinstance(html, str):
        raise TypeError("clean_html expects a string.")

    soup = BeautifulSoup(html, "lxml")

    for tag_name in SCRIPT_STYLE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    for tag in soup.find_all(True):
        if tag.name in BLOCK_TAGS:
            tag.append("\n")

    title = extract_html_title(soup)
    visible_text = soup.get_text(separator=" ", strip=True)
    cleaned = normalize_text(visible_text)

    return CleanedText(
        text=cleaned,
        source_type=SourceType.WEBPAGE,
        original_length=len(html),
        cleaned_length=len(cleaned),
        metadata={
            "source_uri": source_uri,
            "cleaner": "html",
            "title": title,
        },
    )


def clean_email(raw_email: str, *, source_uri: str | None = None) -> CleanedText:
    """Clean email-like content while keeping the main readable body."""

    if not isinstance(raw_email, str):
        raise TypeError("clean_email expects a string.")

    parsed_body = extract_email_body(raw_email)
    without_headers = remove_email_headers(parsed_body)
    without_quotes = remove_quoted_reply_lines(without_headers)
    cleaned = normalize_text(without_quotes)

    return CleanedText(
        text=cleaned,
        source_type=SourceType.EMAIL,
        original_length=len(raw_email),
        cleaned_length=len(cleaned),
        metadata={
            "source_uri": source_uri,
            "cleaner": "email",
        },
    )


def extract_html_title(soup: BeautifulSoup) -> str | None:
    """Extract a normalized HTML title when present."""

    if soup.title is None or soup.title.string is None:
        return None

    title = normalize_text(soup.title.string)

    return title or None


def extract_email_body(raw_email: str) -> str:
    """Extract the best-effort body from a raw email string."""

    message = Parser(policy=policy.default).parsestr(raw_email)

    if message.is_multipart():
        for part in message.walk():
            content_type = part.get_content_type()

            if content_type == "text/plain":
                payload = part.get_payload(decode=False)
                if isinstance(payload, str):
                    return payload

        return raw_email

    payload = message.get_payload(decode=False)

    if isinstance(payload, str):
        return payload

    return raw_email


def remove_email_headers(text: str) -> str:
    """Remove common email header lines from text."""

    lines = []

    for line in text.splitlines():
        if EMAIL_HEADER_PATTERN.match(line):
            continue

        lines.append(line)

    return "\n".join(lines)


def remove_quoted_reply_lines(text: str) -> str:
    """Remove common quoted reply lines from email text."""

    lines = []

    for line in text.splitlines():
        if QUOTED_REPLY_PATTERN.match(line):
            continue

        lines.append(line)

    return "\n".join(lines)


def cleaned_text_to_document(
    cleaned: CleanedText,
    *,
    document_id: str,
) -> TextDocument:
    """Convert CleanedText into a TextDocument."""

    return TextDocument(
        id=document_id,
        text=cleaned.text,
        source_type=cleaned.source_type,
        source_uri=cleaned.metadata.get("source_uri"),
        metadata=cleaned.metadata,
    )


def create_text_document(
    *,
    document_id: str,
    text: str,
    source_type: SourceType = SourceType.UNKNOWN,
    source_uri: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> TextDocument:
    """Clean raw text and return a TextDocument."""

    cleaned = clean_text(
        text,
        source_type=source_type,
        source_uri=source_uri,
    )

    merged_metadata = dict(cleaned.metadata)

    if metadata:
        merged_metadata.update(metadata)

    return TextDocument(
        id=document_id,
        text=cleaned.text,
        source_type=cleaned.source_type,
        source_uri=source_uri,
        metadata=merged_metadata,
    )


__all__ = [
    "CleanedText",
    "clean_email",
    "clean_html",
    "clean_plain_text",
    "clean_text",
    "cleaned_text_to_document",
    "create_text_document",
    "extract_email_body",
    "extract_html_title",
    "remove_email_headers",
    "remove_quoted_reply_lines",
]
