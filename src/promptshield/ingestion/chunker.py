"""
src/promptshield/ingestion/chunker.py

Purpose
-------
Chunking utilities for PromptShield.

Why this file matters
---------------------
PromptShield will often receive long untrusted content from webpages, emails,
PDFs, RAG documents, and tool outputs.

Long text is harder to scan reliably because:

- rule scanners may need precise span locations
- ML models have max token/input limits
- suspicious instructions may appear deep inside a document
- API responses should explain which section was risky

This module splits normalized documents into smaller overlapping chunks while
preserving character offsets back to the original cleaned document.

What this module does
---------------------
1. Split TextDocument objects into TextChunk objects
2. Preserve start and end character offsets
3. Support configurable chunk size and overlap
4. Prefer splitting on paragraph/sentence-like boundaries when possible
5. Provide safe defaults for scanner and model input

Design scope
------------
This file does not clean text or classify risk.

It only prepares cleaned documents for downstream scanning and model inference.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from promptshield.core.types import TextChunk, TextDocument

BOUNDARY_PATTERN = re.compile(r"(\n\n|\n|[.!?]\s+)")


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Settings for splitting documents into chunks."""

    chunk_size: int = 1200
    chunk_overlap: int = 150
    min_chunk_chars: int = 40
    prefer_boundaries: bool = True

    def __post_init__(self) -> None:
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be at least 1.")

        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        if self.min_chunk_chars < 1:
            raise ValueError("min_chunk_chars must be at least 1.")


def chunk_document(
    document: TextDocument,
    config: ChunkingConfig | None = None,
) -> list[TextChunk]:
    """Split one TextDocument into chunks."""

    active_config = config or ChunkingConfig()
    text = document.text

    if not text:
        return []

    ranges = chunk_text_ranges(
        text,
        chunk_size=active_config.chunk_size,
        chunk_overlap=active_config.chunk_overlap,
        min_chunk_chars=active_config.min_chunk_chars,
        prefer_boundaries=active_config.prefer_boundaries,
    )

    chunks = []

    for index, (start, end) in enumerate(ranges):
        chunk_text = text[start:end].strip()

        if not chunk_text:
            continue

        chunk = TextChunk(
            id=f"{document.id}_chunk_{index:04d}",
            document_id=document.id,
            text=chunk_text,
            start_char=start,
            end_char=end,
            source_type=document.source_type,
            metadata={
                **document.metadata,
                "chunk_index": index,
                "source_uri": document.source_uri,
            },
        )
        chunks.append(chunk)

    return chunks


def chunk_documents(
    documents: list[TextDocument],
    config: ChunkingConfig | None = None,
) -> list[TextChunk]:
    """Split multiple documents into a flat list of chunks."""

    chunks: list[TextChunk] = []

    for document in documents:
        chunks.extend(chunk_document(document, config=config))

    return chunks


def chunk_text_ranges(
    text: str,
    *,
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
    min_chunk_chars: int = 40,
    prefer_boundaries: bool = True,
) -> list[tuple[int, int]]:
    """Return character ranges for chunking text."""

    if not isinstance(text, str):
        raise TypeError("chunk_text_ranges expects a string.")

    config = ChunkingConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_chars=min_chunk_chars,
        prefer_boundaries=prefer_boundaries,
    )

    text_length = len(text)

    if text_length == 0:
        return []

    if text_length <= config.chunk_size:
        return [(0, text_length)]

    ranges = []
    start = 0

    while start < text_length:
        target_end = min(start + config.chunk_size, text_length)
        end = target_end

        if config.prefer_boundaries and target_end < text_length:
            end = find_best_boundary(
                text,
                start=start,
                target_end=target_end,
                min_chunk_chars=config.min_chunk_chars,
            )

        if end <= start:
            end = target_end

        if end - start >= config.min_chunk_chars or not ranges:
            ranges.append((start, end))

        if end >= text_length:
            break

        next_start = max(0, end - config.chunk_overlap)

        if next_start <= start:
            next_start = end

        start = next_start

    return merge_tiny_tail_ranges(
        ranges,
        text_length=text_length,
        min_chunk_chars=config.min_chunk_chars,
    )


def find_best_boundary(
    text: str,
    *,
    start: int,
    target_end: int,
    min_chunk_chars: int,
) -> int:
    """Find a clean split point near the target end."""

    search_start = start + min_chunk_chars
    search_region = text[search_start:target_end]

    if not search_region:
        return target_end

    boundary_matches = list(BOUNDARY_PATTERN.finditer(search_region))

    if not boundary_matches:
        return target_end

    best_match = boundary_matches[-1]

    return search_start + best_match.end()


def merge_tiny_tail_ranges(
    ranges: list[tuple[int, int]],
    *,
    text_length: int,
    min_chunk_chars: int,
) -> list[tuple[int, int]]:
    """Merge a very small final chunk into the previous chunk."""

    if len(ranges) < 2:
        return ranges

    previous_start, _ = ranges[-2]
    tail_start, tail_end = ranges[-1]

    if tail_end != text_length:
        return ranges

    if tail_end - tail_start >= min_chunk_chars:
        return ranges

    return [*ranges[:-2], (previous_start, tail_end)]


def estimate_chunk_count(
    text_length: int,
    *,
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
) -> int:
    """Estimate number of chunks without inspecting text boundaries."""

    if text_length <= 0:
        return 0

    config = ChunkingConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    if text_length <= config.chunk_size:
        return 1

    step = config.chunk_size - config.chunk_overlap

    return 1 + max(0, (text_length - config.chunk_size + step - 1) // step)


__all__ = [
    "ChunkingConfig",
    "chunk_document",
    "chunk_documents",
    "chunk_text_ranges",
    "estimate_chunk_count",
    "find_best_boundary",
    "merge_tiny_tail_ranges",
]
