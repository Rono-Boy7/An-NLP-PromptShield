"""
src/promptshield/utils/paths.py

Purpose
-------
Centralized path utilities for PromptShield.

Why this file matters
---------------------
ML/security projects usually touch many folders:

- raw datasets
- processed datasets
- model artifacts
- reports
- configs
- logs

Hardcoding paths across scripts makes the project fragile. If one folder moves,
many files break.

This module gives us one clean place to resolve project paths safely.

What this module does
---------------------
1. Finds the project root
2. Builds common project paths
3. Creates directories when needed
4. Provides small helpers for validating files and folders

Design scope
------------
This file does not load data or configs.

It only handles filesystem paths so the rest of the project can stay clean and
consistent.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_MARKERS = ("pyproject.toml", ".git")


def find_project_root(start_path: Path | None = None) -> Path:
    """Find the project root by walking upward from a starting path."""

    current_path = (start_path or Path.cwd()).resolve()

    if current_path.is_file():
        current_path = current_path.parent

    for path in (current_path, *current_path.parents):
        if any((path / marker).exists() for marker in PROJECT_MARKERS):
            return path

    raise FileNotFoundError(
        "Could not find project root. Expected pyproject.toml or .git in parent paths."
    )


def project_path(*parts: str | Path) -> Path:
    """Build an absolute path from the project root."""

    return find_project_root().joinpath(*parts)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""

    directory = Path(path).resolve()
    directory.mkdir(parents=True, exist_ok=True)

    return directory


def ensure_project_dir(*parts: str | Path) -> Path:
    """Create a directory inside the project root."""

    directory = project_path(*parts)
    directory.mkdir(parents=True, exist_ok=True)

    return directory


def require_file(path: str | Path) -> Path:
    """Return a file path or raise a clear error if it does not exist."""

    file_path = Path(path).resolve()

    if not file_path.is_file():
        raise FileNotFoundError(f"Required file not found: {file_path}")

    return file_path


def require_project_file(*parts: str | Path) -> Path:
    """Return a project file path or raise a clear error if it does not exist."""

    return require_file(project_path(*parts))


def require_dir(path: str | Path) -> Path:
    """Return a directory path or raise a clear error if it does not exist."""

    directory = Path(path).resolve()

    if not directory.is_dir():
        raise NotADirectoryError(f"Required directory not found: {directory}")

    return directory


def require_project_dir(*parts: str | Path) -> Path:
    """Return a project directory path or raise a clear error if it does not exist."""

    return require_dir(project_path(*parts))


def relative_to_root(path: str | Path) -> Path:
    """Return a path relative to the project root when possible."""

    root = find_project_root()
    resolved_path = Path(path).resolve()

    try:
        return resolved_path.relative_to(root)
    except ValueError:
        return resolved_path


__all__ = [
    "ensure_dir",
    "ensure_project_dir",
    "find_project_root",
    "project_path",
    "relative_to_root",
    "require_dir",
    "require_file",
    "require_project_dir",
    "require_project_file",
]
