"""
scripts/prepare_dataset.py

Purpose
-------
Prepare the raw Deepset prompt injection dataset for PromptShield training.

Why this file matters
---------------------
The raw dataset is stored as Parquet files. Training scripts should not directly
depend on raw files because we want a clean, repeatable preprocessing step.

This script converts the raw dataset into normalized JSONL files:

- train split
- validation split
- test split
- dataset summary

What this script does
---------------------
1. Load raw train/test Parquet files
2. Validate required columns
3. Normalize text
4. Clean labels
5. Remove empty rows and duplicates
6. Create a validation split from the raw train set
7. Save processed JSONL files for model training

Design scope
------------
This script prepares data only.

It does not train models or print raw adversarial examples to the terminal.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from promptshield.ingestion.normalizer import normalize_text
from promptshield.utils.config import load_app_config
from promptshield.utils.logging import configure_logging, get_logger
from promptshield.utils.paths import ensure_project_dir, project_path, require_project_file

PROCESSED_TRAIN_FILENAME = "deepset_train.jsonl"
PROCESSED_VAL_FILENAME = "deepset_val.jsonl"
PROCESSED_TEST_FILENAME = "deepset_test.jsonl"
SUMMARY_FILENAME = "deepset_dataset_summary.json"


@dataclass(frozen=True, slots=True)
class DatasetSummary:
    """Small dataset summary saved after preprocessing."""

    train_rows: int
    val_rows: int
    test_rows: int
    train_label_counts: dict[str, int]
    val_label_counts: dict[str, int]
    test_label_counts: dict[str, int]
    removed_empty_rows: int
    removed_duplicate_rows: int
    output_dir: str


def main() -> None:
    """Prepare the Deepset dataset."""

    configure_logging()
    log = get_logger(component="prepare_dataset")

    app_config = load_app_config()
    dataset_config = app_config.datasets.deepset_prompt_injections

    raw_train_path = require_project_file(dataset_config.train_path)
    raw_test_path = require_project_file(dataset_config.test_path)
    output_dir = ensure_project_dir(app_config.paths.processed_data_dir)

    log.info("Loading raw dataset files.")
    raw_train_df = pd.read_parquet(raw_train_path)
    raw_test_df = pd.read_parquet(raw_test_path)

    train_df, removed_train_empty, removed_train_duplicates = clean_dataset_frame(
        raw_train_df,
        text_column=dataset_config.text_column,
        label_column=dataset_config.label_column,
        split_name="raw_train",
    )
    test_df, removed_test_empty, removed_test_duplicates = clean_dataset_frame(
        raw_test_df,
        text_column=dataset_config.text_column,
        label_column=dataset_config.label_column,
        split_name="raw_test",
    )

    train_df, val_df = split_train_validation(train_df)

    train_path = output_dir / PROCESSED_TRAIN_FILENAME
    val_path = output_dir / PROCESSED_VAL_FILENAME
    test_path = output_dir / PROCESSED_TEST_FILENAME
    summary_path = output_dir / SUMMARY_FILENAME

    write_jsonl(train_df, train_path)
    write_jsonl(val_df, val_path)
    write_jsonl(test_df, test_path)

    summary = DatasetSummary(
        train_rows=len(train_df),
        val_rows=len(val_df),
        test_rows=len(test_df),
        train_label_counts=label_counts(train_df),
        val_label_counts=label_counts(val_df),
        test_label_counts=label_counts(test_df),
        removed_empty_rows=removed_train_empty + removed_test_empty,
        removed_duplicate_rows=removed_train_duplicates + removed_test_duplicates,
        output_dir=str(output_dir.relative_to(project_path())),
    )

    write_summary(summary, summary_path)

    log.success("Processed dataset saved.")
    log.info("Train: {}", train_path.relative_to(project_path()))
    log.info("Validation: {}", val_path.relative_to(project_path()))
    log.info("Test: {}", test_path.relative_to(project_path()))
    log.info("Summary: {}", summary_path.relative_to(project_path()))
    log.info("Train rows: {}", summary.train_rows)
    log.info("Validation rows: {}", summary.val_rows)
    log.info("Test rows: {}", summary.test_rows)


def clean_dataset_frame(
    df: pd.DataFrame,
    *,
    text_column: str,
    label_column: str,
    split_name: str,
) -> tuple[pd.DataFrame, int, int]:
    """Clean one raw dataset frame."""

    validate_columns(
        df,
        required_columns=[text_column, label_column],
        split_name=split_name,
    )

    cleaned = df[[text_column, label_column]].copy()
    cleaned = cleaned.rename(
        columns={
            text_column: "text",
            label_column: "label",
        }
    )

    original_rows = len(cleaned)

    cleaned["text"] = cleaned["text"].astype(str).map(normalize_text)
    cleaned["label"] = cleaned["label"].map(normalize_label)

    cleaned = cleaned[cleaned["text"].str.len() > 0].copy()
    removed_empty_rows = original_rows - len(cleaned)

    before_dedup = len(cleaned)
    cleaned = cleaned.drop_duplicates(subset=["text", "label"]).copy()
    removed_duplicate_rows = before_dedup - len(cleaned)

    cleaned = cleaned.reset_index(drop=True)
    cleaned.insert(0, "id", [f"{split_name}_{index:05d}" for index in range(len(cleaned))])
    cleaned["source_dataset"] = "deepset_prompt_injections"

    return cleaned, removed_empty_rows, removed_duplicate_rows


def validate_columns(
    df: pd.DataFrame,
    *,
    required_columns: list[str],
    split_name: str,
) -> None:
    """Validate required columns exist in a dataframe."""

    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(
            f"{split_name} is missing required columns: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )


def normalize_label(value: object) -> int:
    """Normalize a dataset label into 0 or 1."""

    label = int(value)

    if label not in {0, 1}:
        raise ValueError(f"Expected binary label 0 or 1, got {label}.")

    return label


def split_train_validation(
    df: pd.DataFrame, *, val_size: float = 0.20
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a stratified validation split from the training data."""

    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        random_state=42,
        stratify=df["label"],
    )

    train_df = train_df.reset_index(drop=True).copy()
    val_df = val_df.reset_index(drop=True).copy()

    train_df["id"] = [f"train_{index:05d}" for index in range(len(train_df))]
    val_df["id"] = [f"val_{index:05d}" for index in range(len(val_df))]

    return train_df, val_df


def write_jsonl(df: pd.DataFrame, path: Path) -> None:
    """Write a dataframe to JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient="records", lines=True, force_ascii=False)


def write_summary(summary: DatasetSummary, path: Path) -> None:
    """Write dataset summary as JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(asdict(summary), file, indent=2, sort_keys=True)


def label_counts(df: pd.DataFrame) -> dict[str, int]:
    """Return label counts with string keys for JSON output."""

    counts = df["label"].value_counts().sort_index()

    return {str(label): int(count) for label, count in counts.items()}


if __name__ == "__main__":
    main()
