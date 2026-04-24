"""
src/promptshield/utils/config.py

Purpose
-------
Typed configuration loading for PromptShield.

Why this file matters
---------------------
A production-style ML/security project should not scatter constants across
training scripts, API files, and policy code.

PromptShield has multiple config files:

- configs/default.yaml
- configs/model.yaml
- configs/policy.yaml

This module loads those YAML files, validates their shape with Pydantic, and
returns typed Python objects that the rest of the project can safely use.

What this module does
---------------------
1. Load YAML files from the project configs folder
2. Validate config fields with Pydantic models
3. Provide typed loaders for app, model, and policy configs
4. Support simple environment variable overrides for local development

Design scope
------------
This module only loads and validates config.

It does not train models, scan text, or run API logic.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from promptshield.utils.paths import require_project_file

YamlDict = dict[str, Any]


class StrictConfigModel(BaseModel):
    """Base model for strict config validation."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class AppSection(StrictConfigModel):
    """Basic app metadata and runtime mode."""

    name: str
    version: str
    environment: str
    log_level: str


class PathsSection(StrictConfigModel):
    """Common project paths."""

    project_root: str
    data_dir: str
    raw_data_dir: str
    interim_data_dir: str
    processed_data_dir: str
    samples_data_dir: str
    artifacts_dir: str
    models_dir: str
    reports_dir: str


class DatasetSpec(StrictConfigModel):
    """Dataset file paths and column names."""

    train_path: str
    test_path: str
    text_column: str
    label_column: str


class DatasetsSection(StrictConfigModel):
    """All supported dataset definitions."""

    deepset_prompt_injections: DatasetSpec


class ApiSection(StrictConfigModel):
    """Local API server settings."""

    host: str
    port: int
    reload: bool
    title: str
    description: str


class RuntimeSection(StrictConfigModel):
    """Runtime limits and defaults."""

    random_seed: int
    max_input_chars: int
    default_chunk_size: int
    default_chunk_overlap: int


class AppConfig(StrictConfigModel):
    """Validated config from configs/default.yaml."""

    app: AppSection
    paths: PathsSection
    datasets: DatasetsSection
    api: ApiSection
    runtime: RuntimeSection


class LabelSection(StrictConfigModel):
    """Model label mapping."""

    benign: int
    suspicious: int


class BaselineSection(StrictConfigModel):
    """TF-IDF + Logistic Regression baseline settings."""

    name: str
    text_column: str
    label_column: str
    max_features: int
    ngram_range: tuple[int, int]
    min_df: int
    max_df: float
    lowercase: bool
    class_weight: str | None
    random_seed: int


class TransformerSection(StrictConfigModel):
    """Transformer classifier settings."""

    name: str
    base_model: str
    text_column: str
    label_column: str
    max_length: int
    train_batch_size: int
    eval_batch_size: int
    learning_rate: float
    weight_decay: float
    epochs: int
    warmup_ratio: float
    random_seed: int


class ThresholdSection(StrictConfigModel):
    """Model probability thresholds."""

    suspicious_probability: float = Field(ge=0.0, le=1.0)
    high_risk_probability: float = Field(ge=0.0, le=1.0)
    block_probability: float = Field(ge=0.0, le=1.0)


class ArtifactSection(StrictConfigModel):
    """Model output locations."""

    baseline_model_dir: str
    transformer_model_dir: str
    metrics_dir: str
    predictions_dir: str


class ModelConfig(StrictConfigModel):
    """Validated config from configs/model.yaml."""

    labels: LabelSection
    baseline: BaselineSection
    transformer: TransformerSection
    thresholds: ThresholdSection
    artifacts: ArtifactSection


class RiskLevelPolicy(StrictConfigModel):
    """Score range mapped to a policy decision."""

    min_score: float = Field(ge=0.0, le=1.0)
    max_score: float = Field(ge=0.0, le=1.0)
    decision: str


class CategoryPolicy(StrictConfigModel):
    """Risk category weight and explanation."""

    weight: float = Field(ge=0.0, le=1.0)
    description: str


class SafeContextSection(StrictConfigModel):
    """Safe context builder settings."""

    enabled: bool
    include_risk_summary: bool
    include_removed_span_count: bool
    max_safe_context_chars: int


class SanitizationSection(StrictConfigModel):
    """Sanitization behavior for risky spans."""

    replace_removed_spans_with_marker: bool
    marker: str
    preserve_original_offsets: bool


class PolicyConfig(StrictConfigModel):
    """Validated config from configs/policy.yaml."""

    risk_levels: dict[str, RiskLevelPolicy]
    categories: dict[str, CategoryPolicy]
    safe_context: SafeContextSection
    sanitization: SanitizationSection


def load_yaml_file(path: str | Path) -> YamlDict:
    """Load a YAML file into a dictionary."""

    config_path = Path(path)

    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise TypeError(f"Expected YAML object in {config_path}, got {type(data).__name__}")

    return data


def load_project_yaml(*parts: str | Path) -> YamlDict:
    """Load a YAML file from the project root."""

    return load_yaml_file(require_project_file(*parts))


def load_app_config() -> AppConfig:
    """Load and validate configs/default.yaml."""

    data = load_project_yaml("configs", "default.yaml")
    data = _apply_app_env_overrides(data)

    return AppConfig.model_validate(data)


def load_model_config() -> ModelConfig:
    """Load and validate configs/model.yaml."""

    data = load_project_yaml("configs", "model.yaml")

    return ModelConfig.model_validate(data)


def load_policy_config() -> PolicyConfig:
    """Load and validate configs/policy.yaml."""

    data = load_project_yaml("configs", "policy.yaml")

    return PolicyConfig.model_validate(data)


def _apply_app_env_overrides(data: YamlDict) -> YamlDict:
    """Apply simple environment overrides for local development."""

    updated = dict(data)
    app = dict(updated.get("app", {}))
    api = dict(updated.get("api", {}))
    paths = dict(updated.get("paths", {}))

    if env := os.getenv("PROMPTSHIELD_ENV"):
        app["environment"] = env

    if log_level := os.getenv("PROMPTSHIELD_LOG_LEVEL"):
        app["log_level"] = log_level

    if api_host := os.getenv("PROMPTSHIELD_API_HOST"):
        api["host"] = api_host

    if api_port := os.getenv("PROMPTSHIELD_API_PORT"):
        api["port"] = int(api_port)

    if data_dir := os.getenv("PROMPTSHIELD_DATA_DIR"):
        paths["data_dir"] = data_dir

    if raw_data_dir := os.getenv("PROMPTSHIELD_RAW_DATA_DIR"):
        paths["raw_data_dir"] = raw_data_dir

    if processed_data_dir := os.getenv("PROMPTSHIELD_PROCESSED_DATA_DIR"):
        paths["processed_data_dir"] = processed_data_dir

    if artifacts_dir := os.getenv("PROMPTSHIELD_ARTIFACTS_DIR"):
        paths["artifacts_dir"] = artifacts_dir

    if models_dir := os.getenv("PROMPTSHIELD_MODELS_DIR"):
        paths["models_dir"] = models_dir

    updated["app"] = app
    updated["api"] = api
    updated["paths"] = paths

    return updated


__all__ = [
    "AppConfig",
    "ModelConfig",
    "PolicyConfig",
    "load_app_config",
    "load_model_config",
    "load_policy_config",
    "load_project_yaml",
    "load_yaml_file",
]
