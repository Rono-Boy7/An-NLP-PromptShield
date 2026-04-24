# PromptShield/src/promptshield/core/labels.py
#
# Shared labels used by classifiers, rule scanners, policies, and API responses.

from __future__ import annotations

from enum import IntEnum, StrEnum


class InjectionLabel(IntEnum):
    """Binary model labels for prompt injection detection."""

    BENIGN = 0
    SUSPICIOUS = 1


class RiskCategory(StrEnum):
    """Security categories used to explain why content was flagged."""

    INSTRUCTION_OVERRIDE = "instruction_override"
    ROLE_MANIPULATION = "role_manipulation"
    TOOL_MANIPULATION = "tool_manipulation"
    DATA_EXFILTRATION = "data_exfiltration"
    CONTEXT_CONFUSION = "context_confusion"
    HIDDEN_INSTRUCTION = "hidden_instruction"


class RiskDecision(StrEnum):
    """Final policy decision after detection."""

    ALLOW = "ALLOW"
    ALLOW_WITH_WARNING = "ALLOW_WITH_WARNING"
    SANITIZE = "SANITIZE"
    BLOCK = "BLOCK"


class RiskLevel(StrEnum):
    """Human-readable risk level for reports and API responses."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


LABEL_TO_NAME: dict[int, str] = {
    InjectionLabel.BENIGN.value: "benign",
    InjectionLabel.SUSPICIOUS.value: "suspicious",
}

NAME_TO_LABEL: dict[str, InjectionLabel] = {
    "benign": InjectionLabel.BENIGN,
    "suspicious": InjectionLabel.SUSPICIOUS,
}

CATEGORY_DISPLAY_NAMES: dict[RiskCategory, str] = {
    RiskCategory.INSTRUCTION_OVERRIDE: "Instruction Override",
    RiskCategory.ROLE_MANIPULATION: "Role Manipulation",
    RiskCategory.TOOL_MANIPULATION: "Tool Manipulation",
    RiskCategory.DATA_EXFILTRATION: "Data Exfiltration",
    RiskCategory.CONTEXT_CONFUSION: "Context Confusion",
    RiskCategory.HIDDEN_INSTRUCTION: "Hidden Instruction",
}


def label_to_name(label: int | InjectionLabel) -> str:
    """Convert a numeric label into a readable label name."""

    label_value = int(label)

    if label_value not in LABEL_TO_NAME:
        raise ValueError(f"Unknown injection label: {label_value}")

    return LABEL_TO_NAME[label_value]


def name_to_label(name: str) -> InjectionLabel:
    """Convert a readable label name into an InjectionLabel."""

    normalized = name.strip().lower()

    if normalized not in NAME_TO_LABEL:
        raise ValueError(f"Unknown injection label name: {name}")

    return NAME_TO_LABEL[normalized]


def category_display_name(category: RiskCategory | str) -> str:
    """Return a clean display name for a risk category."""

    risk_category = RiskCategory(category)
    return CATEGORY_DISPLAY_NAMES[risk_category]


__all__ = [
    "CATEGORY_DISPLAY_NAMES",
    "LABEL_TO_NAME",
    "NAME_TO_LABEL",
    "InjectionLabel",
    "RiskCategory",
    "RiskDecision",
    "RiskLevel",
    "category_display_name",
    "label_to_name",
    "name_to_label",
]
