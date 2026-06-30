"""Risk assessment helpers, mirroring hashline's ``risk.rs``."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RiskLevel(Enum):
    Low = "low"
    Medium = "medium"
    High = "high"

    def as_str(self) -> str:
        return self.value


class RiskReason(Enum):
    MutatesFile = "mutates_file"
    DeletesContent = "deletes_content"
    AppliesPatch = "applies_patch"
    Unknown = "unknown"


@dataclass(frozen=True)
class RiskAssessment:
    level: RiskLevel
    reasons: list[RiskReason]


def assess_command(command) -> RiskAssessment | None:
    name = command if isinstance(command, str) else command.__class__.__name__.lower()
    if "delete" in name:
        return RiskAssessment(RiskLevel.High, [RiskReason.DeletesContent])
    if any(word in name for word in ("edit", "insert", "patch", "replace", "move", "swap", "indent")):
        return RiskAssessment(RiskLevel.Medium, [RiskReason.MutatesFile])
    return None


def blocked_assessment(error: Exception) -> RiskAssessment | None:
    text = str(error).lower()
    if "delete" in text or "overwrite" in text:
        return RiskAssessment(RiskLevel.High, [RiskReason.Unknown])
    return None
