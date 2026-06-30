"""The result of a bash-approval prompt.

Kept in its own module so the modal (`widgets/approval.py`), the registry
(`tools_registry.py`), and the app can share the type without an import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class ApprovalDecision:
    kind: Literal["accept", "deny", "alternative"]
    text: str = ""  # the instruction to relay to the model when kind == "alternative"
