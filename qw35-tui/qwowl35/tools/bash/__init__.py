"""Bash tool with suspicious-command parsing.

Ports the Go reference bash tool and its approval analyzer into Python.
"""

from .analyzer import (
    AnalysisResult,
    ApprovalPromptOptions,
    PatternType,
    SUSPICIOUS_PATTERNS,
    SuspiciousPattern,
    analyze_command,
    append_write_targets,
    build_bash_approval_options,
    describe_bash_allowlist,
    extract_bash_prefix,
    is_command_outside_cwd,
    is_suspicious,
    parse_bash,
    truncating_write_targets,
)
from .executor import BASH_TIMEOUT_SECONDS, MAX_OUTPUT_SIZE, BashTool, CappedBuffer
from .guidance import GUIDANCE

__all__ = [
    "AnalysisResult",
    "ApprovalPromptOptions",
    "PatternType",
    "SUSPICIOUS_PATTERNS",
    "SuspiciousPattern",
    "BashTool",
    "CappedBuffer",
    "BASH_TIMEOUT_SECONDS",
    "MAX_OUTPUT_SIZE",
    "GUIDANCE",
    "analyze_command",
    "append_write_targets",
    "build_bash_approval_options",
    "describe_bash_allowlist",
    "extract_bash_prefix",
    "is_command_outside_cwd",
    "is_suspicious",
    "parse_bash",
    "truncating_write_targets",
]
