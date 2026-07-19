"""LSP layer (multilspy): edit-time diagnostics + the `lsp` navigation tool.

Diagnostics: :func:`configure` (Config.lsp master switch),
:func:`supported_language`, :func:`lsp_check_file`, :func:`shutdown_all`.
Everything degrades: a missing ``multilspy`` install, a missing language-server
binary, a booting server, or a diagnostics timeout all yield ``None`` from
:func:`lsp_check_file`, and the router in ``tools/syntax/validate.py`` falls
back to the tree-sitter checker.

Navigation: :class:`LspQueryTool` (wire name :data:`LSP_NAME`) exposes
goToDefinition / findReferences / hover / documentSymbol with qwen-code's tool
interface, reusing the same per-language servers via :func:`get_ready_server`.
"""

from __future__ import annotations

from .diagnostics import (
    configure,
    get_ready_server,
    is_enabled,
    lsp_check_file,
    shutdown_all,
    supported_language,
)
from .query import GUIDANCE as LSP_GUIDANCE
from .query import LSP_NAME, LSP_SCHEMA, LspQueryTool

__all__ = [
    "configure",
    "get_ready_server",
    "is_enabled",
    "lsp_check_file",
    "shutdown_all",
    "supported_language",
    "LSP_GUIDANCE",
    "LSP_NAME",
    "LSP_SCHEMA",
    "LspQueryTool",
]
