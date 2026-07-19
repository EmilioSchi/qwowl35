"""The catalog of local ``/``-commands, shared by the command palette and its
drift-guard test.

This is a *parallel* source of truth to :meth:`app.QwowlApp._dispatch_command`,
not the dispatcher itself: the dispatcher stays a hand-written if/elif whose
exact matching is pinned by tests, and this catalog only describes the commands
for discovery (the palette) and documentation. ``tests/command_palette_test.py``
asserts the two never drift — every name and alias here must still route through
the dispatcher.

``filter_commands`` does a case-insensitive **prefix** match on the command name
and its aliases (the leading ``/`` is ignored), so typing ``/se`` narrows to
``/sessions``. It is pure and unit-tested in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CommandSpec:
    name: str                        # canonical, with the leading slash: "/quit"
    aliases: tuple[str, ...] = ()    # equivalent spellings, e.g. ("/exit", ...)
    description: str = ""            # one line of discovery text
    takes_args: bool = False        # e.g. /mode <name> — accept fills, never runs bare
    arg_hint: str = ""              # shown after the name, e.g. "[normal|plan|web|chat]"


# Order here is the order shown in the palette (and returned for an empty query).
COMMANDS: tuple[CommandSpec, ...] = (
    CommandSpec("/clear", (), "start a new conversation"),
    CommandSpec(
        "/mode", (), "switch mode, or cycle through them",
        takes_args=True, arg_hint="[normal|plan|web|chat]",
    ),
    CommandSpec("/theme", (), "pick a color theme"),
    CommandSpec("/sessions", (), "restore a past session"),
    CommandSpec("/quit", ("/exit", "/abort", "/close"), " "),
)


def filter_commands(
    query: str, specs: tuple[CommandSpec, ...] = COMMANDS
) -> list[CommandSpec]:
    """The specs whose name or any alias starts with ``query`` (case-insensitive,
    leading ``/`` ignored on both sides). An empty/blank query returns them all
    in catalog order; catalog order is preserved among matches."""
    needle = query.strip().lstrip("/").lower()
    if not needle:
        return list(specs)
    matches = []
    for spec in specs:
        names = (spec.name, *spec.aliases)
        if any(name.lstrip("/").lower().startswith(needle) for name in names):
            matches.append(spec)
    return matches
