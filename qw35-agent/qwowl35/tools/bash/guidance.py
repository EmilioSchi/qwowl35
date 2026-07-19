"""System-prompt guidance for the shell tool (`run_shell_command`).

Kept next to the implementation so the tool owns its own prompt text; the
registry collects it into the system prompt dynamically.
"""

from __future__ import annotations

# Shared skeleton; the two write-feedback bullets differ per agent dialect
# (hashline anchors for the NORMAL agent vs the `edit` sub-agent delegator for
# the freestyle executor), so they are substituted below rather than patched
# with fragile string .replace() at the call sites.
_BASE = """\
Use run_shell_command for listing, searching, creating files, and running
commands. `is_background` is required: false for one-time commands, true for
long-running processes (servers, watchers) that must not block further work.

Shell command style:
- Combine related operations in one command using `&&` and `|`.
- Search multiple patterns at once with `grep -E 'a|b|c'`.
- Use single quotes for status markers, for example `echo 'DONE: description'`.
<<CHANGE_FILE_BULLET>>
- If a result ends with a `Syntax check (bash)` block, the command's shell syntax
  is malformed (often an unterminated quote or heredoc); fix it and re-run.
<<WRITE_FEEDBACK_BULLET>>
- Long results are auto-compressed; a `[compressed: ...]` tail reports what was
  elided — re-call the same tool with `compress:false` only if you truly need
  everything.

Example:
<tool_call>
<function=run_shell_command>
<parameter=command>touch file.py</parameter>
<parameter=is_background>false</parameter>
</function>
</tool_call>"""

# NORMAL agent: hashline edit tools, anchor-id write feedback.
GUIDANCE = _BASE.replace(
    "<<CHANGE_FILE_BULLET>>",
    """\
- Create a file once with the shell. To CHANGE a file that already exists, do
  NOT `cat >` rewrite it — use the edit tools to fix only the wrong lines.
  Re-emitting the whole file repeats mistakes and wastes time.""",
).replace(
    "<<WRITE_FEEDBACK_BULLET>>",
    """\
- When a command writes a file via `>` or `>>`, the result includes that file's
  line ids and a `Syntax check (<lang>)` block. If it lists errors, fix each one
  with edit using the provided `edit id:` anchors — do NOT rewrite the file.""",
)

# Freestyle executor: file changes go through the `edit` sub-agent delegator
# (plain line numbers), so write feedback is a validation report, not anchors.
GUIDANCE_EXECUTOR = _BASE.replace(
    "<<CHANGE_FILE_BULLET>>",
    """\
- Create a file once with bash. To CHANGE a file that already exists, use the
  `edit` tool instead of a `cat >` rewrite — re-emitting the whole file repeats
  mistakes and wastes time.""",
).replace(
    "<<WRITE_FEEDBACK_BULLET>>",
    """\
- When a command writes a source file via `>` or `>>`, the result ends with a
  `Syntax check (<lang>)` block for that file. If it lists errors, fix ONLY
  those lines with the `edit` tool — do NOT rewrite the file through bash.""",
)
