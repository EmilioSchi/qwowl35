"""System-prompt guidance for the bash tool.

Kept next to the bash implementation so the tool owns its own prompt text; the
registry collects it into the system prompt dynamically.
"""

from __future__ import annotations

GUIDANCE = """\
Use bash for listing, searching, creating files, and running commands.

Bash command style:
- Combine related operations in one command using `&&` and `|`.
- Search multiple patterns at once with `grep -E 'a|b|c'`.
- Use single quotes for status markers, for example `echo 'DONE: description'`.
- Create a file once with bash. To CHANGE a file that already exists, do NOT
  `cat >` rewrite it — run `read`, then `edit` to fix
  only the wrong lines. Re-emitting the whole file repeats mistakes and wastes time.
- If a result ends with a `Syntax check (bash)` block, the command's shell syntax
  is malformed (often an unterminated quote or heredoc); fix it and re-run.

Bash example:
<tool_call>
<function=bash>
<parameter=command>touch file.py</parameter>
</function>
</tool_call>"""
