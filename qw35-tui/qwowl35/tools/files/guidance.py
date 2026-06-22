"""System-prompt guidance for the anchored file tools (read/edit/insert/delete).

Kept next to the file-tool adapter so the tool owns its own prompt text. The
anchored-edit contract here is guidance only; it changes no tool logic.
"""

from __future__ import annotations

GUIDANCE = """\
read output is line:hash|content. Copy only the qualified line:hash anchor,
for example from `12:af|    return value` use `12:af`; ranges are
`12:af..18:9c`. A range is inclusive of both endpoints: `12:af..18:9c` covers
lines 12 through 18, including both line 12 and line 18.

Use read to get anchors before editing.

Use file edit tools for existing non-empty files:
- edit: replace one anchored line or range in a file that already exists and is non-empty.
- insert: insert before/after one anchor in a file that already exists and is non-empty; default after.
  Do not use range anchors for insert. To insert between two lines, use the
  first line's anchor with position=after or the second line's anchor with position=before.
- delete: delete one anchored line or range from a file that already exists and is non-empty.

For existing files, read anchors first and prefer edit,
insert, or delete over shell rewrites or replacing the whole file. Do not use empty
anchors. Mutations cannot create files or target empty files; create files with
bash directly. Content is literal, so include exact indentation. After each
mutation, use the fresh anchors it returns for further edits. If an anchor is
stale or ambiguous, read again and retry with a fresh qualified anchor.

If a read or edit result ends with a `Syntax check (...)` block, the file does
not parse: fix the reported line(s) before relying on it or moving on.

File tool examples:
Read anchors:
<tool_call>
<function=read>
<parameter=file>path.py</parameter>
</function>
</tool_call>

Read anchored context:
<tool_call>
<function=read>
<parameter=file>path.py</parameter>
<parameter=anchor>12:af</parameter>
<parameter=context>4</parameter>
</function>
</tool_call>

Replace one line:
<tool_call>
<function=edit>
<parameter=file>path.py</parameter>
<parameter=anchor>12:af</parameter>
<parameter=content>    return 2</parameter>
</function>
</tool_call>

Replace a range:
<tool_call>
<function=edit>
<parameter=file>path.py</parameter>
<parameter=anchor>12:af..18:9c</parameter>
<parameter=content>replacement</parameter>
</function>
</tool_call>

Insert after:
<tool_call>
<function=insert>
<parameter=file>path.py</parameter>
<parameter=anchor>12:af</parameter>
<parameter=content>new line</parameter>
<parameter=position>after</parameter>
</function>
</tool_call>

Insert before:
<tool_call>
<function=insert>
<parameter=file>path.py</parameter>
<parameter=anchor>12:af</parameter>
<parameter=content>new line</parameter>
<parameter=position>before</parameter>
</function>
</tool_call>

Delete one line:
<tool_call>
<function=delete>
<parameter=file>path.py</parameter>
<parameter=anchor>12:af</parameter>
</function>
</tool_call>

Delete a range:
<tool_call>
<function=delete>
<parameter=file>path.py</parameter>
<parameter=anchor>12:af..18:9c</parameter>
</function>
</tool_call>"""
