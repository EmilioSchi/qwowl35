"""System-prompt guidance for the id-based file tools (read_file/replace/insert/delete).

Kept next to the file-tool adapter so the tool owns its own prompt text. The
line-id edit contract here is guidance only; it changes no tool logic.
"""

from __future__ import annotations

GUIDANCE = """\
read_file output is line<hash>|content. Copy only the line<hash> id, for
example from `12af|    return value` use `12af`; ranges are `12af..189c`. A range
is inclusive of both endpoints: `12af..189c` covers lines 12 through 18,
including both line 12 and line 18.

Use read_file to open a file and get its line ids before editing; it takes an
absolute file_path, and big files can be paged with offset (0-based) and
limit. replace, insert, and delete each require a prior read_file on the file
in this session; calls without it are denied, because that output is the only
source of valid line ids.

Use the file edit tools for existing non-empty files:
- replace: replace one line or range by id in a file that already exists and is non-empty.
- insert: insert before/after one id in a file that already exists and is non-empty; default after.
  Do not use range ids for insert. To insert between two lines, use the first
  line's id with position=after or the second line's id with position=before.
- delete: delete one line or range by id from a file that already exists and is non-empty.

For existing files, read_file for ids first and prefer replace, insert, or
delete over shell rewrites or replacing the whole file. Do not use empty ids.
Mutations cannot create files or target empty files; create files with bash
directly. Content is literal, so include exact indentation. After each mutation,
use the fresh ids it returns for further edits. If an id is stale or ambiguous,
read_file again and retry with a fresh id.

If a read_file or replace result ends with a `Syntax check (...)` block, the
file does not parse: fix the reported line(s) before relying on it or moving on.

File tool examples:
Open a file for editing (read its line ids; file_path is absolute):
<tool_call>
<function=read_file>
<parameter=file_path>/absolute/path/to/path.py</parameter>
</function>
</tool_call>

Replace one line:
<tool_call>
<function=replace>
<parameter=file>path.py</parameter>
<parameter=id>12af</parameter>
<parameter=content>    return 2</parameter>
</function>
</tool_call>

Replace a range:
<tool_call>
<function=replace>
<parameter=file>path.py</parameter>
<parameter=id>12af..189c</parameter>
<parameter=content>replacement</parameter>
</function>
</tool_call>

Insert after:
<tool_call>
<function=insert>
<parameter=file>path.py</parameter>
<parameter=id>12af</parameter>
<parameter=content>new line</parameter>
<parameter=position>after</parameter>
</function>
</tool_call>

Insert before:
<tool_call>
<function=insert>
<parameter=file>path.py</parameter>
<parameter=id>12af</parameter>
<parameter=content>new line</parameter>
<parameter=position>before</parameter>
</function>
</tool_call>

Delete one line:
<tool_call>
<function=delete>
<parameter=file>path.py</parameter>
<parameter=id>12af</parameter>
</function>
</tool_call>

Delete a range:
<tool_call>
<function=delete>
<parameter=file>path.py</parameter>
<parameter=id>12af..189c</parameter>
</function>
</tool_call>"""
