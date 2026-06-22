"""Prove the server's tools system block matches the model's training format.

The qw35-server renders the `# Tools ... </IMPORTANT>` system block in Rust
(`toolcall::render_tools_system_block`). Its unit test pins that output to an
EXPECTED string. This script closes the loop with the *weights*: it reads the
`tokenizer.chat_template` embedded in the GGUF, renders it with jinja2 exactly
the way `transformers.apply_chat_template` does, and asserts the tools section it
produces equals the same EXPECTED string. So:

    Rust renderer  ==  Rust unit-test pin  ==  jinja(chat_template)  ==  weights

If the model ships with a different tool format in a future build, this fails and
tells us to re-derive the Rust renderer — we never trust web docs, only the GGUF.

Usage:
    python3 verify_tool_template.py [--gguf PATH]
Exit code 0 on match, 1 on mismatch (prints a unified diff).
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path

DEFAULT_GGUF = Path(__file__).resolve().parents[2] / ".gguf" / "Qwen3.5-9B-Q4_K_M.gguf"

# The single fixture tool, in the exact OpenAI nested shape qwowl35 sends. Both
# the server (dumping ToolDef.raw) and jinja (`tool | tojson`) serialize this.
FIXTURE_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature units",
                },
            },
            "required": ["city"],
        },
    },
}

# Must stay identical to the EXPECTED string in
# qw35-server/src/toolcall.rs::renders_tools_block_and_tool_call_block.
EXPECTED = (
    "# Tools\n\nYou have access to the following functions:\n\n<tools>\n"
    + json.dumps(FIXTURE_TOOL, ensure_ascii=False)
    + "\n</tools>\n\n"
    "If you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
    "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\n"
    "value_1\n</parameter>\n<parameter=example_parameter_2>\n"
    "This is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n"
    "</function>\n</tool_call>\n\n"
    "<IMPORTANT>\nReminder:\n"
    "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n"
    "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
    "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n"
    "</IMPORTANT>"
)


# A historical assistant tool call whose argument value contains the XML-special
# characters `<`, `>`, `&`. The model's embedded chat_template renders parameter
# values VERBATIM (it has no `|e` escape filter), so the server's
# `render_tool_call_block` must too — escaping them was off-distribution and made
# the model loop converting `>`->`&gt;`. This pins jinja(template) == EXPECTED, and
# the Rust unit test `tool_call_values_render_verbatim_not_escaped` pins
# render_tool_call_block == EXPECTED, closing the loop to the weights.
TOOL_CALL_VALUE = "if a > b and c < d and e & f"

EXPECTED_TOOL_CALL = (
    "<tool_call>\n"
    "<function=edit>\n"
    "<parameter=content>\n"
    f"{TOOL_CALL_VALUE}\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>"
)

EDIT_TOOL = {
    "type": "function",
    "function": {
        "name": "edit",
        "parameters": {
            "type": "object",
            "properties": {"content": {"type": "string"}},
        },
    },
}


def read_chat_template(gguf_path: Path) -> str:
    """Extract tokenizer.chat_template from the GGUF, reusing the gguf reader."""
    from gguf import GGUFReader

    reader = GGUFReader(str(gguf_path), "r")
    field = reader.get_field("tokenizer.chat_template")
    if field is None:
        raise SystemExit("tokenizer.chat_template not found in GGUF metadata")
    # A string field's last part is the bytes of the value.
    return str(field.contents())


def render_template(template_text: str) -> str:
    """Render the chat_template the way transformers.apply_chat_template does."""
    from jinja2 import Environment
    from jinja2.exceptions import TemplateError

    def raise_exception(message: str):
        raise TemplateError(message)

    def tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(
            value,
            ensure_ascii=ensure_ascii,
            indent=indent,
            separators=separators,
            sort_keys=sort_keys,
        )

    env = Environment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    template = env.from_string(template_text)
    return template.render(
        messages=[{"role": "user", "content": "What's the weather in Paris?"}],
        tools=[FIXTURE_TOOL],
        add_generation_prompt=True,
        enable_thinking=False,
        add_vision_id=False,
    )


def render_template_tool_call(template_text: str) -> str:
    """Render the chat_template with a conversation containing an assistant tool
    call, so we can pin how the template renders historical tool-call args."""
    from jinja2 import Environment
    from jinja2.exceptions import TemplateError

    def raise_exception(message: str):
        raise TemplateError(message)

    def tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(
            value,
            ensure_ascii=ensure_ascii,
            indent=indent,
            separators=separators,
            sort_keys=sort_keys,
        )

    env = Environment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    template = env.from_string(template_text)
    # ``arguments`` as a dict: transformers parses the JSON before applying the
    # template, so this matches how the model saw it during training.
    return template.render(
        messages=[
            {"role": "user", "content": "fix it"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c0",
                        "type": "function",
                        "function": {"name": "edit", "arguments": {"content": TOOL_CALL_VALUE}},
                    }
                ],
            },
            {"role": "user", "content": "go"},
        ],
        tools=[EDIT_TOOL],
        add_generation_prompt=True,
        enable_thinking=False,
        add_vision_id=False,
    )


def extract_tools_section(prompt: str) -> str:
    start = prompt.index("# Tools")
    end = prompt.index("</IMPORTANT>") + len("</IMPORTANT>")
    return prompt[start:end]


def extract_tool_call(prompt: str) -> str:
    """The assistant's rendered <tool_call>...</tool_call> (not the system example)."""
    fn = prompt.index("<function=edit>")
    start = prompt.rindex("<tool_call>", 0, fn)
    end = prompt.index("</tool_call>", fn) + len("</tool_call>")
    return prompt[start:end]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    args = parser.parse_args()

    if not args.gguf.exists():
        raise SystemExit(f"GGUF not found: {args.gguf}")

    template_text = read_chat_template(args.gguf)

    failures = 0

    actual_tools = extract_tools_section(render_template(template_text))
    if actual_tools == EXPECTED:
        print("PASS: GGUF chat_template tools section matches the server's EXPECTED block.")
    else:
        failures += 1
        print("FAIL: GGUF chat_template tools section differs from EXPECTED.\n")
        print(
            "\n".join(
                difflib.unified_diff(
                    EXPECTED.splitlines(),
                    actual_tools.splitlines(),
                    fromfile="EXPECTED (server/toolcall.rs)",
                    tofile="chat_template (GGUF)",
                    lineterm="",
                )
            )
        )

    actual_call = extract_tool_call(render_template_tool_call(template_text))
    if actual_call == EXPECTED_TOOL_CALL:
        print(
            "PASS: GGUF chat_template renders tool-call argument values VERBATIM "
            "(matches render_tool_call_block — no entity escaping)."
        )
    else:
        failures += 1
        print("FAIL: GGUF chat_template tool-call render differs from EXPECTED_TOOL_CALL.\n")
        print(
            "\n".join(
                difflib.unified_diff(
                    EXPECTED_TOOL_CALL.splitlines(),
                    actual_call.splitlines(),
                    fromfile="EXPECTED_TOOL_CALL (server/toolcall.rs render)",
                    tofile="chat_template (GGUF)",
                    lineterm="",
                )
            )
        )

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
