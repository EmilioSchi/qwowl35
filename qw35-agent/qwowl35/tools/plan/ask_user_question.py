"""`ask_user_question` — qwen-code askUserQuestion.ts replica (schema).

Execution goes through the orchestrator's question callback (a modal in the
TUI); this module owns only the wire schema and argument validation. Schema
text and validation error strings mirror qwen-code commit 14993b1 verbatim,
except the plan-mode note (approval here goes through the `plan` call, not
ExitPlanMode) and the 12-char header cap, which stays schema guidance only —
our TUI renders the header inline, so a long one is harmless and rejecting
the call would force a full re-generation. The `metadata` field is accepted
for wire parity but ignored
(qwen-code uses it for analytics/plan-gate state we don't have).
"""

from __future__ import annotations

ASK_USER_QUESTION_SCHEMA = {
    "name": "ask_user_question",
    "description": (
        "Use this tool when you need to ask the user questions during "
        "execution. This allows you to:\n"
        "1. Gather user preferences or requirements\n"
        "2. Clarify ambiguous instructions\n"
        "3. Get decisions on implementation choices as you work\n"
        "4. Offer choices to the user about what direction to take.\n"
        "\n"
        "Usage notes:\n"
        '- Users will always be able to select "Other" to provide custom '
        "text input\n"
        "- Use multiSelect: true to allow multiple answers to be selected "
        "for a question\n"
        "- If you recommend a specific option, make that the first option "
        'in the list and add "(Recommended)" at the end of the label\n'
        "\n"
        "Plan mode note: In plan mode, use this tool to clarify requirements "
        "or choose between approaches BEFORE finalizing your plan. Do NOT "
        'use this tool to ask "Is this plan ready?" or "Should I proceed?" '
        "- use the `plan` call for plan approval.\n"
    ),
    "parameters": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "questions": {
                "description": "Questions to ask the user (1-4 questions)",
                "minItems": 1,
                "maxItems": 4,
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "description": (
                                "The complete question to ask the user. Should "
                                "be clear, specific, and end with a question "
                                'mark. Example: "Which library should we use '
                                'for date formatting?" If multiSelect is true, '
                                'phrase it accordingly, e.g. "Which features '
                                'do you want to enable?"'
                            ),
                            "type": "string",
                        },
                        "header": {
                            "description": (
                                "Very short label displayed as a chip/tag (max "
                                '12 chars). Examples: "Auth method", '
                                '"Library", "Approach".'
                            ),
                            "type": "string",
                        },
                        "options": {
                            "description": (
                                "The available choices for this question. Must "
                                "have 2-4 options. Each option should be a "
                                "distinct, mutually exclusive choice (unless "
                                "multiSelect is enabled). There should be no "
                                "'Other' option, that will be provided "
                                "automatically."
                            ),
                            "minItems": 2,
                            "maxItems": 4,
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "label": {
                                        "description": (
                                            "The display text for this option "
                                            "that the user will see and "
                                            "select. Should be concise (1-5 "
                                            "words) and clearly describe the "
                                            "choice."
                                        ),
                                        "type": "string",
                                    },
                                    "description": {
                                        "description": (
                                            "Explanation of what this option "
                                            "means or what will happen if "
                                            "chosen. Useful for providing "
                                            "context about trade-offs or "
                                            "implications."
                                        ),
                                        "type": "string",
                                    },
                                },
                                "required": ["label", "description"],
                                "additionalProperties": False,
                            },
                        },
                        "multiSelect": {
                            "description": (
                                "Set to true to allow the user to select "
                                "multiple options instead of just one. Use "
                                "when choices are not mutually exclusive."
                            ),
                            "default": False,
                            "type": "boolean",
                        },
                    },
                    "required": ["question", "header", "options"],
                    "additionalProperties": False,
                },
            },
            "metadata": {
                "description": (
                    "Optional metadata for tracking and analytics purposes. "
                    "Not displayed to user."
                ),
                "type": "object",
                "properties": {
                    "source": {
                        "description": (
                            "Optional identifier for the source of this "
                            'question (e.g., "remember" for /remember '
                            "command). Used for analytics tracking."
                        ),
                        "type": "string",
                    },
                },
                "additionalProperties": False,
            },
        },
        "required": ["questions"],
        "additionalProperties": False,
    },
}


def _is_blank(value: object) -> bool:
    return not isinstance(value, str) or value.strip() == ""


def validate_questions(arguments: dict) -> str | None:
    """Returns an error string for the model, or None when the args are usable."""
    questions = arguments.get("questions")
    if not isinstance(questions, list):
        return 'Parameter "questions" must be an array.'
    if not (1 <= len(questions) <= 4):
        return 'Parameter "questions" must contain between 1 and 4 questions.'
    for index, question in enumerate(questions):
        n = index + 1
        if not isinstance(question, dict) or _is_blank(question.get("question")):
            return f'Question {n}: "question" must be a non-empty string.'
        if _is_blank(question.get("header")):
            return f'Question {n}: "header" must be a non-empty string.'
        options = question.get("options")
        if not isinstance(options, list):
            return f'Question {n}: "options" must be an array.'
        if not (2 <= len(options) <= 4):
            return f'Question {n}: "options" must contain between 2 and 4 options.'
        for opt_index, option in enumerate(options):
            m = opt_index + 1
            if not isinstance(option, dict) or _is_blank(option.get("label")):
                return (
                    f'Question {n}, Option {m}: "label" must be a non-empty string.'
                )
            if _is_blank(option.get("description")):
                return (
                    f'Question {n}, Option {m}: "description" must be a '
                    "non-empty string."
                )
        multi_select = question.get("multiSelect")
        if multi_select is not None and not isinstance(multi_select, bool):
            return f'Question {n}: "multiSelect" must be a boolean.'
    return None
