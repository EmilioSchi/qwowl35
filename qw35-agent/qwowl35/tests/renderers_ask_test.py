"""Tests for the ask_user_question card."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from widgets.chat import ChatView, ToolBlock  # noqa: E402

import theme  # noqa: E402

from chat_test_helpers import _ansi, _plain, assert_true  # noqa: E402


_ASK_QUESTIONS = [
    {
        "question": "Which database should we use?",
        "header": "Database",
        "options": [
            {"label": "SQLite", "description": "d"},
            {"label": "Postgres", "description": "d"},
        ],
    },
    {
        "question": "Which auth method?",
        "header": "Auth",
        "options": [
            {"label": "JWT", "description": "d"},
            {"label": "OAuth", "description": "d"},
        ],
    },
]


def _ask_block(view: ChatView) -> ToolBlock:
    import json

    block = ToolBlock("ask_user_question")
    block.args_buf = json.dumps({"questions": _ASK_QUESTIONS})
    block.stream_done = True
    block.update = lambda *_: None  # type: ignore[method-assign]
    view._tool_blocks[0] = block
    view._bump = lambda: None  # type: ignore[method-assign]
    return block


def test_shimmer_text_animates_with_single_dim_bounce() -> None:
    from widgets.chat.renderers.ask import _ASK_LABEL
    from widgets.chat.primitives import _shimmer_text

    frame0 = _ansi(_shimmer_text(_ASK_LABEL, 0), width=40)
    frame1 = _ansi(_shimmer_text(_ASK_LABEL, 1), width=40)
    assert_true(frame0 != frame1, "consecutive frames differ (hue drift + bounce)")

    # The dim span's start index ping-pongs 0..n-1..0 across the label.
    def bounce_index(frame: int) -> int:
        text = _shimmer_text(_ASK_LABEL, frame)
        starts = [span.start for span in text.spans if "dim" in str(span.style)]
        assert_true(len(starts) == 1, f"exactly one dim span: {starts}")
        return starts[0]

    period = 2 * (len(_ASK_LABEL) - 1)
    assert_true(bounce_index(3) == 3, "bounce advances with the frame")
    assert_true(bounce_index(period - 3) == 3, "bounce reflects back after the last char")


def test_ask_call_card_lists_questions_with_guides() -> None:
    view = ChatView()
    block = _ask_block(view)
    plain = _plain(view._render_tool_call(block))
    assert_true("~ Asking questions..." in plain, f"shimmer label present: {plain}")
    assert_true(" ├─ [Database] Which database should we use?" in plain,
                f"first question under a branch guide: {plain}")
    assert_true(" └─ [Auth] Which auth method?" in plain,
                f"last question under the closing guide: {plain}")
    # The "Ask" badge row is suppressed — the ask card has its own custom
    # header ("~ Asking questions...") instead of the generic badge row.
    assert_true(not plain.splitlines()[0].startswith("Ask"), f"no generic badge row: {plain}")
    assert_true("Result" not in plain, "no Result box")


def test_ask_call_card_shows_and_grows_during_emission() -> None:
    # The card is live from the first streamed token: shimmer label alone,
    # then the tree grows as each question/header string lands in the buffer.
    view = ChatView()
    block = ToolBlock("ask_user_question")
    block.update = lambda *_: None  # type: ignore[method-assign]
    view._tool_blocks[0] = block
    view._bump = lambda: None  # type: ignore[method-assign]

    block.args_buf = '{"questions": [{"'
    plain = _plain(view._render_tool_call(block))
    assert_true("~ Asking questions..." in plain, f"label shows before any question: {plain}")
    assert_true(not plain.splitlines()[0].startswith("Ask"), f"no generic badge during emission: {plain}")

    block.args_buf = '{"questions": [{"question": "Which database sho'
    plain = _plain(view._render_tool_call(block))
    assert_true(" └─ Which database sho" in plain, f"partial question grows in: {plain}")

    block.args_buf = (
        '{"questions": [{"question": "Which database should we use?", '
        '"header": "Database", "options": [{"label": "SQLite", "description": "d"}], '
        '"multiSelect": false}, {"question": "Which auth'
    )
    plain = _plain(view._render_tool_call(block))
    assert_true(" ├─ [Database] Which database should we use?" in plain,
                f"finished question keeps its header chip: {plain}")
    assert_true(" └─ Which auth" in plain, f"next question streams onto a new row: {plain}")
    assert_true("SQLite" not in plain, "option labels stay out of the tree")

    from widgets.chat.chat_view import _THINK_FRAME_TICKS

    for _ in range(2 * _THINK_FRAME_TICKS):
        view._tick()
    assert_true(block.anim_frame == 2, f"shimmer animates during emission: {block.anim_frame}")


def test_ask_partial_questions_recovers_from_raw_xml_dialect() -> None:
    from widgets.chat.renderers.ask import _ask_partial_questions

    xml = (
        "\n<function=ask_user_question>\n<parameter=questions>\n"
        '[{"question": "Say \\"hi\\"?", "header": "Greeting"}, {"question": "And'
    )
    parsed = _ask_partial_questions(xml)
    assert_true(parsed == [{"question": 'Say "hi"?', "header": "Greeting"},
                           {"question": "And"}],
                f"escapes decoded, headers paired in order: {parsed}")
    assert_true(_ask_partial_questions('{"questions": [') == [], "no strings yet -> no rows")


def test_ask_card_grows_answers_and_skips() -> None:
    view = ChatView()
    block = _ask_block(view)
    view.note_question_answer(0, "SQLite")
    view.note_question_answer(1, None)
    lines = _plain(view._render_tool_call(block)).splitlines()
    first = next(i for i, line in enumerate(lines) if "Which database" in line)
    assert_true(lines[first + 1].startswith(" │  → SQLite"),
                f"answer nested under its question with a continuation stem: {lines}")
    last = next(i for i, line in enumerate(lines) if "Which auth" in line)
    assert_true(lines[last + 1].startswith("    (skipped)"),
                f"dismissed modal leaves (skipped) under the last question: {lines}")

    # No live ask block: the notification is display-only and must no-op.
    ChatView().note_question_answer(0, "ignored")


def test_ask_card_tick_advances_frames_only_while_live() -> None:
    from widgets.chat.chat_view import _THINK_FRAME_TICKS

    view = ChatView()
    block = _ask_block(view)
    for _ in range(2 * _THINK_FRAME_TICKS):
        view._tick()
    assert_true(block.anim_frame == 2,
                f"label advances every {_THINK_FRAME_TICKS} ticks: {block.anim_frame}")
    view.add_tool_result(0, "ask_user_question", "User declined to answer the questions.")
    for _ in range(2 * _THINK_FRAME_TICKS):
        view._tick()
    assert_true(block.anim_frame == 2, "animation stops once the result lands")
    assert_true(0 not in view._tool_blocks, "result promoted the block out of the live map")


def test_parse_ask_result_shapes() -> None:
    from widgets.chat.renderers.ask import _parse_ask_result

    pairs = _parse_ask_result(
        "User has provided the following answers:\n\n"
        "**Database**: SQLite\n**Auth**: JWT, OAuth"
    )
    assert_true(pairs == [("Database", "SQLite"), ("Auth", "JWT, OAuth")],
                f"answer lines parsed: {pairs}")
    assert_true(_parse_ask_result("User declined to answer the questions.") == [],
                "declined marker parses to no pairs")
    for body in (
        "Error: ask_user_question is no longer available — the plan has "
        "already been approved. Continue with the `plan` tool.",
        'Parameter "questions" must be an array.',
        "User has provided the following answers:\n\nnot an answer line",
        "arbitrary text",
    ):
        assert_true(_parse_ask_result(body) is None, f"unknown shape rejected: {body!r}")


def test_ask_result_renders_frozen_answer_tree() -> None:
    view = ChatView()
    block = _ask_block(view)
    block.full_result = (
        "User has provided the following answers:\n\n"
        "**Database**: SQLite\n**Auth**: OAuth"
    )
    text = _plain(view._render_tool_result(block))
    assert_true("~ Asking questions..." in text, f"label kept at rest: {text}")
    assert_true(" ├─ Database: SQLite" in text and " └─ Auth: OAuth" in text,
                f"header: answer rows under guides: {text}")
    assert_true("Which database" not in text, "question texts dropped once frozen")
    assert_true("Result" not in text, "generic Result label replaced by the tree")

    ansi = _ansi(view._render_tool_result(block), width=100)
    assert_true("\x1b[2;" not in ansi and "\x1b[2m" not in ansi, "no dim bounce when frozen")
    faint = theme.FG_FAINT.lstrip("#")
    triplet = ";".join(str(int(faint[i:i + 2], 16)) for i in (0, 2, 4))
    assert_true(triplet in ansi, f"frozen label uses theme FG_FAINT: {ansi!r}")


def test_ask_result_declined_and_error_fallback() -> None:
    view = ChatView()
    block = _ask_block(view)
    block.full_result = "User declined to answer the questions."
    text = _plain(view._render_tool_result(block))
    assert_true(" └─ (declined — no answers)" in text, f"declined row rendered: {text}")
    assert_true("Result" not in text, "declined still uses the frozen card")

    err = _ask_block(ChatView())
    err.full_result = 'Parameter "questions" must be an array.'
    err.is_error = True
    err_text = _plain(ChatView()._render_tool_result(err))
    assert_true("Result" in err_text and "must be an array" in err_text,
                "error keeps the plain fallback box")


def main() -> None:
    test_shimmer_text_animates_with_single_dim_bounce()
    test_ask_call_card_lists_questions_with_guides()
    test_ask_call_card_shows_and_grows_during_emission()
    test_ask_partial_questions_recovers_from_raw_xml_dialect()
    test_ask_card_grows_answers_and_skips()
    test_ask_card_tick_advances_frames_only_while_live()
    test_parse_ask_result_shapes()
    test_ask_result_renders_frozen_answer_tree()
    test_ask_result_declined_and_error_fallback()
    print("ask renderer tests passed")


if __name__ == "__main__":
    main()
