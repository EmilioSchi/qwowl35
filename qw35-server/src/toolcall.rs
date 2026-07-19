//! Tool-calling support shared by the chat-completions and responses
//! endpoints: tool definitions parsed from requests, the system-prompt block
//! that advertises them in a compact Qwen3 XML format, and a streaming
//! parser that splits generated text into content, reasoning, and tool calls.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

// ── Tool definitions and prompt rendering ──────────────────────────────────

#[derive(Debug, Clone)]
pub struct ToolDef {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Option<serde_json::Value>,
    /// The original tool object as the client sent it (e.g. the chat form
    /// `{"type":"function","function":{...}}`). The system-prompt block dumps
    /// this verbatim as JSON, exactly like the chat_template's `tool | tojson`.
    pub raw: serde_json::Value,
}

/// Parses tool definitions from either wire shape: the chat-completions
/// nested form `{"type":"function","function":{...}}` or the responses-API
/// flat form `{"type":"function","name":...}`. Built-in hosted tool types
/// without a function name (e.g. `web_search`) are skipped rather than
/// rejected, since they cannot run against a local model anyway.
pub fn parse_tool_defs(tools: &serde_json::Value) -> Result<Vec<ToolDef>, String> {
    let serde_json::Value::Array(items) = tools else {
        return Err("tools must be an array of tool definitions".to_string());
    };
    let mut defs = Vec::new();
    for item in items {
        let serde_json::Value::Object(obj) = item else {
            return Err("each tool definition must be an object".to_string());
        };
        let source = match obj.get("function") {
            Some(serde_json::Value::Object(function)) => function,
            _ => obj,
        };
        let Some(name) = source.get("name").and_then(serde_json::Value::as_str) else {
            let tool_type = obj
                .get("type")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("function");
            if matches!(tool_type, "function" | "custom") {
                return Err("tool definition is missing a function name".to_string());
            }
            continue;
        };
        defs.push(ToolDef {
            name: name.to_string(),
            description: source
                .get("description")
                .and_then(serde_json::Value::as_str)
                .map(str::to_string),
            parameters: source
                .get("parameters")
                .filter(|value| !value.is_null())
                .cloned(),
            raw: item.clone(),
        });
    }
    Ok(defs)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolChoice {
    Auto,
    None,
    Required,
    Named(String),
}

/// Accepts every tool_choice shape the agents send: `"auto"`, `"none"`,
/// `"required"`/`"any"`, the chat form `{"type":"function","function":
/// {"name":..}}`, and the responses form `{"type":"function","name":..}`.
/// Unknown shapes fall back to auto instead of failing the request.
pub fn parse_tool_choice(value: Option<&serde_json::Value>) -> ToolChoice {
    let Some(value) = value else {
        return ToolChoice::Auto;
    };
    match value {
        serde_json::Value::String(choice) => match choice.as_str() {
            "none" => ToolChoice::None,
            "required" | "any" => ToolChoice::Required,
            _ => ToolChoice::Auto,
        },
        serde_json::Value::Object(obj) => {
            let name = obj
                .get("function")
                .and_then(|function| function.get("name"))
                .and_then(serde_json::Value::as_str)
                .or_else(|| obj.get("name").and_then(serde_json::Value::as_str));
            match name {
                Some(name) => ToolChoice::Named(name.to_string()),
                None => ToolChoice::Auto,
            }
        }
        _ => ToolChoice::Auto,
    }
}

/// Renders the Qwen 3.5 tools section to prepend to the system
/// prompt. Returns None when there is nothing to advertise.
///
/// Byte-for-byte identical to the model's embedded `tokenizer.chat_template`:
/// each tool is dumped as JSON (the template's `tool | tojson`) inside
/// `<tools>`, followed by the exact call-format instruction and `<IMPORTANT>`
/// reminder the model was trained on. Keeping this on-distribution is what makes
/// the model emit well-formed `<tool_call>` blocks; an `examples`-style XML
/// signature list (what this used to render) is off-distribution.
///
/// Callers gate on `ToolChoice::None` themselves (no advertisement at all);
/// the `Required`/`Named` must-call instruction is NOT part of this block —
/// it renders past the stable prompt boundary via [`enforcement_suffix`], so
/// `tool_choice` never perturbs the session-cache checkpoint prefix.
pub fn render_tools_system_block(defs: &[ToolDef]) -> Option<String> {
    if defs.is_empty() {
        return None;
    }
    let mut out = String::from("# Tools\n\nYou have access to the following functions:\n\n<tools>");
    for def in defs {
        out.push('\n');
        out.push_str(&tool_to_jinja_json(&def.raw));
    }
    out.push_str(
"
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>",
    );
    Some(out)
}

/// The `tool_choice: required`/named must-call instruction. Rendered as a
/// volatile user turn after the stable prompt boundary (see
/// `render_qwen35_chat_prompt_with_boundaries`), not inside the `<tools>`
/// system block, so forcing a call never changes the cached prefix.
pub fn enforcement_suffix(choice: &ToolChoice) -> Option<String> {
    match choice {
        ToolChoice::Required => {
            Some("You must call at least one function before answering.".to_string())
        }
        ToolChoice::Named(name) => Some(format!("You must call the function {name:?}.")),
        ToolChoice::Auto | ToolChoice::None => None,
    }
}

/// The forced tool-call opening for a named `tool_choice`: injected into the
/// prompt right after the generation header (past the stable boundary), so
/// the model can only complete the call's parameters — prose is impossible,
/// unlike the instruction-only [`enforcement_suffix`]. `required` stays
/// instruction-only: the model must still pick which function to call.
pub fn forced_call_prefix(choice: &ToolChoice) -> Option<String> {
    match choice {
        ToolChoice::Named(name) => Some(format!("<tool_call>\n<function={name}>\n")),
        ToolChoice::Required | ToolChoice::Auto | ToolChoice::None => None,
    }
}

/// Serializes a tool definition the way Jinja2's `tojson` filter does, so the
/// rendered `<tools>` block matches the chat_template byte-for-byte: `", "`
/// between elements and `": "` after keys (vs serde_json's compact `,`/`:`),
/// no indentation, keys in insertion order (serde_json's `preserve_order`),
/// non-ASCII passed through as UTF-8 (matching `ensure_ascii=False`).
fn tool_to_jinja_json(value: &serde_json::Value) -> String {
    use serde::Serialize;

    struct JinjaFormatter;
    impl serde_json::ser::Formatter for JinjaFormatter {
        fn begin_array_value<W>(&mut self, writer: &mut W, first: bool) -> std::io::Result<()>
        where
            W: ?Sized + std::io::Write,
        {
            if first {
                Ok(())
            } else {
                writer.write_all(b", ")
            }
        }
        fn begin_object_key<W>(&mut self, writer: &mut W, first: bool) -> std::io::Result<()>
        where
            W: ?Sized + std::io::Write,
        {
            if first {
                Ok(())
            } else {
                writer.write_all(b", ")
            }
        }
        fn begin_object_value<W>(&mut self, writer: &mut W) -> std::io::Result<()>
        where
            W: ?Sized + std::io::Write,
        {
            writer.write_all(b": ")
        }
    }

    let mut buf = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut buf, JinjaFormatter);
    value
        .serialize(&mut ser)
        .expect("serde_json::Value always serializes");
    String::from_utf8(buf).expect("serde_json emits valid UTF-8")
}

/// Renders a historical tool call back into the model-facing Qwen3 XML form.
/// `arguments` is the OpenAI-style JSON string; if it does not parse as an
/// object, it is kept under the legacy `arguments` key rather than dropped.
pub fn render_tool_call_block(name: &str, arguments: &str) -> String {
    let args_value: serde_json::Value = if arguments.trim().is_empty() {
        serde_json::json!({})
    } else {
        serde_json::from_str(arguments)
            .unwrap_or_else(|_| serde_json::Value::String(arguments.to_string()))
    };
    render_function_parameter_tool_call_block(name, &args_value)
}

fn render_function_parameter_tool_call_block(name: &str, args_value: &serde_json::Value) -> String {
    let mut out = String::from("<tool_call>\n");
    out.push_str(&render_xml_open_named_tag("function", name));
    out.push('\n');
    match args_value {
        serde_json::Value::Object(args) => {
            for (key, value) in args {
                out.push_str(&render_xml_parameter_block(key, value));
            }
        }
        other => out.push_str(&render_xml_parameter_block("arguments", other)),
    }
    out.push_str("</function>\n</tool_call>");
    out
}

fn render_xml_parameter_block(name: &str, value: &serde_json::Value) -> String {
    let text = match value {
        serde_json::Value::String(text) => text.clone(),
        other => other.to_string(),
    };
    // The embedded chat_template renders parameter values VERBATIM (no entity
    // escaping — it has no Jinja `|e` filter; verified against the GGUF). Escaping
    // `<`/`>`/`&` here was off-distribution and caused a loop: the model saw its own
    // prior tool-call args as `&gt;`/`&lt;`/`&amp;` while read_file showed the literal
    // characters, so it kept "fixing" `>`→`&gt;`. Render verbatim to match the model.
    format!(
        "{}\n{}\n</parameter>\n",
        render_xml_open_named_tag("parameter", name),
        text
    )
}

fn render_xml_open_named_tag(tag: &str, name: &str) -> String {
    if is_simple_xml_target(name) {
        format!("<{tag}={name}>")
    } else {
        format!("<{tag} name=\"{}\">", xml_escape_attr(name))
    }
}

fn is_simple_xml_target(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.'))
}

fn xml_escape_attr(text: &str) -> String {
    let mut out = String::new();
    for ch in text.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            '\n' => out.push_str("&#10;"),
            '\r' => out.push_str("&#13;"),
            '\t' => out.push_str("&#9;"),
            _ => out.push(ch),
        }
    }
    out
}

pub fn new_call_id() -> String {
    static NEXT_ID: AtomicU64 = AtomicU64::new(1);
    let seq = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("call_{nanos:x}{seq:x}")
}

// ── Streaming assistant-output parser ──────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssistantEvent {
    Content(String),
    Reasoning(String),
    ToolCallBegin {
        index: usize,
        id: String,
        name: String,
    },
    /// A raw fragment of the arguments JSON for the call at `index`.
    ToolCallArgs {
        index: usize,
        fragment: String,
    },
    /// The call at `index` is complete; `arguments` is the full JSON string.
    ToolCallEnd {
        index: usize,
        arguments: String,
    },
    /// Raw-streaming mode only: the function name for the call at `index`
    /// became known (from the streamed header, or authoritatively at the end
    /// of the block). May repeat; the last one wins.
    ToolCallName {
        index: usize,
        name: String,
    },
    /// Raw-streaming mode only: the block at `index` failed to parse after a
    /// `ToolCallBegin` was already emitted. The caller must roll the call
    /// back; its text follows as ordinary `Content`/`Reasoning`.
    ToolCallDemoted {
        index: usize,
    },
}

/// Incremental parser over the visible generated text. Detects
/// `<think>`/`</think>` and Qwen3 XML tool-call blocks without ever leaking
/// partial tag text as content. The documented Qwen3 XML form allows
/// `<function=...>` to start a tool call even without an outer `<tool_call>`.
/// Feed deltas with [`feed`], then flush with [`finish`].
#[derive(Debug)]
pub struct AssistantStreamParser {
    state: ParserState,
    pending: String,
    trim_leading: bool,
    emitted_calls: usize,
    current_index: usize,
    /// Stream tool-call bodies incrementally: `ToolCallBegin` fires the moment
    /// `<tool_call>` is seen (empty name), raw XML body fragments flow as
    /// `ToolCallArgs`, the name arrives via `ToolCallName`, and the parsed
    /// arguments via `ToolCallEnd`. A block that fails to parse emits
    /// `ToolCallDemoted` before its text. Off = the legacy buffered behavior
    /// (Begin + one Args + End at block end).
    stream_raw: bool,
    /// Detect `<tool_call>` / rootless `<function=` blocks at all. Off when
    /// the request advertised no tools: the XML flows through as ordinary
    /// content text while `<think>` handling keeps working.
    parse_tool_calls: bool,
}

#[derive(Debug)]
enum ParserState {
    Text,
    Thinking,
    ToolCall {
        builder: ToolCallBuilder,
        origin: ToolCallOrigin,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolCallOrigin {
    Text,
    Thinking,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolCallBodyStatus {
    Ready,
    NeedMore,
    NotToolCall,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RootlessFunctionStatus {
    Ready(usize),
    NeedMore,
    NotToolCall,
}

const END_TOOL_CALL: &str = "</tool_call>";

impl AssistantStreamParser {
    /// `start_in_thinking` is set when the generation header already opened a
    /// `<think>` block in the prompt, so the model output begins inside it.
    pub fn new(start_in_thinking: bool) -> Self {
        Self::with_options(start_in_thinking, false, true)
    }

    /// `stream_raw` turns on incremental tool-call streaming (see the field
    /// doc); only the flagged chat-completions stream uses it.
    /// `parse_tool_calls` off disables tool-call detection entirely (see the
    /// field doc); think handling is unaffected.
    pub fn with_options(start_in_thinking: bool, stream_raw: bool, parse_tool_calls: bool) -> Self {
        Self {
            state: if start_in_thinking {
                ParserState::Thinking
            } else {
                ParserState::Text
            },
            pending: String::new(),
            trim_leading: true,
            emitted_calls: 0,
            current_index: 0,
            stream_raw,
            parse_tool_calls,
        }
    }

    pub fn feed(&mut self, delta: &str) -> Vec<AssistantEvent> {
        self.pending.push_str(delta);
        let mut events = Vec::new();
        loop {
            if self.pending.is_empty() {
                break;
            }
            let progressed = if matches!(self.state, ParserState::ToolCall { .. }) {
                self.drain_tool_call(&mut events)
            } else {
                self.drain_markup(&mut events)
            };
            if !progressed {
                break;
            }
        }
        events
    }

    /// Flushes held-back text and finalizes any open tool call. Unparseable
    /// blocks degrade to content so generated text is never lost.
    pub fn finish(&mut self) -> Vec<AssistantEvent> {
        let mut events = Vec::new();
        match std::mem::replace(&mut self.state, ParserState::Text) {
            ParserState::ToolCall {
                mut builder,
                origin,
            } => {
                let text_as_reasoning = origin == ToolCallOrigin::Thinking;
                if !self.pending.is_empty() {
                    let pending = std::mem::take(&mut self.pending);
                    let builder_events = builder.feed_scanned(&pending);
                    self.forward(builder_events, &mut events, text_as_reasoning);
                }
                let final_events = builder.finish_eof();
                self.forward(final_events, &mut events, text_as_reasoning);
            }
            state => {
                self.state = state;
                let pending = std::mem::take(&mut self.pending);
                if !pending.is_empty() {
                    self.emit_text(&pending, &mut events);
                }
            }
        }
        events
    }

    fn drain_markup(&mut self, events: &mut Vec<AssistantEvent>) -> bool {
        let Some(lt) = self.pending.find('<') else {
            let text = std::mem::take(&mut self.pending);
            self.emit_text(&text, events);
            return false;
        };
        if lt > 0 {
            let text: String = self.pending.drain(..lt).collect();
            self.emit_text(&text, events);
        }

        let in_thinking = matches!(self.state, ParserState::Thinking);
        if self.parse_tool_calls && starts_with_function_call(&self.pending) {
            match rootless_function_call_end(&self.pending) {
                RootlessFunctionStatus::Ready(end) => {
                    let raw: String = self.pending.drain(..end).collect();
                    self.emit_rootless_function_call(&raw, events, in_thinking);
                    self.trim_leading = true;
                    return true;
                }
                RootlessFunctionStatus::NeedMore => return false,
                RootlessFunctionStatus::NotToolCall => {}
            }
        }
        let candidates: &[&str] = match (self.parse_tool_calls, in_thinking) {
            (true, true) => &["<tool_call>", "</think>"],
            (true, false) => &["<tool_call>", "<think>", "</think>"],
            (false, true) => &["</think>"],
            (false, false) => &["<think>", "</think>"],
        };
        for tag in candidates {
            if self.pending.starts_with(tag) {
                if in_thinking && *tag == "<tool_call>" {
                    match tool_call_body_status(&self.pending[tag.len()..]) {
                        ToolCallBodyStatus::Ready => {}
                        ToolCallBodyStatus::NeedMore => return false,
                        ToolCallBodyStatus::NotToolCall => continue,
                    }
                }
                self.pending.drain(..tag.len());
                match *tag {
                    "<tool_call>" => {
                        let builder = ToolCallBuilder::new(self.stream_raw);
                        if self.stream_raw {
                            // Commit the call immediately (empty name) so the
                            // raw XML body can stream from the first token.
                            let begin = vec![builder.begin_event()];
                            self.forward(begin, events, in_thinking);
                        }
                        self.state = ParserState::ToolCall {
                            builder,
                            origin: if in_thinking {
                                ToolCallOrigin::Thinking
                            } else {
                                ToolCallOrigin::Text
                            },
                        };
                    }
                    "<think>" => {
                        self.state = ParserState::Thinking;
                        self.trim_leading = true;
                    }
                    // A stray `</think>` (the prompt header already closed the
                    // block) is swallowed, matching the engine's legacy filter.
                    "</think>" => {
                        self.state = ParserState::Text;
                        self.trim_leading = true;
                    }
                    _ => unreachable!(),
                }
                return true;
            }
        }
        let possible_starts: &[&str] = match (self.parse_tool_calls, in_thinking) {
            (true, true) => &["<tool_call>", "<function=", "<function ", "</think>"],
            (true, false) => &[
                "<tool_call>",
                "<function=",
                "<function ",
                "<think>",
                "</think>",
            ],
            (false, true) => &["</think>"],
            (false, false) => &["<think>", "</think>"],
        };
        if possible_starts
            .iter()
            .any(|tag| tag.starts_with(self.pending.as_str()))
        {
            return false; // possible partial tag; wait for more input
        }
        // A `<` that opens no recognized tag is ordinary text.
        let lt_text: String = self.pending.drain(..1).collect();
        self.emit_text(&lt_text, events);
        true
    }

    fn drain_tool_call(&mut self, events: &mut Vec<AssistantEvent>) -> bool {
        let ParserState::ToolCall { builder, origin } = &mut self.state else {
            return false;
        };
        let text_as_reasoning = *origin == ToolCallOrigin::Thinking;
        if let Some(idx) = self.pending.find(END_TOOL_CALL) {
            let before = self.pending[..idx].to_string();
            self.pending.drain(..idx + END_TOOL_CALL.len());
            let builder_events = builder.feed_scanned(&before);
            self.forward(builder_events, events, text_as_reasoning);
            let ParserState::ToolCall { builder, origin } =
                std::mem::replace(&mut self.state, ParserState::Text)
            else {
                unreachable!()
            };
            let final_events = builder.finalize();
            let demoted_to_text = final_events
                .iter()
                .all(|event| matches!(event, BuilderEvent::Text(_) | BuilderEvent::Demote(_)));
            let text_as_reasoning = origin == ToolCallOrigin::Thinking;
            self.forward(final_events, events, text_as_reasoning);
            if text_as_reasoning && demoted_to_text {
                self.state = ParserState::Thinking;
            }
            self.trim_leading = true;
            return true;
        }

        // Hold back a tail that could still grow into the end tag.
        let mut hold = 0;
        for len in (1..END_TOOL_CALL.len()).rev() {
            if self.pending.len() < len {
                continue;
            }
            let split = self.pending.len() - len;
            if !self.pending.is_char_boundary(split) {
                continue;
            }
            if END_TOOL_CALL.starts_with(&self.pending[split..]) {
                hold = len;
                break;
            }
        }
        let feed_to = self.pending.len() - hold;
        if feed_to == 0 {
            return false;
        }
        let before: String = self.pending.drain(..feed_to).collect();
        let builder_events = builder.feed_scanned(&before);
        self.forward(builder_events, events, text_as_reasoning);
        true
    }

    fn emit_text(&mut self, text: &str, events: &mut Vec<AssistantEvent>) {
        let mut text = text;
        if self.trim_leading {
            text = text.trim_start_matches(['\r', '\n']);
            if text.is_empty() {
                return;
            }
            self.trim_leading = false;
        }
        let event = if matches!(self.state, ParserState::Thinking) {
            AssistantEvent::Reasoning(text.to_string())
        } else {
            AssistantEvent::Content(text.to_string())
        };
        events.push(event);
    }

    fn emit_rootless_function_call(
        &mut self,
        raw: &str,
        events: &mut Vec<AssistantEvent>,
        text_as_reasoning: bool,
    ) {
        if let Some((name, arguments)) = parse_tool_call_xml(raw) {
            let id = new_call_id();
            self.forward(
                vec![
                    BuilderEvent::Begin { id, name },
                    BuilderEvent::Args(arguments.clone()),
                    BuilderEvent::End(arguments),
                ],
                events,
                text_as_reasoning,
            );
            return;
        }
        warn_unparsed_tool_call(raw);
        if text_as_reasoning {
            events.push(AssistantEvent::Reasoning(raw.to_string()));
        } else {
            events.push(AssistantEvent::Content(raw.to_string()));
        }
    }

    fn forward(
        &mut self,
        builder_events: Vec<BuilderEvent>,
        events: &mut Vec<AssistantEvent>,
        text_as_reasoning: bool,
    ) {
        for event in builder_events {
            match event {
                BuilderEvent::Begin { id, name } => {
                    self.current_index = self.emitted_calls;
                    self.emitted_calls += 1;
                    events.push(AssistantEvent::ToolCallBegin {
                        index: self.current_index,
                        id,
                        name,
                    });
                }
                BuilderEvent::Args(fragment) => events.push(AssistantEvent::ToolCallArgs {
                    index: self.current_index,
                    fragment,
                }),
                BuilderEvent::End(arguments) => events.push(AssistantEvent::ToolCallEnd {
                    index: self.current_index,
                    arguments,
                }),
                BuilderEvent::Text(text) => {
                    if text_as_reasoning {
                        events.push(AssistantEvent::Reasoning(text));
                    } else {
                        events.push(AssistantEvent::Content(text));
                    }
                }
                BuilderEvent::Name(name) => events.push(AssistantEvent::ToolCallName {
                    index: self.current_index,
                    name,
                }),
                BuilderEvent::Demote(text) => {
                    // The call was committed with an early Begin but its block
                    // failed to parse: roll the index back so the next call
                    // stays dense, then re-emit the block as ordinary text.
                    events.push(AssistantEvent::ToolCallDemoted {
                        index: self.current_index,
                    });
                    self.emitted_calls = self.emitted_calls.saturating_sub(1);
                    if text_as_reasoning {
                        events.push(AssistantEvent::Reasoning(text));
                    } else {
                        events.push(AssistantEvent::Content(text));
                    }
                }
            }
        }
    }
}

fn tool_call_body_status(tail: &str) -> ToolCallBodyStatus {
    let trimmed = tail.trim_start_matches(['\r', '\n', '\t', ' ']);
    if trimmed.is_empty() {
        ToolCallBodyStatus::NeedMore
    } else if !trimmed.starts_with('<') || trimmed.starts_with("</") {
        ToolCallBodyStatus::NotToolCall
    } else {
        compact_tool_call_body_status(trimmed)
    }
}

fn compact_tool_call_body_status(trimmed: &str) -> ToolCallBodyStatus {
    if is_recoverable_bash_command_body(trimmed) && trimmed.contains(END_TOOL_CALL) {
        return ToolCallBodyStatus::Ready;
    }
    let Some(end) = find_open_tag_end(trimmed) else {
        if is_recoverable_bash_command_body(trimmed) {
            return ToolCallBodyStatus::Ready;
        }
        return ToolCallBodyStatus::NeedMore;
    };
    let mut tag_body = trimmed[1..end].trim();
    let self_closing = tag_body.ends_with('/');
    if self_closing {
        tag_body = tag_body[..tag_body.len() - 1].trim_end();
    }
    if tag_body == "function"
        || tag_body.starts_with("function=")
        || tag_body.starts_with("function ")
    {
        return ToolCallBodyStatus::Ready;
    }
    let Some((name, _)) = split_xml_element_name(tag_body) else {
        return ToolCallBodyStatus::NotToolCall;
    };
    if name == "parameter" || !is_simple_xml_target(name) {
        return ToolCallBodyStatus::NotToolCall;
    }
    if self_closing {
        return ToolCallBodyStatus::Ready;
    }
    let close = format!("</{name}>");
    let after_open = trimmed[end + 1..].trim_start();
    if after_open.starts_with(&close) {
        ToolCallBodyStatus::Ready
    } else if after_open.is_empty() || close.starts_with(after_open) {
        ToolCallBodyStatus::NeedMore
    } else {
        ToolCallBodyStatus::NotToolCall
    }
}

fn starts_with_function_call(text: &str) -> bool {
    text.starts_with("<function=") || text.starts_with("<function ")
}

fn rootless_function_call_end(text: &str) -> RootlessFunctionStatus {
    if !starts_with_function_call(text) {
        return RootlessFunctionStatus::NotToolCall;
    }
    let Some(open_end) = find_open_tag_end(text) else {
        return RootlessFunctionStatus::NeedMore;
    };
    if parse_named_open_tag(text, "function").is_none() {
        return RootlessFunctionStatus::NotToolCall;
    }
    let Some(close) = find_close_tag(text, "function", open_end + 1) else {
        return RootlessFunctionStatus::NeedMore;
    };
    RootlessFunctionStatus::Ready(close.end)
}

// ── Tool-call block builder ────────────────────────────────────────────────

#[derive(Debug)]
enum BuilderEvent {
    Begin {
        id: String,
        name: String,
    },
    Args(String),
    End(String),
    /// Degraded output: the block was not a parseable tool call.
    Text(String),
    /// Raw-streaming mode: the function name became known.
    Name(String),
    /// Raw-streaming mode: degraded output after an early `Begin` was already
    /// emitted; the caller rolls the call back and re-emits this as text.
    Demote(String),
}

/// Parses the inside of one Qwen3 XML `<tool_call>...</tool_call>` block.
/// Model-emitted JSON inside `<tool_call>` is intentionally not accepted.
#[derive(Debug)]
struct ToolCallBuilder {
    id: String,
    raw: String,
    /// Incremental mode: emit the raw body as `Args` fragments as it arrives
    /// and the function name (`Name`) as soon as the header is recognizable.
    /// The whole-block parse at the end stays authoritative.
    stream_raw: bool,
    /// The header was classified (or given up on); stop rescanning `raw`.
    header_settled: bool,
}

/// Give up classifying the streamed header once the body has grown this far
/// without a complete first tag: every recognizable header (`<function=NAME>`,
/// `<bash …`, a compact `<name attr=…>`) fits well within it, and rescanning an
/// unbounded prefix on every feed would be quadratic on pathological bodies.
const HEADER_SCAN_LIMIT: usize = 256;

impl ToolCallBuilder {
    fn new(stream_raw: bool) -> Self {
        Self {
            id: new_call_id(),
            raw: String::new(),
            stream_raw,
            header_settled: false,
        }
    }

    /// The early commit for raw-streaming mode: the call exists (stable id,
    /// index assigned by `forward`) before its name is known.
    fn begin_event(&self) -> BuilderEvent {
        BuilderEvent::Begin {
            id: self.id.clone(),
            name: String::new(),
        }
    }

    /// Consumes text that the outer parser already verified contains no end
    /// tag. Always consumes the full input.
    fn feed_scanned(&mut self, text: &str) -> Vec<BuilderEvent> {
        self.raw.push_str(text);
        if !self.stream_raw {
            return Vec::new();
        }
        let mut events = Vec::new();
        if !self.header_settled {
            if let Some(name) = self.scan_header() {
                events.push(BuilderEvent::Name(name));
            }
        }
        if !text.is_empty() {
            events.push(BuilderEvent::Args(text.to_string()));
        }
        events
    }

    /// Best-effort early recognition of the function name from the streamed
    /// header: `<function=NAME>`, the recoverable `<bash …` attribute form, or
    /// a compact `<name attr=…>` element. Purely cosmetic for live display —
    /// the authoritative name is re-emitted by the whole-block parse.
    fn scan_header(&mut self) -> Option<String> {
        let trimmed = self.raw.trim_start();
        if trimmed.is_empty() {
            return None;
        }
        if !trimmed.starts_with('<') {
            // Whatever this is, it is not a recognizable header; the final
            // parse will settle it (likely a demote).
            self.header_settled = true;
            return None;
        }
        if is_recoverable_bash_command_body(trimmed) {
            self.header_settled = true;
            return Some("bash".to_string());
        }
        let Some(end) = find_open_tag_end(trimmed) else {
            if self.raw.len() > HEADER_SCAN_LIMIT {
                self.header_settled = true;
            }
            return None;
        };
        self.header_settled = true;
        if let Some((name, _)) = parse_named_open_tag(trimmed, "function") {
            return Some(name);
        }
        // Compact form `<name attr=…>`: mirror parse_compact_tool_call_xml's
        // element-name acceptance.
        let (name, _) = split_xml_element_name(trimmed[1..end].trim())?;
        if matches!(name, "function" | "parameter") || !is_simple_xml_target(name) {
            return None;
        }
        Some(name.to_string())
    }

    /// Called when the end tag arrives.
    fn finalize(self) -> Vec<BuilderEvent> {
        self.finish_parsed(true)
    }

    /// Called at end of stream with no end tag in sight.
    fn finish_eof(self) -> Vec<BuilderEvent> {
        self.finish_parsed(false)
    }

    fn finish_parsed(self, closed: bool) -> Vec<BuilderEvent> {
        let mut events = Vec::new();
        if let Some((name, arguments)) = parse_tool_call_xml(&self.raw) {
            if self.stream_raw {
                // Begin and the raw Args fragments already went out; deliver
                // the authoritative name (overwrites any streamed guess) and
                // the parsed arguments.
                events.push(BuilderEvent::Name(name));
                events.push(BuilderEvent::End(arguments));
            } else {
                events.push(BuilderEvent::Begin {
                    id: self.id.clone(),
                    name,
                });
                events.push(BuilderEvent::Args(arguments.clone()));
                events.push(BuilderEvent::End(arguments));
            }
        } else {
            warn_unparsed_tool_call(&self.raw);
            let text = if closed {
                format!("<tool_call>{}</tool_call>", self.raw)
            } else {
                format!("<tool_call>{}", self.raw)
            };
            if self.stream_raw {
                events.push(BuilderEvent::Demote(text));
            } else {
                events.push(BuilderEvent::Text(text));
            }
        }
        events
    }
}

/// XML tool-call text the model emitted could not be resolved into a call and
/// is demoted to plain text. Surface it on stderr so format drift is
/// diagnosable.
fn warn_unparsed_tool_call(raw: &str) {
    let trimmed = raw.trim();
    let preview: String = trimmed.chars().take(120).collect();
    let ellipsis = if preview.len() < trimmed.len() {
        "…"
    } else {
        ""
    };
    eprintln!(
        "qw35: model output warning: malformed XML tool call could not be parsed; treating it as assistant text: {preview}{ellipsis}"
    );
}

/// Whole-block XML parser. Returns the function name and OpenAI-style
/// arguments JSON string.
fn parse_tool_call_xml(raw: &str) -> Option<(String, String)> {
    let body = raw.trim();
    parse_compact_tool_call_xml(body)
        .or_else(|| parse_recoverable_bash_tool_call_xml(body))
        .or_else(|| parse_function_parameter_tool_call_xml(body))
}

fn parse_compact_tool_call_xml(body: &str) -> Option<(String, String)> {
    if !body.starts_with('<') || body.starts_with("</") {
        return None;
    }
    let end = find_open_tag_end(body)?;
    let mut tag_body = body[1..end].trim();
    let self_closing = tag_body.ends_with('/');
    if self_closing {
        tag_body = tag_body[..tag_body.len() - 1].trim_end();
    }
    let (name, attr_tail) = split_xml_element_name(tag_body)?;
    if matches!(name, "function" | "parameter") || !is_simple_xml_target(name) {
        return None;
    }
    let arguments = parse_xml_attributes(attr_tail)?;
    let after_open = &body[end + 1..];
    if self_closing {
        if !after_open.trim().is_empty() {
            return None;
        }
    } else {
        let close = format!("</{name}>");
        let after_open = after_open.trim();
        if after_open != close {
            return None;
        }
    }
    Some((
        name.to_string(),
        serde_json::Value::Object(arguments).to_string(),
    ))
}

fn parse_recoverable_bash_tool_call_xml(body: &str) -> Option<(String, String)> {
    if !is_recoverable_bash_command_body(body) {
        return None;
    }
    let after_name = body.trim_start().strip_prefix("<bash")?;
    let after_name = after_name.trim_start();
    if after_name.starts_with('>') || after_name.starts_with("/>") {
        return None;
    }
    let command_pos = after_name.find("command")?;
    let before_command = &after_name[..command_pos];
    if before_command
        .chars()
        .any(|ch| !ch.is_whitespace() && ch != '/')
    {
        return None;
    }
    let after_command = after_name[command_pos + "command".len()..].trim_start();
    let raw_value = after_command.strip_prefix('=')?.trim_start();
    let command = recover_bash_command_attribute(raw_value)?;
    let mut args = serde_json::Map::new();
    args.insert(
        "command".to_string(),
        serde_json::Value::String(decode_recovered_bash_command(command)),
    );
    Some((
        "bash".to_string(),
        serde_json::Value::Object(args).to_string(),
    ))
}

fn is_recoverable_bash_command_body(body: &str) -> bool {
    let trimmed = body.trim_start();
    let Some(after_name) = trimmed.strip_prefix("<bash") else {
        return false;
    };
    matches!(
        after_name.chars().next(),
        Some(ch) if ch.is_whitespace() || ch == '>' || ch == '/'
    )
}

fn recover_bash_command_attribute(raw_value: &str) -> Option<String> {
    if raw_value.is_empty() {
        return None;
    }
    let mut chars = raw_value.chars();
    let first = chars.next()?;
    if first == '"' || first == '\'' {
        let value = &raw_value[first.len_utf8()..];
        return Some(trim_recovered_bash_command_tail(value, Some(first)));
    }
    Some(trim_recovered_bash_command_tail(raw_value, None))
}

fn trim_recovered_bash_command_tail(value: &str, quote: Option<char>) -> String {
    let mut text = value.trim_end();
    if let Some(rest) = text.strip_suffix("</bash>") {
        text = rest.trim_end();
    }
    if let Some(rest) = text.strip_suffix("/>") {
        text = rest.trim_end();
    }
    if let Some(quote) = quote {
        if let Some(rest) = text.strip_suffix('>') {
            let rest = rest.trim_end();
            if rest.ends_with(quote) {
                text = rest;
            }
        }
    }
    if let Some(quote) = quote {
        if let Some(rest) = text.strip_suffix(quote) {
            text = rest.trim_end();
        }
    }
    xml_unescape_text(text)
}

fn decode_recovered_bash_command(command: String) -> String {
    decode_recovered_escaped_controls(command)
}

fn decode_recovered_escaped_controls(text: String) -> String {
    if !text.contains("\\n") && !text.contains("\\r") && !text.contains("\\t") {
        return text;
    }
    let mut out = String::new();
    let mut chars = text.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch != '\\' {
            out.push(ch);
            continue;
        }
        match chars.peek().copied() {
            Some('n') => {
                chars.next();
                out.push('\n');
            }
            Some('r') => {
                chars.next();
                out.push('\r');
            }
            Some('t') => {
                chars.next();
                out.push('\t');
            }
            _ => out.push(ch),
        }
    }
    out
}

fn parse_function_parameter_tool_call_xml(body: &str) -> Option<(String, String)> {
    let (name, function_body_start) = parse_named_open_tag(body, "function")?;
    // Match the *last* `</function>`: a verbatim parameter value may contain the
    // literal substring `</function>`, so the first match can fall inside it.
    let function_close = find_last_close_tag(body, "function", function_body_start)?;
    if !body[function_close.end..].trim().is_empty() {
        return None;
    }
    let function_body = &body[function_body_start..function_close.start];
    let arguments = parse_xml_parameters(function_body)?;
    Some((name, serde_json::Value::Object(arguments).to_string()))
}

fn split_xml_element_name(tag_body: &str) -> Option<(&str, &str)> {
    let tag_body = tag_body.trim();
    if tag_body.is_empty() {
        return None;
    }
    let name_end = tag_body
        .find(|ch: char| ch.is_whitespace())
        .unwrap_or(tag_body.len());
    let name = &tag_body[..name_end];
    let tail = &tag_body[name_end..];
    if name.is_empty() {
        None
    } else {
        Some((name, tail))
    }
}

fn parse_xml_attributes(mut tail: &str) -> Option<serde_json::Map<String, serde_json::Value>> {
    let mut args = serde_json::Map::new();
    loop {
        tail = tail.trim_start();
        if tail.is_empty() {
            break;
        }
        let key_end = tail
            .find(|ch: char| ch.is_whitespace() || ch == '=')
            .unwrap_or(tail.len());
        if key_end == 0 {
            return None;
        }
        let key = &tail[..key_end];
        if !is_simple_xml_target(key) {
            return None;
        }
        tail = tail[key_end..].trim_start();
        let rest = tail.strip_prefix('=')?;
        let (value, consumed) = parse_tag_value(rest)?;
        args.insert(key.to_string(), json_or_string_value(value));
        tail = &rest[consumed..];
    }
    Some(args)
}

#[derive(Debug, Clone, Copy)]
struct CloseTag {
    start: usize,
    end: usize,
}

fn find_close_tag(s: &str, tag: &str, from: usize) -> Option<CloseTag> {
    let close = format!("</{tag}>");
    let relative = s[from..].find(&close)?;
    let start = from + relative;
    Some(CloseTag {
        start,
        end: start + close.len(),
    })
}

/// Like [`find_close_tag`] but returns the *last* occurrence. Parameter values
/// are emitted verbatim (no entity escaping — see `render_xml_parameter_block`),
/// so a bash command can legitimately contain the literal substring
/// `</function>`. The real function close is always the final one, since the
/// caller requires nothing but whitespace after it.
fn find_last_close_tag(s: &str, tag: &str, from: usize) -> Option<CloseTag> {
    let close = format!("</{tag}>");
    let relative = s[from..].rfind(&close)?;
    let start = from + relative;
    Some(CloseTag {
        start,
        end: start + close.len(),
    })
}

/// Finds the `</parameter>` that actually closes a parameter value, skipping any
/// `</parameter>` the (verbatim) value contains. The closing tag is the first
/// one followed by either end-of-body or the next `<parameter` open tag; a
/// `</parameter>` sitting in the middle of value text is followed by more value
/// and is treated as literal content.
fn find_parameter_value_close(body: &str, from: usize) -> Option<CloseTag> {
    let needle = "</parameter>";
    let mut search = from;
    loop {
        let relative = body[search..].find(needle)?;
        let start = search + relative;
        let end = start + needle.len();
        let rest = body[end..].trim_start();
        if rest.is_empty() || rest.starts_with("<parameter") {
            return Some(CloseTag { start, end });
        }
        search = end;
    }
}

fn parse_xml_parameters(body: &str) -> Option<serde_json::Map<String, serde_json::Value>> {
    let mut args = serde_json::Map::new();
    let mut pos = 0usize;
    loop {
        pos = skip_xml_ws(body, pos);
        if pos >= body.len() {
            break;
        }
        let (name, value_start) = parse_named_open_tag(&body[pos..], "parameter")?;
        let value_start = pos + value_start;
        let close = find_parameter_value_close(body, value_start)?;
        let raw_value = &body[value_start..close.start];
        args.insert(name, parameter_value(raw_value));
        pos = close.end;
    }
    Some(args)
}

fn parameter_value(raw: &str) -> serde_json::Value {
    let normalized = strip_parameter_boundary_newlines(raw);
    let text = xml_unescape_text(normalized);
    json_or_string_value(text)
}

fn json_or_string_value(text: String) -> serde_json::Value {
    let trimmed = text.trim();
    if trimmed.starts_with('{') || trimmed.starts_with('[') {
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed) {
            return value;
        }
    }
    serde_json::Value::String(text)
}

fn strip_parameter_boundary_newlines(mut value: &str) -> &str {
    if let Some(rest) = value.strip_prefix("\r\n") {
        value = rest;
    } else if let Some(rest) = value.strip_prefix('\n') {
        value = rest;
    }
    if let Some(rest) = value.strip_suffix("\r\n") {
        value = rest;
    } else if let Some(rest) = value.strip_suffix('\n') {
        value = rest;
    }
    value
}

fn parse_named_open_tag(s: &str, tag: &str) -> Option<(String, usize)> {
    let prefix = format!("<{tag}");
    if !s.starts_with(&prefix) {
        return None;
    }
    let after_name = prefix.len();
    if let Some(ch) = s[after_name..].chars().next() {
        if !ch.is_whitespace() && ch != '=' && ch != '>' {
            return None;
        }
    }
    let end = find_open_tag_end(s)?;
    let tail = &s[after_name..end];
    let name = parse_tag_target(tail)?;
    if name.is_empty() {
        return None;
    }
    Some((name, end + 1))
}

fn find_open_tag_end(s: &str) -> Option<usize> {
    let mut quote: Option<char> = None;
    for (idx, ch) in s.char_indices() {
        if idx == 0 {
            continue;
        }
        match quote {
            Some(mark) if ch == mark => quote = None,
            Some(_) => {}
            None if ch == '"' || ch == '\'' => quote = Some(ch),
            None if ch == '>' => return Some(idx),
            None => {}
        }
    }
    None
}

fn parse_tag_target(tail: &str) -> Option<String> {
    let tail = tail.trim();
    if let Some(rest) = tail.strip_prefix('=') {
        return parse_tag_value(rest).map(|(value, _)| value);
    }
    parse_name_attribute(tail)
}

fn parse_name_attribute(mut tail: &str) -> Option<String> {
    while !tail.trim_start().is_empty() {
        tail = tail.trim_start();
        let key_end = tail
            .find(|ch: char| ch.is_whitespace() || ch == '=')
            .unwrap_or(tail.len());
        let key = &tail[..key_end];
        tail = tail[key_end..].trim_start();
        let rest = tail.strip_prefix('=')?;
        let (value, consumed) = parse_tag_value(rest)?;
        if key == "name" {
            return Some(value);
        }
        tail = &rest[consumed..];
    }
    None
}

fn parse_tag_value(raw: &str) -> Option<(String, usize)> {
    let leading = raw.len() - raw.trim_start().len();
    let raw = raw.trim_start();
    let first = raw.chars().next()?;
    if first == '"' || first == '\'' {
        let mut escaped = false;
        for (idx, ch) in raw.char_indices().skip(1) {
            if escaped {
                escaped = false;
                continue;
            }
            if ch == '\\' {
                escaped = true;
                continue;
            }
            if ch == first {
                let value = xml_unescape_text(&raw[1..idx]);
                return Some((value, leading + idx + ch.len_utf8()));
            }
        }
        return None;
    }
    let end = raw
        .find(|ch: char| ch.is_whitespace() || ch == '>')
        .unwrap_or(raw.len());
    let value = raw[..end].trim();
    if value.is_empty() {
        None
    } else {
        Some((xml_unescape_text(value), leading + end))
    }
}

fn skip_xml_ws(s: &str, mut pos: usize) -> usize {
    while pos < s.len() {
        let Some(ch) = s[pos..].chars().next() else {
            break;
        };
        if !ch.is_whitespace() {
            break;
        }
        pos += ch.len_utf8();
    }
    pos
}

fn xml_unescape_text(text: &str) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some(idx) = rest.find('&') {
        out.push_str(&rest[..idx]);
        rest = &rest[idx..];
        let replacement = if rest.starts_with("&lt;") {
            Some(("<", 4))
        } else if rest.starts_with("&gt;") {
            Some((">", 4))
        } else if rest.starts_with("&amp;") {
            Some(("&", 5))
        } else if rest.starts_with("&quot;") {
            Some(("\"", 6))
        } else if rest.starts_with("&apos;") {
            Some(("'", 6))
        } else if rest.starts_with("&#10;") {
            Some(("\n", 5))
        } else if rest.starts_with("&#13;") {
            Some(("\r", 5))
        } else if rest.starts_with("&#9;") {
            Some(("\t", 4))
        } else if rest.starts_with("&#xA;") || rest.starts_with("&#xa;") {
            Some(("\n", 5))
        } else if rest.starts_with("&#xD;") || rest.starts_with("&#xd;") {
            Some(("\r", 5))
        } else if rest.starts_with("&#x9;") {
            Some(("\t", 5))
        } else {
            None
        };
        if let Some((value, consumed)) = replacement {
            out.push_str(value);
            rest = &rest[consumed..];
        } else {
            out.push('&');
            rest = &rest[1..];
        }
    }
    out.push_str(rest);
    out
}

// ── Whole-text convenience ─────────────────────────────────────────────────

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ParsedToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ParsedAssistantOutput {
    pub content: String,
    pub reasoning: String,
    pub tool_calls: Vec<ParsedToolCall>,
}

/// Runs the parser over a complete generation (the non-streaming path).
pub fn parse_assistant_text(
    text: &str,
    start_in_thinking: bool,
    parse_tool_calls: bool,
) -> ParsedAssistantOutput {
    let mut parser = AssistantStreamParser::with_options(start_in_thinking, false, parse_tool_calls);
    let mut events = parser.feed(text);
    events.extend(parser.finish());
    assemble_events(&events)
}

/// Folds a stream of events into the final message shape.
pub fn assemble_events(events: &[AssistantEvent]) -> ParsedAssistantOutput {
    let mut out = ParsedAssistantOutput::default();
    for event in events {
        match event {
            AssistantEvent::Content(text) => out.content.push_str(text),
            AssistantEvent::Reasoning(text) => out.reasoning.push_str(text),
            AssistantEvent::ToolCallBegin { index, id, name } => {
                if out.tool_calls.len() == *index {
                    out.tool_calls.push(ParsedToolCall {
                        id: id.clone(),
                        name: name.clone(),
                        arguments: String::new(),
                    });
                }
            }
            AssistantEvent::ToolCallArgs { .. } => {}
            AssistantEvent::ToolCallEnd { index, arguments } => {
                if let Some(call) = out.tool_calls.get_mut(*index) {
                    call.arguments = arguments.clone();
                }
            }
            AssistantEvent::ToolCallName { index, name } => {
                if let Some(call) = out.tool_calls.get_mut(*index) {
                    call.name = name.clone();
                }
            }
            // A demoted call is rolled back; its text follows as Content or
            // Reasoning and is folded in by the arms above.
            AssistantEvent::ToolCallDemoted { index } => {
                out.tool_calls.truncate(*index);
            }
        }
    }
    out
}

#[cfg(test)]
#[path = "tests/toolcall.rs"]
mod tests;
