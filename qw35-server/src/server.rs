use crate::model::{ChatTurn, Engine, FinishReason, GenerateError, GenerateRequest, TokenLimit};
use crate::toolcall::{self, AssistantEvent, ParsedAssistantOutput};
use axum::extract::{Path, State};
use axum::http::header;
use axum::http::{Method, StatusCode};
use axum::{
    response::{sse::Event, IntoResponse, Json, Response, Sse},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;

// ── Request types ──────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct ChatRequest {
    #[serde(default)]
    model: Option<String>,
    messages: Vec<ChatMessage>,
    #[serde(alias = "max_completion_tokens", alias = "num_predict")]
    max_tokens: Option<i64>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    min_p: Option<f32>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    repetition_penalty: Option<f32>,
    repeat_last_n: Option<i32>,
    enable_thinking: Option<bool>,
    preserve_thinking: Option<bool>,
    /// OpenAI-style thinking control, sent by clients (pi, openai SDKs) that
    /// have no `enable_thinking` concept. "none"/"minimal" leave thinking off.
    #[serde(default)]
    reasoning_effort: Option<String>,
    /// vLLM-style Qwen template kwargs; pi's `qwen-chat-template` format.
    #[serde(default)]
    chat_template_kwargs: Option<ChatTemplateKwargs>,
    ignore_eos: Option<bool>,
    /// Per-request override of the thinking-budget wrap-up message (empty string
    /// forces a bare `</think>`). Falls back to the server default.
    #[serde(default)]
    reasoning_budget_message: Option<String>,
    #[serde(default)]
    stop: Option<StopSpec>,
    #[serde(default)]
    n: Option<u32>,
    #[serde(default)]
    tools: Option<serde_json::Value>,
    #[serde(default)]
    tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    response_format: Option<serde_json::Value>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    stream_options: Option<StreamOptions>,
    /// qw35 extension (sent by qwowl35): stream tool-call bodies incrementally
    /// as raw XML `arguments` fragments plus `qw35_tool_call` side-channel
    /// chunks, instead of buffering each call to one delta at its end.
    #[serde(default)]
    stream_tool_call_xml: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct ChatTemplateKwargs {
    #[serde(default)]
    enable_thinking: Option<bool>,
    #[serde(default)]
    preserve_thinking: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum StopSpec {
    One(String),
    Many(Vec<String>),
}

impl StopSpec {
    fn to_vec(&self) -> Vec<String> {
        match self {
            StopSpec::One(stop) => vec![stop.clone()],
            StopSpec::Many(stops) => stops.clone(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    role: String,
    #[serde(default)]
    content: Option<MessageContent>,
    #[serde(default)]
    tool_calls: Option<Vec<ChatToolCall>>,
}

#[derive(Debug, Deserialize)]
struct ChatToolCall {
    function: ChatToolCallFunction,
}

#[derive(Debug, Deserialize)]
struct ChatToolCallFunction {
    name: String,
    #[serde(default)]
    arguments: Option<serde_json::Value>,
}

impl ChatToolCallFunction {
    /// OpenAI sends `arguments` as a JSON string, but some clients replay it
    /// as a decoded object; accept both.
    fn arguments_text(&self) -> String {
        match &self.arguments {
            None => String::new(),
            Some(serde_json::Value::String(text)) => text.clone(),
            Some(other) => other.to_string(),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Deserialize)]
struct ContentPart {
    #[serde(default, rename = "type")]
    part_type: Option<String>,
    #[serde(default)]
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct StreamOptions {
    #[serde(default)]
    include_usage: bool,
}

impl MessageContent {
    fn join(&self) -> Result<String, String> {
        match self {
            MessageContent::Text(text) => Ok(text.clone()),
            MessageContent::Parts(parts) => {
                let mut out = String::new();
                for part in parts {
                    let part_type = part.part_type.as_deref().unwrap_or("text");
                    if !is_text_part_type(part_type) {
                        return Err(format!(
                            "unsupported chat content part type {part_type:?}; qw35 accepts text-only requests"
                        ));
                    }
                    let text = part.text.as_deref().ok_or_else(|| {
                        format!("chat content part type {part_type:?} is missing text")
                    })?;
                    if !out.is_empty() {
                        out.push('\n');
                    }
                    out.push_str(text);
                }
                Ok(out)
            }
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct ResponsesRequest {
    #[serde(default)]
    model: Option<String>,
    input: ResponsesInput,
    #[serde(default)]
    instructions: Option<String>,
    #[serde(alias = "max_tokens", alias = "max_completion_tokens")]
    max_output_tokens: Option<i64>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    min_p: Option<f32>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    repetition_penalty: Option<f32>,
    repeat_last_n: Option<i32>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    previous_response_id: Option<String>,
    #[serde(default)]
    conversation: Option<serde_json::Value>,
    #[serde(default)]
    reasoning: Option<ResponsesReasoning>,
    #[serde(default)]
    reasoning_budget_message: Option<String>,
    #[serde(default)]
    tools: Option<serde_json::Value>,
    #[serde(default)]
    tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    text: Option<serde_json::Value>,
    #[serde(default)]
    store: Option<bool>,
    #[serde(default)]
    metadata: Option<serde_json::Value>,
    #[serde(default)]
    parallel_tool_calls: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum ResponsesInput {
    Text(String),
    Items(Vec<ResponsesInputItem>),
}

#[derive(Debug, Clone, Deserialize)]
struct ResponsesInputItem {
    #[serde(default, rename = "type")]
    item_type: Option<String>,
    #[serde(default)]
    role: Option<String>,
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    content: Option<serde_json::Value>,
    #[serde(default)]
    output: Option<serde_json::Value>,
    #[serde(default)]
    result: Option<serde_json::Value>,
    #[serde(default)]
    summary: Option<serde_json::Value>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    namespace: Option<String>,
    #[serde(default)]
    arguments: Option<serde_json::Value>,
    #[serde(default)]
    input: Option<serde_json::Value>,
    #[serde(default)]
    action: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
struct ResponsesReasoning {
    #[serde(default)]
    effort: Option<String>,
    #[serde(default)]
    summary: Option<serde_json::Value>,
}

#[derive(Debug, Clone)]
pub struct GenerationDefaults {
    pub max_tokens: TokenLimit,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub min_p: f32,
    pub presence_penalty: f32,
    pub repetition_penalty: f32,
    pub repeat_last_n: i32,
    /// Whether thinking is on by default when the request gives no thinking
    /// signal (no top-level `enable_thinking`, no `chat_template_kwargs`, no
    /// `reasoning_effort`). Seeded by the `--mode` preset; `false` for the
    /// agentic-coding default so param-less agent clients are unchanged.
    pub enable_thinking: bool,
    /// Wrap-up message forced before `</think>` when the thinking budget is
    /// exhausted (mirrors llama.cpp's `--reasoning-budget-message`). A per-request
    /// `reasoning_budget_message` overrides it; `None` forces a bare `</think>`.
    pub reasoning_budget_message: Option<String>,
}

/// Default wrap-up message forced before `</think>` at the thinking budget. A
/// concise first-person handoff so the model conditions on "answer now" rather
/// than being cut off mid-thought. Must not itself contain `</think>`.
pub const DEFAULT_REASONING_BUDGET_MESSAGE: &str = "\n\nI'll stop reasoning here and answer now.\n";

/// Official Qwen3.5 launch presets. Each seeds the full sampling profile and
/// the think/no-think principal config; per-request params still override
/// individual fields via the `unwrap_or(defaults.x)` chain. Selected with the
/// server `--mode <name>` flag; when absent, the `thinking-coding` preset is
/// used instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    ThinkingGeneral,
    ThinkingCoding,
    InstructGeneral,
    InstructReasoning,
}

impl Mode {
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "thinking-general" => Some(Self::ThinkingGeneral),
            "thinking-coding" => Some(Self::ThinkingCoding),
            "instruct-general" => Some(Self::InstructGeneral),
            "instruct-reasoning" => Some(Self::InstructReasoning),
            _ => None,
        }
    }

    /// The full default profile for this preset (official Qwen3.5 values).
    pub fn defaults(self) -> GenerationDefaults {
        // top_k=20, min_p=0 are common to all four; temperature, top_p,
        // presence_penalty, repetition_penalty and the thinking switch vary per
        // preset. A windowed repetition_penalty (>1.0) is the measured
        // loop-breaker — the flat presence penalty alone cannot break an
        // established loop — so the thinking presets keep it on; instruct
        // presets follow the official Qwen3.5 sampling profile (rep=1.0).
        let profile =
            |temperature, top_p, presence_penalty, repetition_penalty, enable_thinking| {
                GenerationDefaults {
                    max_tokens: TokenLimit::Context,
                    temperature,
                    top_p,
                    top_k: 20,
                    min_p: 0.0,
                    presence_penalty,
                    repetition_penalty,
                    repeat_last_n: 64,
                    enable_thinking,
                    reasoning_budget_message: Some(DEFAULT_REASONING_BUDGET_MESSAGE.to_string()),
                }
            };
        match self {
            //                       temp  top_p  presence  rep  think
            Self::ThinkingGeneral => profile(1.0, 0.95, 1.5, 1.1, true),
            Self::ThinkingCoding => profile(0.6, 0.95, 0.0, 1.1, true),
            Self::InstructGeneral => profile(0.7, 0.80, 1.5, 1.0, false),
            Self::InstructReasoning => profile(1.0, 0.95, 1.5, 1.0, false),
        }
    }
}

impl Default for GenerationDefaults {
    fn default() -> Self {
        // Agentic-coding defaults: agent clients (pi, codex, opencode) never
        // send penalty params and often omit temperature, and the 9B falls
        // into verbatim loops without a backstop. Low temperature keeps code
        // structure rigid at depth; a mild presence penalty (output-only,
        // OpenAI semantics) discourages echoing earlier chat content. The
        // presence penalty alone is flat and cannot break an established
        // loop (measured: 2/3 number-sequence runs still looped), so a mild
        // repetition penalty stays on as the loop-breaker — windowed to the
        // last 64 tokens precisely so recurring code syntax (brackets,
        // keywords, identifiers) outside the window is never penalized.
        Self {
            max_tokens: TokenLimit::Context,
            temperature: 0.8,
            top_p: 0.95,
            top_k: 20,
            min_p: 0.0,
            presence_penalty: 0.3,
            repetition_penalty: 1.1,
            repeat_last_n: 64,
            // Off unless the request asks for thinking — preserves the prior
            // behavior for headless agent clients that send no thinking signal.
            enable_thinking: false,
            reasoning_budget_message: Some(DEFAULT_REASONING_BUDGET_MESSAGE.to_string()),
        }
    }
}

impl GenerationDefaults {
    pub fn validate(&self) -> Result<(), String> {
        if matches!(self.max_tokens, TokenLimit::Fixed(0)) {
            return Err("--tokens must be greater than zero; use --num-predict -1 for context-limited generation".to_string());
        }
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err("--temperature must be a finite non-negative number".to_string());
        }
        if !self.top_p.is_finite() || self.top_p < 0.0 {
            return Err("--top-p must be a finite non-negative number".to_string());
        }
        if !self.min_p.is_finite() || self.min_p < 0.0 {
            return Err("--min-p must be a finite non-negative number".to_string());
        }
        if !self.presence_penalty.is_finite() {
            return Err("--presence-penalty must be finite".to_string());
        }
        if !self.repetition_penalty.is_finite() || self.repetition_penalty <= 0.0 {
            return Err("--repeat-penalty must be finite and greater than zero".to_string());
        }
        if self.repeat_last_n < -1 {
            return Err("--repeat-last-n must be -1 or greater".to_string());
        }
        Ok(())
    }
}

fn into_generate_request(
    req: &ChatRequest,
    default_model: &str,
    defaults: &GenerationDefaults,
) -> Result<GenerateRequest, String> {
    if let Some(n) = req.n {
        if n != 1 {
            return Err("n must be 1; multiple choices are not supported".to_string());
        }
    }
    let model = req.model.as_deref().unwrap_or(default_model).to_string();
    let mut messages = Vec::with_capacity(req.messages.len());
    for msg in &req.messages {
        let mut content = match &msg.content {
            Some(content) => content.join()?,
            None => String::new(),
        };
        if msg.role == "assistant" {
            if let Some(tool_calls) = &msg.tool_calls {
                for call in tool_calls {
                    if !content.is_empty() {
                        content.push('\n');
                    }
                    content.push_str(&toolcall::render_tool_call_block(
                        &call.function.name,
                        &call.function.arguments_text(),
                    ));
                }
            }
        }
        messages.push(ChatTurn {
            role: msg.role.clone(),
            content,
        });
    }

    let tool_defs = match &req.tools {
        Some(tools) => toolcall::parse_tool_defs(tools)?,
        None => Vec::new(),
    };
    let tool_choice = toolcall::parse_tool_choice(req.tool_choice.as_ref());
    if let Some(block) = toolcall::render_tools_system_block(&tool_defs, &tool_choice) {
        prepend_system_block(&mut messages, &block);
    }
    if let Some(format) = &req.response_format {
        match format
            .get("type")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("text")
        {
            "text" => {}
            "json_object" => append_system_block(
                &mut messages,
                "Respond with a single valid JSON object and nothing else.",
            ),
            "json_schema" => {
                return Err(
                    "response_format.type \"json_schema\" is not supported yet; use \"json_object\" or plain text"
                        .to_string(),
                )
            }
            other => return Err(format!("unsupported response_format type {other:?}")),
        }
    }

    // Precedence: explicit enable_thinking, then chat_template_kwargs, then
    // OpenAI reasoning_effort (same level set the responses endpoint accepts).
    let template_kwargs = req.chat_template_kwargs.as_ref();
    let enable_thinking = req
        .enable_thinking
        .or_else(|| template_kwargs.and_then(|kwargs| kwargs.enable_thinking))
        .or_else(|| {
            // An explicit reasoning_effort implies thinking-on.
            matches!(
                req.reasoning_effort.as_deref(),
                Some("low" | "medium" | "high" | "xhigh")
            )
            .then_some(true)
        })
        // No thinking signal at all → fall back to the server preset default.
        .unwrap_or(defaults.enable_thinking);
    let preserve_thinking = req
        .preserve_thinking
        .or_else(|| template_kwargs.and_then(|kwargs| kwargs.preserve_thinking))
        .unwrap_or(false);
    let max_tokens = match req.max_tokens {
        Some(value) => parse_token_limit("max_tokens", value)?,
        None => defaults.max_tokens,
    };
    let thinking_budget =
        thinking_budget_for(req.reasoning_effort.as_deref(), enable_thinking, max_tokens);
    Ok(GenerateRequest {
        model,
        messages,
        max_tokens,
        temperature: req.temperature.unwrap_or(defaults.temperature),
        top_p: req.top_p.unwrap_or(defaults.top_p),
        top_k: req.top_k.unwrap_or(defaults.top_k),
        min_p: req.min_p.unwrap_or(defaults.min_p),
        presence_penalty: req.presence_penalty.unwrap_or(defaults.presence_penalty),
        frequency_penalty: req.frequency_penalty.unwrap_or(0.0),
        repetition_penalty: req
            .repetition_penalty
            .unwrap_or(defaults.repetition_penalty),
        repeat_last_n: req.repeat_last_n.unwrap_or(defaults.repeat_last_n),
        enable_thinking,
        preserve_thinking,
        thinking_budget,
        reasoning_budget_message: req
            .reasoning_budget_message
            .clone()
            .or_else(|| defaults.reasoning_budget_message.clone()),
        ignore_eos: req.ignore_eos.unwrap_or(false),
        stop_sequences: req.stop.as_ref().map(StopSpec::to_vec).unwrap_or_default(),
        // With thinking enabled the markers pass through so the response can
        // carry reasoning_content; tool-call markers pass through always.
        emit_reasoning: enable_thinking,
        stream_tool_call_xml: req.stream_tool_call_xml.unwrap_or(false),
    })
}

/// Maps a `reasoning_effort` level to a thinking-token budget: the number of
/// tokens the model may spend inside `<think>` before the decoder forces
/// `</think>`. The fractions are tuned for this model, which reasons concisely —
/// generous OpenAI-style fractions let it loop for thousands of tokens before
/// the cap bites (low ≈ 4%, medium ≈ 10%, high ≈ 16% of the answer budget).
///
/// Returns `None` (uncapped) only when thinking is off. Whenever thinking is on
/// the decoder always gets a finite cap — `xhigh`/unspecified fall back to the
/// 0.16 fraction — so a model that degenerates into a reasoning repetition loop
/// is always force-closed eventually rather than running to the full context
/// length and never emitting `</think>`. (`none`/`minimal` mean thinking-off and
/// are handled upstream via `enable_thinking`.)
fn thinking_budget_for(
    effort: Option<&str>,
    enable_thinking: bool,
    max_tokens: TokenLimit,
) -> Option<u32> {
    if !enable_thinking {
        return None;
    }
    let fraction = match effort.map(str::to_ascii_lowercase).as_deref() {
        Some("low") => 0.04_f32,
        Some("medium") => 0.10,
        Some("high") => 0.16,
        // "xhigh", unknown, or unspecified → backstop cap (not uncapped).
        _ => 0.16,
    };
    // Scale against the caller's max_tokens when fixed; otherwise scale against
    // the agentic client default (8192) so context-limited requests still get a
    // proportional cap rather than running unbounded.
    let basis = match max_tokens {
        TokenLimit::Fixed(n) => n as f32,
        TokenLimit::Context => 8192.0,
    };
    Some(((basis * fraction) as u32).max(16))
}

/// Appends an instruction block to the leading system turn, creating one if
/// the conversation has none. Keeping the block inside the first system turn
/// leaves it in the stable prompt prefix, so the session cache still matches
/// across agent turns that resend identical tools.
fn append_system_block(messages: &mut Vec<ChatTurn>, block: &str) {
    if let Some(first) = messages.first_mut() {
        if first.role == "system" || first.role == "developer" {
            if !first.content.is_empty() {
                first.content.push_str("\n\n");
            }
            first.content.push_str(block);
            return;
        }
    }
    messages.insert(
        0,
        ChatTurn {
            role: "system".to_string(),
            content: block.to_string(),
        },
    );
}

/// Prepends a block to the system message, so it leads any user-provided system
/// content. The tools block must come first to match the chat_template, which
/// renders `# Tools ... </IMPORTANT>` and then appends the user system message
/// after a blank line.
fn prepend_system_block(messages: &mut Vec<ChatTurn>, block: &str) {
    if let Some(first) = messages.first_mut() {
        if first.role == "system" || first.role == "developer" {
            if first.content.is_empty() {
                first.content = block.to_string();
            } else {
                first.content = format!("{block}\n\n{}", first.content);
            }
            return;
        }
    }
    messages.insert(
        0,
        ChatTurn {
            role: "system".to_string(),
            content: block.to_string(),
        },
    );
}

fn into_responses_generate_request(
    req: &ResponsesRequest,
    default_model: &str,
    defaults: &GenerationDefaults,
) -> Result<GenerateRequest, String> {
    if req.previous_response_id.is_some() {
        return Err("previous_response_id is not supported; replay full input instead".to_string());
    }
    if req.conversation.is_some() {
        return Err("conversation is not supported; replay full input instead".to_string());
    }

    let model = req.model.as_deref().unwrap_or(default_model).to_string();
    let messages = responses_input_to_chat_turns(req)?;
    if messages.is_empty() {
        return Err("input must contain at least one message".to_string());
    }

    let max_tokens = match req.max_output_tokens {
        Some(value) => parse_token_limit("max_output_tokens", value)?,
        None => defaults.max_tokens,
    };
    let enable_thinking = responses_enable_thinking(req, defaults);
    let thinking_budget = thinking_budget_for(
        req.reasoning
            .as_ref()
            .and_then(|reasoning| reasoning.effort.as_deref()),
        enable_thinking,
        max_tokens,
    );
    Ok(GenerateRequest {
        model,
        messages,
        max_tokens,
        temperature: req.temperature.unwrap_or(defaults.temperature),
        top_p: req.top_p.unwrap_or(defaults.top_p),
        top_k: req.top_k.unwrap_or(defaults.top_k),
        min_p: req.min_p.unwrap_or(defaults.min_p),
        presence_penalty: req.presence_penalty.unwrap_or(defaults.presence_penalty),
        frequency_penalty: req.frequency_penalty.unwrap_or(0.0),
        repetition_penalty: req
            .repetition_penalty
            .unwrap_or(defaults.repetition_penalty),
        repeat_last_n: req.repeat_last_n.unwrap_or(defaults.repeat_last_n),
        enable_thinking,
        preserve_thinking: true,
        thinking_budget,
        reasoning_budget_message: req
            .reasoning_budget_message
            .clone()
            .or_else(|| defaults.reasoning_budget_message.clone()),
        ignore_eos: false,
        stop_sequences: Vec::new(),
        // Thinking is routed into a reasoning output item; tool-call markers
        // pass through always.
        emit_reasoning: enable_thinking,
        // Raw tool-call streaming is a chat-completions extension only.
        stream_tool_call_xml: false,
    })
}

fn responses_input_to_chat_turns(req: &ResponsesRequest) -> Result<Vec<ChatTurn>, String> {
    let mut messages = Vec::new();
    if let Some(instructions) = req.instructions.as_deref() {
        if !instructions.trim().is_empty() {
            messages.push(ChatTurn {
                role: "system".to_string(),
                content: instructions.to_string(),
            });
        }
    }

    match &req.input {
        ResponsesInput::Text(text) => {
            messages.push(ChatTurn {
                role: "user".to_string(),
                content: text.clone(),
            });
        }
        ResponsesInput::Items(items) => {
            for item in items {
                append_response_item_as_chat_turn(item, &mut messages)?;
            }
        }
    }

    let tool_defs = match &req.tools {
        Some(tools) => toolcall::parse_tool_defs(tools)?,
        None => Vec::new(),
    };
    let tool_choice = toolcall::parse_tool_choice(req.tool_choice.as_ref());
    if let Some(block) = toolcall::render_tools_system_block(&tool_defs, &tool_choice) {
        prepend_system_block(&mut messages, &block);
    }

    Ok(messages)
}

fn append_response_item_as_chat_turn(
    item: &ResponsesInputItem,
    messages: &mut Vec<ChatTurn>,
) -> Result<(), String> {
    if let Some(status) = item.status.as_deref() {
        if !status.is_empty() && status != "completed" {
            return Err(format!(
                "responses input item status {status:?} is not replayable"
            ));
        }
    }

    let item_type = item.item_type.as_deref().unwrap_or("message");
    match item_type {
        "message" => {
            let role = item.role.as_deref().unwrap_or("user").to_string();
            let content = match item.content.as_ref() {
                Some(value) => text_from_response_content(value, "content")?,
                None => String::new(),
            };
            messages.push(ChatTurn { role, content });
        }
        "reasoning" => {
            let mut reasoning = String::new();
            if let Some(summary) = item.summary.as_ref() {
                reasoning.push_str(&text_from_response_content(summary, "summary")?);
            }
            if let Some(content) = item.content.as_ref() {
                let content = text_from_response_content(content, "content")?;
                if !content.is_empty() {
                    if !reasoning.is_empty() {
                        reasoning.push('\n');
                    }
                    reasoning.push_str(&content);
                }
            }
            if !reasoning.trim().is_empty() {
                messages.push(ChatTurn {
                    role: "assistant".to_string(),
                    content: format!("<think>\n{}\n</think>", reasoning.trim()),
                });
            }
        }
        "function_call"
        | "custom_tool_call"
        | "local_shell_call"
        | "web_search_call"
        | "tool_search_call"
        | "image_generation_call" => {
            messages.push(ChatTurn {
                role: "assistant".to_string(),
                content: render_response_call_item(item, item_type)?,
            });
        }
        "function_call_output"
        | "custom_tool_call_output"
        | "local_shell_call_output"
        | "web_search_call_output"
        | "tool_search_output"
        | "tool_search_call_output"
        | "image_generation_call_output" => {
            let content = item
                .output
                .as_ref()
                .or(item.result.as_ref())
                .map(|value| raw_text_from_response_value(value, "output"))
                .transpose()?
                .unwrap_or_default();
            messages.push(ChatTurn {
                role: "tool".to_string(),
                content,
            });
        }
        "compaction" | "context_compaction" => {}
        other => {
            return Err(format!(
                "unsupported responses input item type {other:?}; qw35 accepts text, message, reasoning, and tool replay items"
            ));
        }
    }
    Ok(())
}

fn render_response_call_item(item: &ResponsesInputItem, item_type: &str) -> Result<String, String> {
    let name = if let Some(namespace) = item.namespace.as_deref() {
        if let Some(name) = item.name.as_deref() {
            format!("{namespace}{name}")
        } else {
            namespace.to_string()
        }
    } else {
        item.name
            .as_deref()
            .unwrap_or(match item_type {
                "local_shell_call" => "local_shell",
                "web_search_call" => "web_search",
                "tool_search_call" => "tool_search",
                "image_generation_call" => "image_generation",
                _ => "function",
            })
            .to_string()
    };
    let args = item
        .arguments
        .as_ref()
        .or(item.input.as_ref())
        .or(item.action.as_ref())
        .map(|value| raw_text_from_response_value(value, "arguments"))
        .transpose()?
        .unwrap_or_else(|| "{}".to_string());
    // Canonical Qwen 3.5 (Hermes) form — the same shape the model is trained
    // to emit, so replayed calls look like its own past output.
    Ok(toolcall::render_tool_call_block(&name, &args))
}

fn text_from_response_content(value: &serde_json::Value, field: &str) -> Result<String, String> {
    match value {
        serde_json::Value::Null => Ok(String::new()),
        serde_json::Value::String(text) => Ok(text.clone()),
        serde_json::Value::Array(parts) => {
            let mut out = String::new();
            for part in parts {
                let text = match part {
                    serde_json::Value::String(text) => text.clone(),
                    serde_json::Value::Object(obj) => {
                        let part_type = obj
                            .get("type")
                            .and_then(serde_json::Value::as_str)
                            .ok_or_else(|| format!("responses {field} part is missing type"))?;
                        if !is_text_part_type(part_type) {
                            return Err(format!(
                                "unsupported responses {field} part type {part_type:?}; qw35 accepts text-only requests"
                            ));
                        }
                        match obj.get("text") {
                            Some(serde_json::Value::String(text)) => text.clone(),
                            Some(serde_json::Value::Null) | None => String::new(),
                            _ => {
                                return Err(format!(
                                    "responses {field} part type {part_type:?} has non-string text"
                                ))
                            }
                        }
                    }
                    _ => {
                        return Err(format!(
                            "responses {field} must contain strings or typed text objects"
                        ))
                    }
                };
                if !out.is_empty() && !text.is_empty() {
                    out.push('\n');
                }
                out.push_str(&text);
            }
            Ok(out)
        }
        _ => Err(format!(
            "responses {field} must be a string, null, or typed text array"
        )),
    }
}

fn raw_text_from_response_value(value: &serde_json::Value, field: &str) -> Result<String, String> {
    match value {
        serde_json::Value::Null => Ok(String::new()),
        serde_json::Value::String(text) => Ok(text.clone()),
        serde_json::Value::Array(_) => text_from_response_content(value, field),
        other => Ok(other.to_string()),
    }
}

fn responses_enable_thinking(req: &ResponsesRequest, defaults: &GenerationDefaults) -> bool {
    // An explicit reasoning_effort implies thinking-on; otherwise fall back to
    // the server preset default (the Responses API has no `enable_thinking`).
    matches!(
        req.reasoning
            .as_ref()
            .and_then(|reasoning| reasoning.effort.as_deref()),
        Some("low" | "medium" | "high" | "xhigh")
    )
    .then_some(true)
    .unwrap_or(defaults.enable_thinking)
}

fn is_text_part_type(part_type: &str) -> bool {
    matches!(
        part_type,
        "text" | "input_text" | "output_text" | "summary_text" | "reasoning_text"
    )
}

pub fn parse_token_limit(flag: &str, value: i64) -> Result<TokenLimit, String> {
    if value == -1 {
        return Ok(TokenLimit::Context);
    }
    if value <= 0 {
        return Err(format!("{flag} must be greater than zero or -1"));
    }
    let value = u32::try_from(value).map_err(|_| format!("{flag} must be at most {}", u32::MAX))?;
    Ok(TokenLimit::Fixed(value))
}

#[derive(Debug, Serialize)]
struct ErrorBody {
    error: ErrorDetail,
}

#[derive(Debug, Serialize)]
struct ErrorDetail {
    code: String,
    message: String,
    #[serde(rename = "type")]
    error_type: &'static str,
}

// ── App state and router ───────────────────────────────────────────────────

pub async fn serve(
    host: &str,
    port: u16,
    engine: Arc<Engine>,
    defaults: GenerationDefaults,
) -> Result<(), String> {
    let addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|err| format!("failed to listen on {addr}: {err}"))?;
    serve_listener(listener, engine, defaults, None).await
}

pub async fn serve_listener(
    listener: tokio::net::TcpListener,
    engine: Arc<Engine>,
    defaults: GenerationDefaults,
    shutdown: Option<tokio::sync::oneshot::Receiver<()>>,
) -> Result<(), String> {
    let addr = listener
        .local_addr()
        .map_err(|err| format!("failed to read listener address: {err}"))?;
    eprintln!("qw35: listening on http://{addr}");

    let app = build_router(engine, defaults);
    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            match shutdown {
                // Tests drive shutdown via this oneshot; keep the graceful drain
                // path intact for them (no SIGINT involved).
                Some(rx) => {
                    let _ = rx.await;
                }
                // Real server: Ctrl+C must kill immediately, even mid
                // prefill/decode. Generation runs on a spawn_blocking thread with
                // no cancellation hook, so a graceful drain would hang until it
                // finishes. process::exit(130) (128 + SIGINT) terminates now; the
                // OS reclaims the Metal command queue / GPU buffers on exit.
                None => {
                    tokio::signal::ctrl_c().await.ok();
                    std::process::exit(130);
                }
            }
        })
        .await
        .map_err(|err| format!("server error: {err}"))
}

fn build_router(engine: Arc<Engine>, defaults: GenerationDefaults) -> Router {
    let state = AppState { engine, defaults };

    Router::new()
        .route("/health", get(health))
        .route("/props", get(props))
        .route("/v1/models", get(models))
        .route("/v1/models/{model}", get(model))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/responses", post(responses))
        .route("/v1/responses/input_tokens", post(responses_input_tokens))
        .layer(
            tower_http::cors::CorsLayer::new()
                .allow_origin(tower_http::cors::Any)
                .allow_headers([header::CONTENT_TYPE, header::AUTHORIZATION])
                .allow_methods([Method::GET, Method::POST, Method::OPTIONS]),
        )
        .with_state(state)
}

#[derive(Clone)]
struct AppState {
    engine: Arc<Engine>,
    defaults: GenerationDefaults,
}

// ── Handlers ───────────────────────────────────────────────────────────────

async fn health(State(state): State<AppState>) -> Json<serde_json::Value> {
    let engine = &state.engine;
    let summary = engine.metadata_summary();
    let plan = engine.graph_plan();

    let warmup = if let Some(report) = engine.warm_report() {
        serde_json::json!({
            "enabled": true,
            "mode": report.mode,
            "bytes": report.bytes,
            "mapped_bytes": report.mapped_bytes,
            "page_size": report.page_size,
            "touched_pages": report.touched_pages,
            "view_count": report.view_count,
            "checksum": report.checksum,
            "elapsed_ms": report.elapsed.as_millis(),
            "madvise_error": report.madvise_error,
        })
    } else {
        serde_json::json!({"enabled": false})
    };

    Json(serde_json::json!({
        "status": "ok",
        "model": engine.model_id(),
        "decoder_ready": engine.decoder_ready(),
        "test_responder": engine.test_responder(),
        "ctx_size": engine.ctx_size(),
        "prefill_chunk": engine.prefill_chunk(),
        "ffn": engine.ffn_label(),
        "mapped_bytes": summary.mapped_bytes,
        "architecture": summary.architecture,
        "model_name": summary.name,
        "graph": {
            "delta_layers": plan.delta_layers.len(),
            "attention_layers": plan.attention_layers.len(),
            "unsupported_tensor_types": plan.unsupported_tensor_types,
            "missing_tensor_count": plan.missing_tensors.len(),
            "execution_blockers": plan.execution_blockers,
        },
        "warmup": warmup,
    }))
}

/// llama.cpp-compatible `/props`: clients (e.g. pi/little-coder's startup
/// probe) read `default_generation_settings.n_ctx` to register the live
/// context window instead of a config-file guess.
async fn props(State(state): State<AppState>) -> Json<serde_json::Value> {
    let engine = &state.engine;
    Json(serde_json::json!({
        "default_generation_settings": {
            "n_ctx": engine.ctx_size(),
        },
        "total_slots": 1,
        "model_path": engine.model_path().display().to_string(),
    }))
}

async fn models(State(state): State<AppState>) -> Json<serde_json::Value> {
    let engine = &state.engine;
    Json(serde_json::json!({
        "object": "list",
        "data": [model_json(engine)]
    }))
}

async fn model(
    Path(model): Path<String>,
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, AppError> {
    let engine = &state.engine;
    if model != engine.model_id() {
        return Err(AppError::not_found(&format!(
            "model {model:?} is not available"
        )));
    }
    Ok(Json(model_json(engine)))
}

fn model_json(engine: &Engine) -> serde_json::Value {
    let summary = engine.metadata_summary();
    let plan = engine.graph_plan();

    serde_json::json!({
        "id": engine.model_id(),
        "object": "model",
        "created": engine.created(),
        "owned_by": "local",
        "metadata": {
            "architecture": summary.architecture,
            "name": summary.name,
            "size_label": summary.size_label,
            "block_count": summary.block_count,
            "context_length": summary.context_length,
            "embedding_length": summary.embedding_length,
            "vocab_size": summary.vocab_size,
            "tensor_count": summary.tensor_count,
            "mapped_bytes": summary.mapped_bytes,
            "tensor_data_pos": summary.tensor_data_pos,
            "tensor_type_counts": plan.tensor_type_counts,
            "unsupported_tensor_types": plan.unsupported_tensor_types,
            "execution_blockers": plan.execution_blockers,
        }
    })
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> Result<Response, AppError> {
    let engine = &state.engine;
    if !engine.decoder_ready() {
        return Err(AppError::inference_unavailable(
            &engine.inference_unavailable_message(),
        ));
    }

    let generate_req = into_generate_request(&req, engine.model_id(), &state.defaults)?;

    if req.stream {
        let include_usage = req
            .stream_options
            .as_ref()
            .map(|o| o.include_usage)
            .unwrap_or(false);
        stream_response(engine.clone(), &generate_req, include_usage).await
    } else {
        plain_response(engine, &generate_req).await
    }
}

async fn responses(
    State(state): State<AppState>,
    Json(req): Json<ResponsesRequest>,
) -> Result<Response, AppError> {
    let engine = &state.engine;
    if !engine.decoder_ready() {
        return Err(AppError::inference_unavailable(
            &engine.inference_unavailable_message(),
        ));
    }

    let generate_req = into_responses_generate_request(&req, engine.model_id(), &state.defaults)?;

    if req.stream {
        responses_stream_response(engine.clone(), req, generate_req).await
    } else {
        responses_plain_response(engine, &req, &generate_req).await
    }
}

async fn responses_input_tokens(
    State(state): State<AppState>,
    Json(req): Json<ResponsesRequest>,
) -> Result<Json<serde_json::Value>, AppError> {
    let engine = &state.engine;
    let generate_req = into_responses_generate_request(&req, engine.model_id(), &state.defaults)?;
    let input_tokens = engine.count_prompt_tokens(&generate_req)?;
    Ok(Json(serde_json::json!({
        "object": "response.input_tokens",
        "input_tokens": input_tokens,
    })))
}

// ── Non-streaming response ─────────────────────────────────────────────────

async fn plain_response(engine: &Engine, request: &GenerateRequest) -> Result<Response, AppError> {
    let generation = engine.generate(request)?;
    let parsed = toolcall::parse_assistant_text(
        &generation.text,
        request.emit_reasoning && request.enable_thinking,
    );
    let finish_reason = chat_finish_reason(generation.finish_reason, !parsed.tool_calls.is_empty());
    let id = response_id("chatcmpl");
    let timings = timings_json(&generation.timings);
    let body = serde_json::json!({
        "id": id,
        "object": "chat.completion",
        "created": now(),
        "model": engine.model_id(),
        "choices": [{
            "index": 0,
            "message": chat_message_json(&parsed),
            "logprobs": null,
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": generation.prompt_tokens,
            "completion_tokens": generation.completion_tokens,
            "total_tokens": generation.prompt_tokens.saturating_add(generation.completion_tokens),
            "prompt_tokens_details": {"cached_tokens": generation.cached_tokens, "audio_tokens": 0},
            "completion_tokens_details": {
                "reasoning_tokens": 0,
                "audio_tokens": 0,
                "accepted_prediction_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
        },
        "service_tier": "default",
        "system_fingerprint": null,
        "qw35_timings": timings,
    });
    Ok(Json(body).into_response())
}

fn chat_finish_reason(finish: FinishReason, has_tool_calls: bool) -> &'static str {
    match finish {
        FinishReason::Length => "length",
        FinishReason::Stop if has_tool_calls => "tool_calls",
        FinishReason::Stop => "stop",
    }
}

fn chat_message_json(parsed: &ParsedAssistantOutput) -> serde_json::Value {
    let content = if parsed.content.is_empty() && !parsed.tool_calls.is_empty() {
        serde_json::Value::Null
    } else {
        serde_json::Value::String(parsed.content.clone())
    };
    let mut message = serde_json::json!({
        "role": "assistant",
        "content": content,
        "refusal": null,
        "annotations": [],
    });
    if !parsed.reasoning.is_empty() {
        message["reasoning_content"] = serde_json::json!(parsed.reasoning);
    }
    if !parsed.tool_calls.is_empty() {
        let tool_calls: Vec<serde_json::Value> = parsed
            .tool_calls
            .iter()
            .map(|call| {
                serde_json::json!({
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                })
            })
            .collect();
        message["tool_calls"] = serde_json::json!(tool_calls);
    }
    message
}

// ── SSE streaming ──────────────────────────────────────────────────────────

async fn stream_response(
    engine: Arc<Engine>,
    request: &GenerateRequest,
    include_usage: bool,
) -> Result<Response, AppError> {
    let id = response_id("chatcmpl");
    let model = engine.model_id().to_string();
    let created = now();
    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(64);

    // Send initial role chunk
    let _ = tx
        .send(Ok(sse_event(&serde_json::json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "system_fingerprint": null,
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "logprobs": null, "finish_reason": null}],
        }))))
        .await;

    // Run blocking generation on a dedicated thread
    let engine_arc = engine;
    let tx2 = tx.clone();
    let id2 = id.clone();
    let model2 = model.clone();
    let request2 = request.clone();

    tokio::task::spawn_blocking(move || {
        let mut parser = toolcall::AssistantStreamParser::with_options(
            request2.emit_reasoning && request2.enable_thinking,
            request2.stream_tool_call_xml,
        );
        let mut emitter = ChatStreamEmitter {
            tx: &tx2,
            id: &id2,
            model: &model2,
            created,
            emitted_tool_calls: 0,
            stream_raw: request2.stream_tool_call_xml,
        };
        let result = engine_arc.generate_stream_with_progress(
            &request2,
            |chunk| emitter.send_events(parser.feed(chunk)),
            |processed, total| {
                // Prompt-processing progress as a choice-less chunk. OpenAI
                // clients ignore the empty choices and the qw35_prefill field;
                // our TUI maps it to the mascot's prefill percentage.
                let percent = if total > 0 {
                    processed as f64 / total as f64 * 100.0
                } else {
                    0.0
                };
                let _ = tx2.blocking_send(Ok(sse_event(&serde_json::json!({
                    "id": id2,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model2,
                    "choices": [],
                    "qw35_prefill": {
                        "processed": processed,
                        "total": total,
                        "percent": percent,
                    },
                }))));
            },
        );

        match result {
            Ok(generation) => {
                let _ = emitter.send_events(parser.finish());
                let finish_reason =
                    chat_finish_reason(generation.finish_reason, emitter.emitted_tool_calls > 0);
                let timings = timings_json(&generation.timings);
                let _ = tx2.blocking_send(Ok(sse_event(&serde_json::json!({
                    "id": id2,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model2,
                    "system_fingerprint": null,
                    "choices": [{"index": 0, "delta": {}, "logprobs": null, "finish_reason": finish_reason}],
                }))));
                if include_usage {
                    let _ = tx2.blocking_send(Ok(sse_event(&serde_json::json!({
                        "id": id2,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model2,
                        "choices": [],
                        "usage": {
                            "prompt_tokens": generation.prompt_tokens,
                            "completion_tokens": generation.completion_tokens,
                            "total_tokens": generation.prompt_tokens.saturating_add(generation.completion_tokens),
                            "prompt_tokens_details": {"cached_tokens": generation.cached_tokens, "audio_tokens": 0},
                            "completion_tokens_details": {
                                "reasoning_tokens": 0,
                                "audio_tokens": 0,
                                "accepted_prediction_tokens": 0,
                                "rejected_prediction_tokens": 0,
                            },
                        },
                        "qw35_timings": timings,
                    }))));
                }
                let _ = tx2.blocking_send(Ok(Event::default().data("[DONE]")));
            }
            Err(GenerateError::BadRequest(msg)) => {
                let _ = tx2.blocking_send(Ok(sse_event(&serde_json::json!({
                    "error": {"code": "bad_request", "message": msg, "type": "qw35_error"},
                }))));
            }
            Err(GenerateError::InferenceUnavailable(msg)) => {
                let _ = tx2.blocking_send(Ok(sse_event(&serde_json::json!({
                    "error": {"code": "inference_unavailable", "message": msg, "type": "qw35_error"},
                }))));
            }
        }
    });

    // Drop original tx so the stream ends when spawn_blocking finishes
    drop(tx);

    let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
    Ok(Sse::new(stream)
        .keep_alive(
            axum::response::sse::KeepAlive::new().interval(std::time::Duration::from_secs(15)),
        )
        .into_response())
}

/// Sends parser events as chat-completion SSE chunks, following the delta
/// discipline OpenAI clients depend on: the first delta for a tool-call index
/// carries `id` + `function.name` (and an empty arguments string), later
/// deltas carry argument fragments only, with the same stable index and no
/// new ids.
struct ChatStreamEmitter<'a> {
    tx: &'a mpsc::Sender<Result<Event, Infallible>>,
    id: &'a str,
    model: &'a str,
    created: u64,
    /// Live tool calls emitted so far: +1 on Begin, -1 when a raw-streamed
    /// block is demoted back to text. Drives the final finish_reason.
    emitted_tool_calls: usize,
    /// Raw tool-call streaming was requested (`stream_tool_call_xml`): the
    /// parser's `ToolCallName`/`ToolCallEnd`/`ToolCallDemoted` events go out
    /// as choice-less `qw35_tool_call` side-channel chunks (same house style
    /// as `qw35_prefill`; OpenAI clients ignore empty-choices chunks).
    stream_raw: bool,
}

impl ChatStreamEmitter<'_> {
    fn send_events(&mut self, events: Vec<AssistantEvent>) -> Result<(), String> {
        for event in events {
            let delta = match event {
                AssistantEvent::Content(text) => serde_json::json!({"content": text}),
                AssistantEvent::Reasoning(text) => {
                    serde_json::json!({"reasoning_content": text})
                }
                AssistantEvent::ToolCallBegin { index, id, name } => {
                    self.emitted_tool_calls += 1;
                    serde_json::json!({"tool_calls": [{
                        "index": index,
                        "id": id,
                        "type": "function",
                        "function": {"name": name, "arguments": ""},
                    }]})
                }
                AssistantEvent::ToolCallArgs { index, fragment } => {
                    serde_json::json!({"tool_calls": [{
                        "index": index,
                        "function": {"arguments": fragment},
                    }]})
                }
                AssistantEvent::ToolCallEnd { index, arguments } => {
                    if self.stream_raw {
                        // The streamed fragments were raw XML; deliver the
                        // authoritative parsed arguments out-of-band.
                        self.send_side_channel(serde_json::json!({
                            "event": "final",
                            "index": index,
                            "arguments": arguments,
                        }))?;
                    }
                    // Chat completions has no per-call end chunk; the final
                    // finish_reason covers it.
                    continue;
                }
                AssistantEvent::ToolCallName { index, name } => {
                    self.send_side_channel(serde_json::json!({
                        "event": "name",
                        "index": index,
                        "name": name,
                    }))?;
                    continue;
                }
                AssistantEvent::ToolCallDemoted { index } => {
                    // The demoted block's text follows as ordinary content/
                    // reasoning deltas.
                    self.emitted_tool_calls = self.emitted_tool_calls.saturating_sub(1);
                    self.send_side_channel(serde_json::json!({
                        "event": "demoted",
                        "index": index,
                    }))?;
                    continue;
                }
            };
            self.tx
                .blocking_send(Ok(sse_event(&serde_json::json!({
                    "id": self.id,
                    "object": "chat.completion.chunk",
                    "created": self.created,
                    "model": self.model,
                    "system_fingerprint": null,
                    "choices": [{"index": 0, "delta": delta, "logprobs": null, "finish_reason": null}],
                }))))
                .map_err(|err| format!("stream closed: {err}"))?;
        }
        Ok(())
    }

    fn send_side_channel(&self, payload: serde_json::Value) -> Result<(), String> {
        self.tx
            .blocking_send(Ok(sse_event(&serde_json::json!({
                "id": self.id,
                "object": "chat.completion.chunk",
                "created": self.created,
                "model": self.model,
                "choices": [],
                "qw35_tool_call": payload,
            }))))
            .map_err(|err| format!("stream closed: {err}"))
    }
}

// ── Responses API ─────────────────────────────────────────────────────────

async fn responses_plain_response(
    engine: &Engine,
    req: &ResponsesRequest,
    request: &GenerateRequest,
) -> Result<Response, AppError> {
    let generation = engine.generate(request)?;
    let parsed = toolcall::parse_assistant_text(
        &generation.text,
        request.emit_reasoning && request.enable_thinking,
    );
    let id = response_id("resp");
    let created = now();
    let output = responses_output_items(&parsed);
    let body = responses_final_json(req, engine.model_id(), &id, created, &generation, output);
    Ok(Json(body).into_response())
}

/// Builds the final `output` array from a parsed generation: an optional
/// reasoning item, the assistant message (unless the output is tool-calls
/// only), and one `function_call` item per call, with `arguments` as the
/// OpenAI-style JSON string.
fn responses_output_items(parsed: &ParsedAssistantOutput) -> Vec<serde_json::Value> {
    let mut output = Vec::new();
    if !parsed.reasoning.is_empty() {
        output.push(responses_reasoning_item_json(
            &response_id("rs"),
            "completed",
            &parsed.reasoning,
        ));
    }
    if !parsed.content.is_empty() || parsed.tool_calls.is_empty() {
        output.push(responses_message_item_json(
            &response_id("msg"),
            "completed",
            &parsed.content,
        ));
    }
    for call in &parsed.tool_calls {
        output.push(responses_function_call_item_json(
            &response_id("fc"),
            "completed",
            &call.id,
            &call.name,
            &call.arguments,
        ));
    }
    output
}

async fn responses_stream_response(
    engine: Arc<Engine>,
    req: ResponsesRequest,
    request: GenerateRequest,
) -> Result<Response, AppError> {
    let id = response_id("resp");
    let model = engine.model_id().to_string();
    let created = now();
    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(64);

    // All events are emitted from the blocking thread so every one of them
    // (including response.created) carries a monotonic sequence_number.
    tokio::task::spawn_blocking(move || {
        let mut state = ResponsesStreamState {
            sink: ResponsesEventSink { tx: &tx, seq: 0 },
            output_index: 0,
            open: None,
            items: Vec::new(),
        };
        let _ = state.sink.send(
            "response.created",
            serde_json::json!({
                "type": "response.created",
                "response": responses_in_progress_json(&req, &model, &id, created),
            }),
        );
        let _ = state.sink.send(
            "response.in_progress",
            serde_json::json!({
                "type": "response.in_progress",
                "response": responses_in_progress_json(&req, &model, &id, created),
            }),
        );

        let mut parser =
            toolcall::AssistantStreamParser::new(request.emit_reasoning && request.enable_thinking);
        let result = engine.generate_stream(&request, |chunk| {
            for event in parser.feed(chunk) {
                state.handle(event)?;
            }
            Ok(())
        });

        match result {
            Ok(generation) => {
                for event in parser.finish() {
                    if state.handle(event).is_err() {
                        return;
                    }
                }
                if state.close_open().is_err() {
                    return;
                }
                let event_type = match generation.finish_reason {
                    FinishReason::Length => "response.incomplete",
                    FinishReason::Stop => "response.completed",
                };
                let body = responses_final_json(
                    &req,
                    &model,
                    &id,
                    created,
                    &generation,
                    state.items.clone(),
                );
                let _ = state.sink.send(
                    event_type,
                    serde_json::json!({"type": event_type, "response": body}),
                );
            }
            Err(err) => {
                let message = match err {
                    GenerateError::BadRequest(msg) | GenerateError::InferenceUnavailable(msg) => {
                        msg
                    }
                };
                let _ = state.sink.send(
                    "response.failed",
                    serde_json::json!({
                        "type": "response.failed",
                        "response": responses_failed_json(&req, &model, &id, created, &message),
                    }),
                );
            }
        }
    });

    let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
    Ok(Sse::new(stream)
        .keep_alive(
            axum::response::sse::KeepAlive::new().interval(std::time::Duration::from_secs(15)),
        )
        .into_response())
}

/// Stamps a monotonic `sequence_number` on every responses SSE event, which
/// codex uses to order the stream.
struct ResponsesEventSink<'a> {
    tx: &'a mpsc::Sender<Result<Event, Infallible>>,
    seq: u64,
}

impl ResponsesEventSink<'_> {
    fn send(
        &mut self,
        event_type: &'static str,
        mut data: serde_json::Value,
    ) -> Result<(), String> {
        data["sequence_number"] = serde_json::json!(self.seq);
        self.seq += 1;
        self.tx
            .blocking_send(Ok(responses_sse_event(event_type, &data)))
            .map_err(|err| format!("stream closed: {err}"))
    }
}

/// Item-lazy responses stream: output items open only when the parser
/// produces something for them — a message for content, a reasoning item for
/// thinking, a function_call item per tool call — and every finalized item is
/// collected so the terminal event carries the complete `output` array.
struct ResponsesStreamState<'a> {
    sink: ResponsesEventSink<'a>,
    output_index: usize,
    open: Option<OpenResponsesItem>,
    items: Vec<serde_json::Value>,
}

enum OpenResponsesItem {
    Message {
        id: String,
        text: String,
    },
    Reasoning {
        id: String,
        text: String,
    },
    FunctionCall {
        id: String,
        call_id: String,
        name: String,
        arguments: String,
    },
}

impl ResponsesStreamState<'_> {
    fn handle(&mut self, event: AssistantEvent) -> Result<(), String> {
        match event {
            AssistantEvent::Content(text) => {
                if !matches!(self.open, Some(OpenResponsesItem::Message { .. })) {
                    self.close_open()?;
                    let item_id = response_id("msg");
                    self.sink.send(
                        "response.output_item.added",
                        serde_json::json!({
                            "type": "response.output_item.added",
                            "output_index": self.output_index,
                            "item": responses_message_item_json(&item_id, "in_progress", ""),
                        }),
                    )?;
                    self.sink.send(
                        "response.content_part.added",
                        serde_json::json!({
                            "type": "response.content_part.added",
                            "item_id": item_id,
                            "output_index": self.output_index,
                            "content_index": 0,
                            "part": {"type": "output_text", "text": "", "annotations": []},
                        }),
                    )?;
                    self.open = Some(OpenResponsesItem::Message {
                        id: item_id,
                        text: String::new(),
                    });
                }
                let Some(OpenResponsesItem::Message { id, text: buffer }) = &mut self.open else {
                    unreachable!()
                };
                buffer.push_str(&text);
                let item_id = id.clone();
                self.sink.send(
                    "response.output_text.delta",
                    serde_json::json!({
                        "type": "response.output_text.delta",
                        "item_id": item_id,
                        "output_index": self.output_index,
                        "content_index": 0,
                        "delta": text,
                    }),
                )
            }
            AssistantEvent::Reasoning(text) => {
                if !matches!(self.open, Some(OpenResponsesItem::Reasoning { .. })) {
                    self.close_open()?;
                    let item_id = response_id("rs");
                    self.sink.send(
                        "response.output_item.added",
                        serde_json::json!({
                            "type": "response.output_item.added",
                            "output_index": self.output_index,
                            "item": responses_reasoning_item_json(&item_id, "in_progress", ""),
                        }),
                    )?;
                    self.open = Some(OpenResponsesItem::Reasoning {
                        id: item_id,
                        text: String::new(),
                    });
                }
                let Some(OpenResponsesItem::Reasoning { id, text: buffer }) = &mut self.open else {
                    unreachable!()
                };
                buffer.push_str(&text);
                let item_id = id.clone();
                self.sink.send(
                    "response.reasoning_text.delta",
                    serde_json::json!({
                        "type": "response.reasoning_text.delta",
                        "item_id": item_id,
                        "output_index": self.output_index,
                        "content_index": 0,
                        "delta": text,
                    }),
                )
            }
            AssistantEvent::ToolCallBegin {
                id: call_id, name, ..
            } => {
                self.close_open()?;
                let item_id = response_id("fc");
                self.sink.send(
                    "response.output_item.added",
                    serde_json::json!({
                        "type": "response.output_item.added",
                        "output_index": self.output_index,
                        "item": responses_function_call_item_json(
                            &item_id,
                            "in_progress",
                            &call_id,
                            &name,
                            "",
                        ),
                    }),
                )?;
                self.open = Some(OpenResponsesItem::FunctionCall {
                    id: item_id,
                    call_id,
                    name,
                    arguments: String::new(),
                });
                Ok(())
            }
            AssistantEvent::ToolCallArgs { fragment, .. } => {
                let Some(OpenResponsesItem::FunctionCall { id, arguments, .. }) = &mut self.open
                else {
                    return Ok(());
                };
                arguments.push_str(&fragment);
                let item_id = id.clone();
                self.sink.send(
                    "response.function_call_arguments.delta",
                    serde_json::json!({
                        "type": "response.function_call_arguments.delta",
                        "item_id": item_id,
                        "output_index": self.output_index,
                        "delta": fragment,
                    }),
                )
            }
            AssistantEvent::ToolCallEnd { arguments, .. } => {
                if let Some(OpenResponsesItem::FunctionCall {
                    arguments: buffer, ..
                }) = &mut self.open
                {
                    *buffer = arguments;
                }
                self.close_open()
            }
            // Raw-streaming-only events; the responses path never enables
            // stream_tool_call_xml, so these cannot occur here.
            AssistantEvent::ToolCallName { .. } | AssistantEvent::ToolCallDemoted { .. } => Ok(()),
        }
    }

    /// Emits the done events for the open item and collects it into `items`.
    fn close_open(&mut self) -> Result<(), String> {
        let Some(open) = self.open.take() else {
            return Ok(());
        };
        let item = match open {
            OpenResponsesItem::Message { id, text } => {
                self.sink.send(
                    "response.output_text.done",
                    serde_json::json!({
                        "type": "response.output_text.done",
                        "item_id": id,
                        "output_index": self.output_index,
                        "content_index": 0,
                        "text": text,
                    }),
                )?;
                self.sink.send(
                    "response.content_part.done",
                    serde_json::json!({
                        "type": "response.content_part.done",
                        "item_id": id,
                        "output_index": self.output_index,
                        "content_index": 0,
                        "part": {"type": "output_text", "text": text, "annotations": []},
                    }),
                )?;
                responses_message_item_json(&id, "completed", &text)
            }
            OpenResponsesItem::Reasoning { id, text } => {
                self.sink.send(
                    "response.reasoning_text.done",
                    serde_json::json!({
                        "type": "response.reasoning_text.done",
                        "item_id": id,
                        "output_index": self.output_index,
                        "content_index": 0,
                        "text": text,
                    }),
                )?;
                responses_reasoning_item_json(&id, "completed", &text)
            }
            OpenResponsesItem::FunctionCall {
                id,
                call_id,
                name,
                arguments,
            } => {
                let arguments = if arguments.trim().is_empty() {
                    "{}".to_string()
                } else {
                    arguments
                };
                self.sink.send(
                    "response.function_call_arguments.done",
                    serde_json::json!({
                        "type": "response.function_call_arguments.done",
                        "item_id": id,
                        "output_index": self.output_index,
                        "arguments": arguments,
                    }),
                )?;
                responses_function_call_item_json(&id, "completed", &call_id, &name, &arguments)
            }
        };
        self.sink.send(
            "response.output_item.done",
            serde_json::json!({
                "type": "response.output_item.done",
                "output_index": self.output_index,
                "item": item.clone(),
            }),
        )?;
        self.items.push(item);
        self.output_index += 1;
        Ok(())
    }
}

fn responses_sse_event(event_type: &'static str, data: &serde_json::Value) -> Event {
    Event::default()
        .event(event_type)
        .data(serde_json::to_string(data).unwrap())
}

fn responses_in_progress_json(
    req: &ResponsesRequest,
    model: &str,
    id: &str,
    created: u64,
) -> serde_json::Value {
    responses_base_json(
        req,
        model,
        id,
        created,
        "in_progress",
        Vec::new(),
        None,
        None,
    )
}

/// Terminal response body: status and incomplete_details derive from the
/// finish reason ("incomplete"/"max_output_tokens" when the token limit cut
/// the generation short).
fn responses_final_json(
    req: &ResponsesRequest,
    model: &str,
    id: &str,
    created: u64,
    generation: &crate::model::Generation,
    output: Vec<serde_json::Value>,
) -> serde_json::Value {
    let status = match generation.finish_reason {
        FinishReason::Length => "incomplete",
        FinishReason::Stop => "completed",
    };
    let mut body = responses_base_json(
        req,
        model,
        id,
        created,
        status,
        output,
        Some(responses_usage_json(generation)),
        Some(created),
    );
    if generation.finish_reason == FinishReason::Length {
        body["incomplete_details"] = serde_json::json!({"reason": "max_output_tokens"});
    }
    body
}

fn responses_failed_json(
    req: &ResponsesRequest,
    model: &str,
    id: &str,
    created: u64,
    message: &str,
) -> serde_json::Value {
    let mut body = responses_base_json(
        req,
        model,
        id,
        created,
        "failed",
        Vec::new(),
        None,
        Some(created),
    );
    body["error"] = serde_json::json!({
        "code": "server_error",
        "message": message,
    });
    body
}

#[allow(clippy::too_many_arguments)]
fn responses_base_json(
    req: &ResponsesRequest,
    model: &str,
    id: &str,
    created: u64,
    status: &str,
    output: Vec<serde_json::Value>,
    usage: Option<serde_json::Value>,
    completed_at: Option<u64>,
) -> serde_json::Value {
    let mut body = serde_json::json!({
        "id": id,
        "object": "response",
        "created_at": created,
        "status": status,
        "error": null,
        "incomplete_details": null,
        "instructions": req.instructions.clone(),
        "max_output_tokens": req.max_output_tokens,
        "model": model,
        "output": output,
        "parallel_tool_calls": req.parallel_tool_calls.unwrap_or(true),
        "previous_response_id": null,
        "reasoning": {
            "effort": req.reasoning.as_ref().and_then(|reasoning| reasoning.effort.clone()),
            "summary": req.reasoning.as_ref().and_then(|reasoning| reasoning.summary.clone()),
        },
        "store": req.store.unwrap_or(false),
        "temperature": req.temperature.unwrap_or(1.0),
        "text": req.text.clone().unwrap_or_else(|| serde_json::json!({"format": {"type": "text"}})),
        "tool_choice": req.tool_choice.clone().unwrap_or_else(|| serde_json::json!("auto")),
        "tools": req.tools.clone().unwrap_or_else(|| serde_json::json!([])),
        "top_p": req.top_p.unwrap_or(1.0),
        "truncation": "disabled",
        "usage": usage,
        "user": null,
        "metadata": req.metadata.clone().unwrap_or_else(|| serde_json::json!({})),
    });
    if let Some(completed_at) = completed_at {
        body["completed_at"] = serde_json::json!(completed_at);
    }
    body
}

fn responses_message_item_json(id: &str, status: &str, text: &str) -> serde_json::Value {
    serde_json::json!({
        "id": id,
        "type": "message",
        "status": status,
        "role": "assistant",
        "content": [{
            "type": "output_text",
            "text": text,
            "annotations": [],
        }],
    })
}

fn responses_reasoning_item_json(id: &str, status: &str, text: &str) -> serde_json::Value {
    serde_json::json!({
        "id": id,
        "type": "reasoning",
        "status": status,
        "summary": [],
        "content": [{
            "type": "reasoning_text",
            "text": text,
        }],
    })
}

fn responses_function_call_item_json(
    id: &str,
    status: &str,
    call_id: &str,
    name: &str,
    arguments: &str,
) -> serde_json::Value {
    serde_json::json!({
        "id": id,
        "type": "function_call",
        "status": status,
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
    })
}

fn responses_usage_json(generation: &crate::model::Generation) -> serde_json::Value {
    let total_tokens = generation
        .prompt_tokens
        .saturating_add(generation.completion_tokens);
    serde_json::json!({
        "input_tokens": generation.prompt_tokens,
        "input_tokens_details": {"cached_tokens": generation.cached_tokens},
        "output_tokens": generation.completion_tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": total_tokens,
    })
}

fn sse_event(data: &serde_json::Value) -> Event {
    Event::default().data(serde_json::to_string(data).unwrap())
}

fn timings_json(timings: &crate::model::GenerationTimings) -> serde_json::Value {
    serde_json::json!({
        "total_ms": duration_ms(timings.total_duration),
        "render_ms": duration_ms(timings.render_duration),
        "tokenize_ms": duration_ms(timings.tokenize_duration),
        "runtime_lock_ms": duration_ms(timings.runtime_lock_duration),
        "reset_ms": duration_ms(timings.reset_duration),
        "prompt_eval_ms": duration_ms(timings.prompt_eval_duration),
        "prompt_eval_count": timings.prompt_eval_count,
        "prompt_eval_tps": rate(timings.prompt_eval_count, timings.prompt_eval_duration),
        "cached_prompt_tokens": timings.cached_prompt_tokens,
        "session_path": timings.session_path.as_str(),
        "prefill_chunk": timings.prefill_chunk,
        "prefill_path": timings.prefill_path.as_str(),
        "eval_ms": duration_ms(timings.eval_duration),
        "eval_count": timings.eval_count,
        "eval_tps": rate(timings.eval_count, timings.eval_duration),
        "decode_eval_ms": duration_ms(timings.decode_eval_duration),
        "sample_ms": duration_ms(timings.sample_duration),
        "detokenize_ms": duration_ms(timings.detokenize_duration),
        "stream_callback_ms": duration_ms(timings.stream_callback_duration),
        "first_token_ms": timings.first_token_duration.map(duration_ms),
    })
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn rate(count: u32, duration: Duration) -> f64 {
    let seconds = duration.as_secs_f64();
    if count == 0 || seconds <= 0.0 {
        0.0
    } else {
        f64::from(count) / seconds
    }
}

// ── Error type ─────────────────────────────────────────────────────────────

struct AppError {
    status: StatusCode,
    error: ErrorBody,
}

impl AppError {
    fn bad_request(msg: &str) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            error: ErrorBody {
                error: ErrorDetail {
                    code: "bad_request".into(),
                    message: msg.into(),
                    error_type: "qw35_error",
                },
            },
        }
    }

    fn inference_unavailable(msg: &str) -> Self {
        Self {
            status: StatusCode::NOT_IMPLEMENTED,
            error: ErrorBody {
                error: ErrorDetail {
                    code: "inference_unavailable".into(),
                    message: msg.into(),
                    error_type: "qw35_error",
                },
            },
        }
    }

    fn not_found(msg: &str) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            error: ErrorBody {
                error: ErrorDetail {
                    code: "not_found".into(),
                    message: msg.into(),
                    error_type: "qw35_error",
                },
            },
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let mut resp = Json(self.error).into_response();
        *resp.status_mut() = self.status;
        resp
    }
}

impl From<String> for AppError {
    fn from(msg: String) -> Self {
        AppError::bad_request(&msg)
    }
}

impl From<GenerateError> for AppError {
    fn from(err: GenerateError) -> Self {
        match err {
            GenerateError::BadRequest(msg) => AppError::bad_request(&msg),
            GenerateError::InferenceUnavailable(msg) => AppError::inference_unavailable(&msg),
        }
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────

fn response_id(prefix: &str) -> String {
    static NEXT_ID: AtomicU64 = AtomicU64::new(1);
    let seq = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{prefix}_{nanos:x}{seq:x}")
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
#[path = "tests/server_thinking_budget.rs"]
mod thinking_budget_tests;

#[cfg(test)]
#[path = "tests/server_mode.rs"]
mod mode_tests;

#[cfg(test)]
#[path = "tests/server_enable_thinking.rs"]
mod enable_thinking_precedence_tests;
