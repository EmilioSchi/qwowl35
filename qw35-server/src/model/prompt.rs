//! Qwen3.5 chat-prompt rendering (system merge, role turns, thinking markers).

use super::ChatTurn;

pub(super) fn render_qwen35_chat_prompt(
    messages: &[ChatTurn],
    enable_thinking: bool,
    preserve_thinking: bool,
) -> String {
    render_qwen35_chat_prompt_with_boundary(messages, enable_thinking, preserve_thinking).0
}

/// Renders the chat prompt and returns two byte boundaries:
/// 1. `stable_len`: everything before the appended generation header (the
///    session-cache checkpoint position — see below).
/// 2. `preamble_len`: everything through the FIRST user turn (system block +
///    the initial user/task message). This is the prefix that must stay
///    attended under decode-time sliding-window attention, so the model never
///    "forgets" the tool-call format (system prompt) or the task (user turn);
///    the engine uses it as the attention-sink floor.
///
/// Future renders of an extended conversation reproduce the stable prefix
/// verbatim, while the header (`<|im_start|>assistant\n<think>...`) is not
/// reproduced for historical assistant turns.
pub(super) fn render_qwen35_chat_prompt_with_boundary(
    messages: &[ChatTurn],
    enable_thinking: bool,
    preserve_thinking: bool,
) -> (String, usize) {
    let (prompt, stable_len, _preamble_len) =
        render_qwen35_chat_prompt_with_boundaries(messages, enable_thinking, preserve_thinking);
    (prompt, stable_len)
}

pub(super) fn render_qwen35_chat_prompt_with_boundaries(
    messages: &[ChatTurn],
    enable_thinking: bool,
    preserve_thinking: bool,
) -> (String, usize, usize) {
    let mut out = String::new();
    let mut num_sys = 0usize;
    let mut preamble_len = 0usize;
    let mut merged_system = String::new();

    if let Some(first) = messages.first() {
        if is_system_role(&first.role) {
            merged_system.push_str(first.content.trim());
            num_sys = 1;
            if messages
                .get(1)
                .map(|msg| is_system_role(&msg.role))
                .unwrap_or(false)
            {
                if !merged_system.is_empty() {
                    merged_system.push('\n');
                }
                merged_system.push_str(messages[1].content.trim());
                num_sys = 2;
            }
        }
    }

    if !merged_system.is_empty() {
        out.push_str("<|im_start|>system\n");
        out.push_str(&merged_system);
        out.push_str("<|im_end|>\n");
    }

    for (idx, message) in messages.iter().enumerate() {
        if idx < num_sys || is_system_role(&message.role) {
            continue;
        }
        let content = message.content.trim();
        match message.role.as_str() {
            "user" => {
                out.push_str("<|im_start|>user\n");
                out.push_str(content);
                out.push_str("<|im_end|>\n");
            }
            "assistant" => {
                out.push_str("<|im_start|>assistant\n");
                if preserve_thinking {
                    out.push_str(content);
                } else {
                    out.push_str(strip_thinking_content(content).trim_start());
                }
                out.push_str("<|im_end|>\n");
            }
            "tool" => {
                out.push_str("<|im_start|>user\n<tool_response>\n");
                out.push_str(content);
                out.push_str("\n</tool_response><|im_end|>\n");
            }
            _ => {}
        }
        // Preamble = system block + first user/tool turn. Captured the first
        // time a user-side turn is written, so the sliding-window sink can pin
        // both the tool-call format (system) and the task (first user turn).
        if preamble_len == 0 && matches!(message.role.as_str(), "user" | "tool") {
            preamble_len = out.len();
        }
    }

    let stable_len = out.len();
    out.push_str("<|im_start|>assistant\n");
    if enable_thinking {
        out.push_str("<think>\n");
    } else {
        out.push_str("<think>\n\n</think>\n\n");
    }
    (out, stable_len, preamble_len)
}

fn is_system_role(role: &str) -> bool {
    role == "system" || role == "developer"
}

fn strip_thinking_content(content: &str) -> &str {
    let content = content.trim_start();
    if let Some(rest) = content.strip_prefix("</think>") {
        return rest;
    }
    if let Some(end) = content.find("</think>") {
        return &content[end + "</think>".len()..];
    }
    if content.starts_with("<think>") {
        return "";
    }
    content
}

#[cfg(test)]
#[path = "../tests/model_prompt.rs"]
mod tests;
