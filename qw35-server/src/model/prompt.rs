//! Qwen3.5 chat-prompt rendering (system merge, role turns, thinking markers).

use super::ChatTurn;

pub(super) fn render_qwen35_chat_prompt(
    messages: &[ChatTurn],
    enable_thinking: bool,
    preserve_thinking: bool,
) -> String {
    render_qwen35_chat_prompt_with_boundary(messages, enable_thinking, preserve_thinking).0
}

/// Renders the chat prompt and returns the byte length of its stable prefix:
/// everything before the appended generation header. Future renders of an
/// extended conversation reproduce the prefix verbatim, while the header
/// (`<|im_start|>assistant\n<think>...`) is not reproduced for historical
/// assistant turns — so the prefix is the position worth checkpointing for
/// the session cache.
pub(super) fn render_qwen35_chat_prompt_with_boundary(
    messages: &[ChatTurn],
    enable_thinking: bool,
    preserve_thinking: bool,
) -> (String, usize) {
    let mut out = String::new();
    let mut num_sys = 0usize;
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
    }

    let stable_len = out.len();
    out.push_str("<|im_start|>assistant\n");
    if enable_thinking {
        out.push_str("<think>\n");
    } else {
        out.push_str("<think>\n\n</think>\n\n");
    }
    (out, stable_len)
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
mod tests {
    use super::ChatTurn;

    #[test]
    fn renders_qwen35_text_chat_prompt() {
        let prompt = super::render_qwen35_chat_prompt(
            &[
                ChatTurn {
                    role: "system".to_string(),
                    content: "Be concise.".to_string(),
                },
                ChatTurn {
                    role: "developer".to_string(),
                    content: "Prefer plain text.".to_string(),
                },
                ChatTurn {
                    role: "user".to_string(),
                    content: "Say hi.".to_string(),
                },
            ],
            false,
            false,
        );

        assert_eq!(
            prompt,
            "<|im_start|>system\nBe concise.\nPrefer plain text.<|im_end|>\n<|im_start|>user\nSay hi.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }

    #[test]
    fn strips_prior_thinking_unless_preserved() {
        let messages = [ChatTurn {
            role: "assistant".to_string(),
            content: "<think>\nprivate\n</think>\n\nvisible".to_string(),
        }];

        let stripped = super::render_qwen35_chat_prompt(&messages, true, false);
        assert!(stripped.contains("\nvisible<|im_end|>"));
        assert!(!stripped.contains("private"));

        let preserved = super::render_qwen35_chat_prompt(&messages, true, true);
        assert!(preserved.contains("private"));
        assert!(preserved.contains("visible"));

        let unclosed = [ChatTurn {
            role: "assistant".to_string(),
            content: "<think>\nstill private".to_string(),
        }];
        let stripped_unclosed = super::render_qwen35_chat_prompt(&unclosed, true, false);
        assert!(!stripped_unclosed.contains("still private"));
    }

    #[test]
    fn renders_generation_thinking_control_markers() {
        let messages = [ChatTurn {
            role: "user".to_string(),
            content: "Say hi.".to_string(),
        }];

        let thinking = super::render_qwen35_chat_prompt(&messages, true, false);
        let non_thinking = super::render_qwen35_chat_prompt(&messages, false, false);

        assert!(thinking.ends_with("<|im_start|>assistant\n<think>\n"));
        assert!(non_thinking.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    }
}
