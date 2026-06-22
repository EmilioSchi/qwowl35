//! Filters the model's raw token stream into user-visible text: suppresses
//! `<think>`…`</think>` reasoning (unless `emit_reasoning`), trims leading
//! newlines, and re-injects tool-call marker tags.

use crate::tokenizer::QwenTokenizer;

#[derive(Debug)]
pub(super) struct GeneratedTextFilter {
    trim_leading_newlines: bool,
    suppress_thinking: bool,
    emit_reasoning: bool,
}

impl Default for GeneratedTextFilter {
    fn default() -> Self {
        Self::new(false)
    }
}

impl GeneratedTextFilter {
    pub(super) fn new(emit_reasoning: bool) -> Self {
        Self {
            trim_leading_newlines: true,
            suppress_thinking: false,
            emit_reasoning,
        }
    }

    pub(super) fn visible_token_delta(
        &mut self,
        token_id: u32,
        delta: &str,
        tokenizer: &QwenTokenizer,
    ) -> String {
        let spec = &tokenizer.spec;
        if Some(token_id) == spec.tool_call_token_id
            || Some(token_id) == spec.end_tool_call_token_id
        {
            // Tool-call markers decode to empty text as special tokens;
            // re-inject the literal tag so the server-side parser sees the
            // boundary. Inside a suppressed thinking block they stay hidden.
            let tag = tokenizer
                .token_text(token_id)
                .unwrap_or_default()
                .to_string();
            return self.visible_text_delta(&tag);
        }
        if Some(token_id) == spec.think_token_id {
            if self.emit_reasoning {
                return "<think>".to_string();
            }
            self.suppress_thinking = true;
            self.trim_leading_newlines = true;
            return String::new();
        }
        if Some(token_id) == spec.end_think_token_id {
            if self.emit_reasoning {
                return "</think>".to_string();
            }
            self.suppress_thinking = false;
            self.trim_leading_newlines = true;
            return String::new();
        }
        self.visible_text_delta(delta)
    }

    pub(super) fn visible_finish_delta(&mut self, delta: &str) -> String {
        self.visible_text_delta(delta)
    }

    fn visible_text_delta(&mut self, delta: &str) -> String {
        if self.emit_reasoning {
            // The caller's stream parser routes thinking vs. content and owns
            // whitespace cleanup, so the raw text passes through untouched.
            return delta.to_string();
        }
        let mut text = delta;
        if self.suppress_thinking {
            let Some((_, rest)) = text.split_once("</think>") else {
                return String::new();
            };
            self.suppress_thinking = false;
            self.trim_leading_newlines = true;
            text = rest;
        }

        if self.trim_leading_newlines {
            text = text.trim_start_matches(['\r', '\n']);
            loop {
                if let Some(rest) = text.strip_prefix("</think>") {
                    text = rest.trim_start_matches(['\r', '\n']);
                    continue;
                }
                if let Some(rest) = text.strip_prefix("<think>") {
                    if let Some((_, after_think)) = rest.split_once("</think>") {
                        text = after_think.trim_start_matches(['\r', '\n']);
                        continue;
                    }
                    self.suppress_thinking = true;
                    return String::new();
                }
                break;
            }
            if !text.is_empty() {
                self.trim_leading_newlines = false;
            }
        }

        text.to_string()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn generated_text_filter_emit_reasoning_passes_markers_through() {
        let mut filter = super::GeneratedTextFilter::new(true);
        assert_eq!(
            filter.visible_text_delta("\nthinking text"),
            "\nthinking text"
        );
        assert_eq!(
            filter.visible_text_delta("</think>\n\nanswer"),
            "</think>\n\nanswer"
        );
    }

    #[test]
    fn generated_text_filter_handles_leading_newlines_and_think_blocks() {
        let mut filter = super::GeneratedTextFilter::default();
        assert_eq!(filter.visible_text_delta("\n"), "");
        assert!(filter.trim_leading_newlines);
        assert_eq!(filter.visible_text_delta("\r\n\nHello"), "Hello");
        assert!(!filter.trim_leading_newlines);
        assert_eq!(filter.visible_text_delta("\nkept"), "\nkept");

        let mut filter = super::GeneratedTextFilter::default();
        assert_eq!(filter.visible_text_delta(" Hello"), " Hello");
        assert!(!filter.trim_leading_newlines);

        let mut filter = super::GeneratedTextFilter::default();
        assert_eq!(filter.visible_text_delta("\n<think>\nprivate"), "");
        assert!(filter.suppress_thinking);
        assert_eq!(filter.visible_text_delta(" reasoning"), "");
        assert_eq!(filter.visible_text_delta("</think>\nVisible"), "Visible");
        assert!(!filter.suppress_thinking);

        let mut filter = super::GeneratedTextFilter::default();
        assert_eq!(
            filter.visible_text_delta("<think>private</think>\nVisible"),
            "Visible"
        );

        let mut filter = super::GeneratedTextFilter::default();
        assert_eq!(filter.visible_text_delta("</think>\nVisible"), "Visible");
    }
}
