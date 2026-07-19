// Qwen3-Reranker prompt rendering.
//
// The rank-converted GGUF embeds the exact template as
// `tokenizer.chat_template.rerank` (with `{query}`/`{document}` placeholders
// and the default instruction inlined). The renderer below is a hand-written
// mirror of it — same philosophy as `model/prompt.rs` for the chat template —
// and `template_drift_warning` compares the two at engine load so an upstream
// re-conversion with a changed template is flagged instead of silently
// mis-scoring.

/// Instruction baked into the model's embedded rerank template.
pub const DEFAULT_INSTRUCTION: &str =
    "Given a web search query, retrieve relevant passages that answer the query";

/// Render one (query, document) scoring prompt. The prompt ends right after
/// the empty `<think>` block: the next-position logits over the yes/no
/// classification head are the relevance signal. No BOS (the model's
/// `add_bos_token` is false); `instruction` of `None` or empty uses the
/// model's default.
pub fn render_rerank_prompt(query: &str, document: &str, instruction: Option<&str>) -> String {
    let instruction = match instruction {
        Some(text) if !text.trim().is_empty() => text,
        _ => DEFAULT_INSTRUCTION,
    };
    format!(
        "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
}

/// Compare the hand-written renderer against the template embedded in the
/// GGUF. Returns a warning message when they differ (scores may be degraded
/// because the model was trained against its own template).
pub fn template_drift_warning(embedded_template: &str) -> Option<String> {
    let rendered = render_rerank_prompt("{query}", "{document}", None);
    if rendered == embedded_template {
        None
    } else {
        Some(
            "reranker GGUF embeds a rerank chat template that differs from the built-in \
             renderer; scores may be degraded"
                .to_string(),
        )
    }
}

#[cfg(test)]
#[path = "../tests/reranker_prompt.rs"]
mod tests;
