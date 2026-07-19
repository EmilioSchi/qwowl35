use super::{render_rerank_prompt, template_drift_warning, DEFAULT_INSTRUCTION};

/// The template embedded in the rank-converted GGUF
/// (`tokenizer.chat_template.rerank`), byte for byte. The renderer must match
/// it exactly — the model was trained against this framing.
const EMBEDDED_TEMPLATE: &str = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n<Instruct>: Given a web search query, retrieve relevant passages that answer the query\n<Query>: {query}\n<Document>: {document}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

#[test]
fn default_render_matches_the_embedded_template() {
    let rendered = render_rerank_prompt("{query}", "{document}", None);
    assert_eq!(rendered, EMBEDDED_TEMPLATE);
}

#[test]
fn render_substitutes_query_and_document() {
    let rendered = render_rerank_prompt("what is rust", "Rust is a language.", None);
    assert!(rendered.contains("<Query>: what is rust\n"));
    assert!(rendered.contains("<Document>: Rust is a language.<|im_end|>"));
    assert!(rendered.contains(&format!("<Instruct>: {DEFAULT_INSTRUCTION}\n")));
}

#[test]
fn render_ends_with_the_empty_think_block() {
    let rendered = render_rerank_prompt("q", "d", None);
    assert!(rendered.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    // No BOS or trailing content: the next position is the scoring position.
    assert!(rendered.starts_with("<|im_start|>system\n"));
}

#[test]
fn custom_instruction_replaces_the_default() {
    let rendered = render_rerank_prompt("q", "d", Some("Find code relevant to the task"));
    assert!(rendered.contains("<Instruct>: Find code relevant to the task\n"));
    assert!(!rendered.contains(DEFAULT_INSTRUCTION));
}

#[test]
fn empty_or_blank_instruction_falls_back_to_the_default() {
    for instruction in [Some(""), Some("   "), None] {
        let rendered = render_rerank_prompt("q", "d", instruction);
        assert!(rendered.contains(&format!("<Instruct>: {DEFAULT_INSTRUCTION}\n")));
    }
}

#[test]
fn drift_warning_fires_only_on_a_changed_template() {
    assert!(template_drift_warning(EMBEDDED_TEMPLATE).is_none());
    assert!(template_drift_warning("something else").is_some());
}
