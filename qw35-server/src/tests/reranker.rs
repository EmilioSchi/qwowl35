use super::*;
use crate::loader::MappedGguf;

/// Reranker GGUF used by the real_reranker_* tests. QW35_RERANKER_MODEL
/// overrides the default raw q8_0 conversion so the same suite can target the
/// cooked Qwowl3-Reranker-0.6B.gguf.
fn reranker_model_path() -> String {
    std::env::var("QW35_RERANKER_MODEL")
        .unwrap_or_else(|_| ".gguf/qwen3-reranker-0.6b-q8_0.gguf".to_string())
}

fn open_reranker_gguf() -> MappedGguf {
    MappedGguf::open(reranker_model_path())
        .unwrap_or_else(|err| panic!("failed to map reranker GGUF: {err}"))
}

#[cfg(target_os = "macos")]
fn open_real_reranker() -> RerankEngine {
    RerankEngine::open(RerankerConfig {
        model_path: reranker_model_path().into(),
        ctx_size: DEFAULT_RERANKER_CTX,
        prefill_chunk: 32,
        kv_cache_type: crate::metal::KvCacheType::Q8_0,
        verbose: false,
    })
    .unwrap_or_else(|err| panic!("failed to open reranker engine: {err}"))
}

#[test]
#[ignore = "maps the real reranker GGUF and checks validation + hparams mapping"]
fn real_reranker_gguf_validates_and_maps_hparams() {
    let gguf = open_reranker_gguf();
    validate_reranker(&gguf).expect("reranker GGUF should validate");

    let hparams = read_reranker_hparams(&gguf).expect("hparams should map");
    assert_eq!(hparams.full_attention_interval, 1);
    assert_eq!(hparams.attn_gate, 0);
    assert_eq!(hparams.n_cls_out, 2);
    assert_eq!(hparams.transformer_layers, hparams.block_count);
    assert_eq!(hparams.rope_dimension_sections, vec![0, 0, 0, 0]);
    // Dense Qwen3 rotates the full head dim.
    assert_eq!(hparams.rope_dimension_count, hparams.attention_key_length);
    // The interval-1 layer typing must classify every layer as attention.
    for layer in 0..hparams.transformer_layers {
        assert_eq!((layer + 1) % hparams.full_attention_interval, 0);
    }

    let yes = yes_label_index(&gguf).expect("labels should contain yes");
    assert!(yes < 2);
}

#[test]
#[ignore = "maps the real 9B GGUF and checks the reranker validator rejects it"]
fn reranker_validation_rejects_the_9b_chat_model() {
    let path = std::env::var("QW35_REAL_MODEL")
        .unwrap_or_else(|_| ".gguf/Qwowl3.5-9B.gguf".to_string());
    let gguf = MappedGguf::open(path).expect("failed to map 9B GGUF");
    let err = validate_reranker(&gguf).expect_err("the 9B chat model must be rejected");
    assert!(err.contains("not a reranker"), "unexpected error: {err}");
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "loads the real reranker GGUF and scores documents on native Metal"]
fn real_reranker_scores_relevant_document_highest() {
    let engine = open_real_reranker();
    let query = "What is the capital of France?";
    let documents = vec![
        "The mitochondria is the powerhouse of the cell.".to_string(),
        "Paris is the capital and largest city of France.".to_string(),
        "Rust's borrow checker enforces memory safety at compile time.".to_string(),
    ];
    let (scores, timings) = engine
        .score(query, &documents, None)
        .expect("scoring should succeed");

    assert_eq!(scores.len(), 3);
    for &score in &scores {
        assert!(score > 0.0 && score < 1.0, "score out of range: {score}");
    }
    assert!(
        scores[1] > scores[0] && scores[1] > scores[2],
        "relevant document should win: {scores:?}"
    );
    assert_eq!(timings.docs, 3);
    assert_eq!(timings.per_doc_duration.len(), 3);
    assert!(timings.prompt_eval_count > 0);
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "loads the real reranker GGUF and checks shared-prefix reuse across documents"]
fn real_reranker_reuses_the_query_prefix_across_documents() {
    let engine = open_real_reranker();
    let query = "How does the borrow checker work?";
    let documents = vec![
        "The borrow checker tracks ownership and lifetimes of references.".to_string(),
        "Bananas are rich in potassium and easy to digest.".to_string(),
    ];
    let (first_scores, first) = engine.score(query, &documents, None).unwrap();
    // Documents after the first share the system+instruct+query prefix.
    assert!(
        first.prefix_reused_tokens > 0,
        "expected prefix reuse within a request: {first:?}"
    );

    // A repeated request reuses the prefix across requests too, and scoring
    // must be deterministic.
    let (second_scores, second) = engine.score(query, &documents, None).unwrap();
    assert_eq!(first_scores, second_scores);
    assert!(second.prefix_reused_tokens >= first.prefix_reused_tokens);
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "loads the real reranker GGUF and exercises request validation"]
fn real_reranker_rejects_bad_requests() {
    let engine = open_real_reranker();
    let doc = vec!["some document".to_string()];

    let err = engine.score("", &doc, None).unwrap_err();
    assert!(matches!(err, crate::model::GenerateError::BadRequest(_)));

    let err = engine.score("query", &[], None).unwrap_err();
    assert!(matches!(err, crate::model::GenerateError::BadRequest(_)));

    let too_many = vec!["d".to_string(); MAX_RERANK_DOCUMENTS + 1];
    let err = engine.score("query", &too_many, None).unwrap_err();
    assert!(matches!(err, crate::model::GenerateError::BadRequest(_)));
}

#[test]
#[ignore = "diagnostic: prints reranker token ids for tokenizer-parity comparison vs llama.cpp"]
fn real_reranker_tokenization_dump() {
    let gguf = open_reranker_gguf();
    let tokenizer =
        crate::tokenizer::QwenTokenizer::load_with_pre(&gguf, &["qwen2", "qwen35"]).unwrap();
    let cases = [
        ("What is the capital of France?", "Paris is the capital and largest city of France."),
        ("什么是量子纠缠?", "量子纠缠是两个或多个粒子之间的非经典关联，测量其中一个会立即影响另一个的状态。"),
        ("who's coming to the 3:30pm stand-up? it's re-scheduled", "Reminder: the stand-up moved to 3:30pm today; Ana, Luis and I'll join, Marta can't."),
    ];
    for (query, document) in cases {
        let prompt = render_rerank_prompt(query, document, None);
        let ids = tokenizer.encode(&prompt, true).unwrap();
        println!("QUERY {query:?}\nIDS {ids:?}\n");
    }
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "loads the real reranker GGUF and checks oversized documents are truncated, not fatal"]
fn real_reranker_truncates_oversized_documents() {
    let engine = open_real_reranker();
    // ~40k chars of prose-like text: far beyond the 2048-token context, must
    // be truncated to fit rather than erroring out.
    let huge = "The quick brown fox jumps over the lazy dog. ".repeat(900);
    let (scores, _) = engine
        .score("fox jumping", &[huge], None)
        .expect("oversized document should be truncated and scored");
    assert_eq!(scores.len(), 1);
    assert!(scores[0] > 0.0 && scores[0] < 1.0);
}
