// Native reranker engine (Qwen3-Reranker-0.6B family).
//
// A second, much smaller model served by the same process as the principal
// Qwowl3.5-9B chat engine, loaded only on request (`--reranker-model`). It is
// deliberately a SEPARATE engine: its own mmap, its own hparams/validation
// (dense all-attention Qwen3, classification head), its own tokenizer vocab,
// and one dedicated MetalRuntime — nothing here touches the 9B code paths.
// What it shares with the 9B is the generic infrastructure: the GGUF loader,
// the tokenizer implementation, and the Metal runtime/kernels (parameterized
// by hparams: full_attention_interval=1 routes every layer through the
// attention path, attn_gate=0 selects the ungated kernels, n_cls_out=2 wires
// cls.output.weight as the output head).
//
// Scoring (llama.cpp rank-converted GGUF): prefill the templated
// (query, document) prompt, run the output head on the last position over the
// 2-row classification head, and map the two logits through
// sigmoid(z_yes - z_no) — equivalent to softmax P("yes") over the two label
// rows the conversion extracted from the original lm_head.

use crate::loader::MappedGguf;
use crate::graph::QwenHparams;
use crate::metal;
use crate::model::GenerateError;
use crate::tokenizer::QwenTokenizer;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

pub mod prompt;
pub use prompt::{render_rerank_prompt, template_drift_warning, DEFAULT_INSTRUCTION};

pub const DEFAULT_RERANKER_CTX: u32 = 2048;
/// Hard cap on documents per request (each document is a full prefill; the
/// cap bounds worst-case latency and memory of one call).
pub const MAX_RERANK_DOCUMENTS: usize = 256;

#[derive(Debug, Clone)]
pub struct RerankerConfig {
    pub model_path: PathBuf,
    pub ctx_size: u32,
    pub prefill_chunk: u32,
    pub kv_cache_type: metal::KvCacheType,
    pub verbose: bool,
}

/// Timing block returned with every rerank call (surfaced as `qw35_timings`
/// so benchmarks can read prefill throughput like they do for chat).
#[derive(Debug, Clone, Default)]
pub struct RerankTimings {
    pub total_duration: Duration,
    pub render_duration: Duration,
    pub tokenize_duration: Duration,
    pub prompt_eval_duration: Duration,
    pub prompt_eval_count: u32,
    pub docs: u32,
    /// Prompt tokens served from the shared-prefix reuse (not re-evaluated):
    /// the system+instruct+query prefix is evaluated once per request, and a
    /// repeated query reuses it across requests too.
    pub prefix_reused_tokens: u32,
    pub per_doc_duration: Vec<Duration>,
}

struct RerankSession {
    runtime: metal::MetalRuntime,
    /// Tokens whose evaluation has been submitted (KV rows 0..len written).
    /// The KV cache is positional and every layer is attention, so a new
    /// prompt reuses the longest common token prefix and simply overwrites
    /// rows above it — no recurrent state, no checkpoints.
    evaluated: Vec<u32>,
}

pub struct RerankEngine {
    // Declared before `gguf` so the runtime drops first (it reads the mmap).
    session: Mutex<RerankSession>,
    gguf: MappedGguf,
    model_path: PathBuf,
    model_name: String,
    ctx_size: u32,
    prefill_chunk: u32,
    tokenizer: QwenTokenizer,
    /// Index of the "yes" logit in the classification head output, from
    /// `qwen3.classifier.output_labels` (the conversion may order labels
    /// either way).
    yes_index: usize,
    verbose: bool,
}

impl RerankEngine {
    pub fn open(config: RerankerConfig) -> Result<Self, String> {
        metal::require_metal_device()?;
        metal::verify_native_qwen_kernels()?;
        if config.ctx_size == 0 {
            return Err("--reranker-ctx must be greater than zero".to_string());
        }

        let gguf = MappedGguf::open(&config.model_path)?;
        validate_reranker(&gguf)?;
        let hparams = read_reranker_hparams(&gguf)?;
        let yes_index = yes_label_index(&gguf)?;
        // `pre = "qwen2"` on the stock reranker GGUF; the hand-rolled
        // pretokenizer implements that regex family (qwen35 shares it).
        let tokenizer = QwenTokenizer::load_with_pre(&gguf, &["qwen2", "qwen35"])?;

        if let Some(template) = gguf.metadata_string("tokenizer.chat_template.rerank") {
            if let Some(warning) = template_drift_warning(template) {
                eprintln!("qw35: warning: {warning}");
            }
        }

        let model_ctx = gguf.metadata_u32("qwen3.context_length").unwrap_or(u32::MAX);
        if config.ctx_size > model_ctx {
            return Err(format!(
                "--reranker-ctx {} exceeds the reranker's trained context {model_ctx}",
                config.ctx_size
            ));
        }

        let vocab_size: u32 = tokenizer.spec.vocab_size.try_into().map_err(|_| {
            "reranker tokenizer vocabulary is too large for the native Metal runtime".to_string()
        })?;

        // Activation capture (QW35_CAPTURE_ACT_OUT) hooks only the
        // single-token eval path, so force the scalar prefill when capturing
        // reranker act-stats for the AWQ cook.
        let capture = std::env::var("QW35_CAPTURE_ACT_OUT")
            .map(|v| !v.is_empty())
            .unwrap_or(false);
        let prefill_chunk = if capture {
            eprintln!("qw35: reranker act-stats capture ON; forcing --prefill-chunk 1 (slow)");
            1
        } else {
            config.prefill_chunk.max(1)
        };

        let runtime = metal::MetalRuntime::new(
            &gguf,
            &hparams,
            config.ctx_size,
            vocab_size,
            prefill_chunk,
            config.kv_cache_type,
            0, // full attention
            0,
        )?;

        let model_name = gguf
            .metadata_string("general.name")
            .unwrap_or("qwen3-reranker")
            .to_string();

        Ok(Self {
            session: Mutex::new(RerankSession {
                runtime,
                evaluated: Vec::new(),
            }),
            gguf,
            model_path: config.model_path,
            model_name,
            ctx_size: config.ctx_size,
            prefill_chunk,
            tokenizer,
            yes_index,
            verbose: config.verbose,
        })
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    pub fn ctx_size(&self) -> u32 {
        self.ctx_size
    }

    /// FFN codec label for the startup summary and /health ("gguf" for the
    /// raw q8_0 file, "gf4-unified" once the cooked model is served).
    pub fn ffn_label(&self) -> String {
        let blocks = self.gguf.metadata_u32("qwen3.block_count").unwrap_or(0);
        let mut gf4 = 0;
        for layer in 0..blocks {
            let name = format!("blk.{layer}.ffn_gate.weight");
            if self.gguf.tensor(&name).map(|t| t.type_id) == Some(100) {
                gf4 += 1;
            }
        }
        if gf4 > 0 {
            "gf4-unified".to_string()
        } else {
            "gguf".to_string()
        }
    }

    /// Score `documents` against `query`: one relevance score in (0, 1) per
    /// document, in input order.
    pub fn score(
        &self,
        query: &str,
        documents: &[String],
        instruction: Option<&str>,
    ) -> Result<(Vec<f32>, RerankTimings), GenerateError> {
        let total_start = Instant::now();
        if query.trim().is_empty() {
            return Err(GenerateError::BadRequest(
                "query must not be empty".to_string(),
            ));
        }
        if documents.is_empty() {
            return Err(GenerateError::BadRequest(
                "documents must contain at least one entry".to_string(),
            ));
        }
        if documents.len() > MAX_RERANK_DOCUMENTS {
            return Err(GenerateError::BadRequest(format!(
                "documents must contain at most {MAX_RERANK_DOCUMENTS} entries"
            )));
        }

        let mut session = self
            .session
            .lock()
            .map_err(|_| GenerateError::InferenceUnavailable("reranker runtime lock is poisoned".to_string()))?;

        let mut timings = RerankTimings {
            docs: documents.len() as u32,
            per_doc_duration: Vec::with_capacity(documents.len()),
            ..RerankTimings::default()
        };
        let mut scores = Vec::with_capacity(documents.len());

        for document in documents {
            let doc_start = Instant::now();

            let render_start = Instant::now();
            let (prompt, tokens) = self.render_and_fit(query, document, instruction, &mut timings)?;
            let _ = prompt;
            timings.render_duration += render_start.elapsed();

            // Longest common token prefix with the evaluated lineage; the last
            // prompt token must always be (re-)evaluated so the output head
            // runs on it.
            let lcp = tokens
                .iter()
                .zip(session.evaluated.iter())
                .take_while(|(a, b)| a == b)
                .count();
            let start = lcp.min(tokens.len() - 1);
            timings.prefix_reused_tokens += start as u32;
            session.evaluated.truncate(start);

            let eval_start = Instant::now();
            self.prefill(&mut session.runtime, &tokens, start)
                .map_err(GenerateError::InferenceUnavailable)?;
            let logits = session
                .runtime
                .copy_logits()
                .map_err(GenerateError::InferenceUnavailable)?;
            timings.prompt_eval_duration += eval_start.elapsed();
            timings.prompt_eval_count += (tokens.len() - start) as u32;
            session.evaluated = tokens;

            if logits.len() != 2 {
                return Err(GenerateError::InferenceUnavailable(format!(
                    "reranker head returned {} logits, expected 2",
                    logits.len()
                )));
            }
            let z_yes = f64::from(logits[self.yes_index]);
            let z_no = f64::from(logits[1 - self.yes_index]);
            let score = 1.0 / (1.0 + (-(z_yes - z_no)).exp());
            scores.push(score as f32);
            timings.per_doc_duration.push(doc_start.elapsed());
        }

        timings.total_duration = total_start.elapsed();
        if self.verbose {
            eprintln!(
                "qw35: rerank docs={} eval_tokens={} reused={} total={:.1}ms",
                timings.docs,
                timings.prompt_eval_count,
                timings.prefix_reused_tokens,
                timings.total_duration.as_secs_f64() * 1000.0
            );
        }
        Ok((scores, timings))
    }

    /// Render the prompt and tokenize it, truncating the DOCUMENT (never the
    /// template or query) when the whole prompt exceeds the context.
    fn render_and_fit(
        &self,
        query: &str,
        document: &str,
        instruction: Option<&str>,
        timings: &mut RerankTimings,
    ) -> Result<(String, Vec<u32>), GenerateError> {
        let ctx = self.ctx_size as usize;
        let mut doc: &str = document;
        for _ in 0..4 {
            let prompt = render_rerank_prompt(query, doc, instruction);
            let tokenize_start = Instant::now();
            let tokens = self
                .tokenizer
                .encode(&prompt, true)
                .map_err(|err| GenerateError::BadRequest(format!("tokenization failed: {err}")))?;
            timings.tokenize_duration += tokenize_start.elapsed();
            if tokens.len() <= ctx {
                return Ok((prompt, tokens));
            }
            // Shrink the document proportionally (with slack) and re-check.
            let ratio = ctx as f64 / tokens.len() as f64 * 0.9;
            let mut cut = (doc.len() as f64 * ratio) as usize;
            while cut > 0 && !doc.is_char_boundary(cut) {
                cut -= 1;
            }
            if cut == 0 {
                break;
            }
            doc = &doc[..cut];
        }
        Err(GenerateError::BadRequest(format!(
            "document does not fit the reranker context ({ctx} tokens) even after truncation"
        )))
    }

    /// Evaluate prompt positions [start, len) with the output head (full
    /// logits) on the final token. Mirrors the shape of the 9B engine's
    /// prefill loop, kept private here so the two engines stay independent.
    fn prefill(
        &self,
        runtime: &mut metal::MetalRuntime,
        tokens: &[u32],
        start: usize,
    ) -> Result<(), String> {
        let last = tokens.len() - 1;
        if self.prefill_chunk <= 1 || tokens.len() - start == 1 {
            for (pos, &token) in tokens.iter().enumerate().skip(start) {
                let mode = if pos == last {
                    metal::LogitsMode::Full
                } else {
                    metal::LogitsMode::None
                };
                runtime.eval_token(token, pos as u32, mode)?;
            }
        } else {
            let chunk_len = self.prefill_chunk as usize;
            for (chunk_idx, chunk) in tokens[start..].chunks(chunk_len).enumerate() {
                let pos0 = start + chunk_idx * chunk_len;
                let mode = if pos0 + chunk.len() == tokens.len() {
                    metal::LogitsMode::Full
                } else {
                    metal::LogitsMode::None
                };
                runtime.eval_prefill_chunk(chunk, pos0 as u32, mode)?;
            }
        }
        Ok(())
    }
}

/// Structural validation of a rank-converted dense Qwen3 reranker GGUF.
/// Deliberately separate from the 9B's `validate_model` (which stays pinned
/// to the exact Qwen3.5-9B fingerprint): this one is structural, so a future
/// Qwen3-Reranker-4B conversion loads without code changes.
pub fn validate_reranker(gguf: &MappedGguf) -> Result<(), String> {
    let arch = gguf
        .metadata_string("general.architecture")
        .ok_or("missing general.architecture")?;
    if arch == "qwen35" {
        return Err(
            "this is a Qwen3.5 chat model, not a reranker; pass it via --model instead".to_string(),
        );
    }
    if arch != "qwen3" {
        return Err(format!(
            "unsupported reranker architecture {arch:?}: expected qwen3"
        ));
    }
    let pooling = gguf
        .metadata_u32("qwen3.pooling_type")
        .ok_or("missing qwen3.pooling_type: not a rank-converted reranker GGUF")?;
    if pooling != 4 {
        return Err(format!(
            "qwen3.pooling_type is {pooling}, expected 4 (RANK); not a reranker conversion"
        ));
    }

    let labels: Vec<String> = gguf
        .metadata_array_string_iter("qwen3.classifier.output_labels")
        .ok_or("missing qwen3.classifier.output_labels")?
        .map(str::to_string)
        .collect();
    let mut sorted = labels.clone();
    sorted.sort();
    if sorted != ["no", "yes"] {
        return Err(format!(
            "unexpected classifier labels {labels:?}: expected [\"yes\", \"no\"]"
        ));
    }

    let emb = gguf
        .metadata_u32("qwen3.embedding_length")
        .ok_or("missing qwen3.embedding_length")? as u64;
    let cls = gguf
        .tensor("cls.output.weight")
        .ok_or("missing cls.output.weight (classification head)")?;
    if cls.dims.first().copied() != Some(emb) || cls.dims.get(1).copied() != Some(2) {
        return Err(format!(
            "cls.output.weight has dims {:?}, expected [{emb}, 2]",
            cls.dims
        ));
    }

    for name in ["token_embd.weight", "output_norm.weight"] {
        if gguf.tensor(name).is_none() {
            return Err(format!("missing required tensor {name}"));
        }
    }

    let blocks = gguf
        .metadata_u32("qwen3.block_count")
        .ok_or("missing qwen3.block_count")?;
    let mut missing = Vec::new();
    for layer in 0..blocks {
        for suffix in [
            "attn_norm.weight",
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_q_norm.weight",
            "attn_k_norm.weight",
            "attn_output.weight",
            "ffn_norm.weight",
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_down.weight",
        ] {
            let name = format!("blk.{layer}.{suffix}");
            if gguf.tensor(&name).is_none() {
                missing.push(name);
            }
        }
    }
    if !missing.is_empty() {
        return Err(format!(
            "{} expected reranker tensors are missing (first: {})",
            missing.len(),
            missing[0]
        ));
    }
    Ok(())
}

/// Map the reranker's GGUF metadata onto the shared hparams struct: every
/// layer is a full-attention layer (interval 1), attention is ungated, the
/// output head is the 2-row classification head, and the SSM dims are inert
/// 1-stubs (delta layer count is zero, so no SSM kernel ever runs).
pub fn read_reranker_hparams(gguf: &MappedGguf) -> Result<QwenHparams, String> {
    let block_count = gguf
        .metadata_u32("qwen3.block_count")
        .ok_or("missing qwen3.block_count")?;
    let key_length = gguf
        .metadata_u32("qwen3.attention.key_length")
        .ok_or("missing qwen3.attention.key_length")?;
    Ok(QwenHparams {
        block_count,
        transformer_layers: block_count,
        nextn_predict_layers: 0,
        embedding_length: gguf
            .metadata_u32("qwen3.embedding_length")
            .ok_or("missing qwen3.embedding_length")?,
        feed_forward_length: gguf
            .metadata_u32("qwen3.feed_forward_length")
            .ok_or("missing qwen3.feed_forward_length")?,
        attention_heads: gguf
            .metadata_u32("qwen3.attention.head_count")
            .ok_or("missing qwen3.attention.head_count")?,
        attention_kv_heads: gguf
            .metadata_u32("qwen3.attention.head_count_kv")
            .ok_or("missing qwen3.attention.head_count_kv")?,
        attention_key_length: key_length,
        attention_value_length: gguf
            .metadata_u32("qwen3.attention.value_length")
            .unwrap_or(key_length),
        // Dense Qwen3 rotates the full head dim with plain NeoX RoPE: one
        // global frequency ladder, no position sections.
        rope_dimension_count: gguf
            .metadata_u32("qwen3.rope.dimension_count")
            .unwrap_or(key_length),
        rope_dimension_sections: vec![0, 0, 0, 0],
        rope_freq_base: gguf
            .metadata_f32("qwen3.rope.freq_base")
            .unwrap_or(1_000_000.0),
        rms_epsilon: gguf
            .metadata_f32("qwen3.attention.layer_norm_rms_epsilon")
            .unwrap_or(1.0e-6),
        ssm_conv_kernel: 1,
        ssm_state_size: 1,
        ssm_group_count: 1,
        ssm_time_step_rank: 1,
        ssm_inner_size: 1,
        full_attention_interval: 1,
        attn_gate: 0,
        n_cls_out: 2,
    })
}

/// Index of the "yes" row in the classification head output, from the label
/// order the conversion recorded.
pub fn yes_label_index(gguf: &MappedGguf) -> Result<usize, String> {
    let labels: Vec<String> = gguf
        .metadata_array_string_iter("qwen3.classifier.output_labels")
        .ok_or("missing qwen3.classifier.output_labels")?
        .map(str::to_string)
        .collect();
    labels
        .iter()
        .position(|label| label == "yes")
        .ok_or_else(|| format!("classifier labels {labels:?} do not contain \"yes\""))
}

#[cfg(test)]
#[path = "../tests/reranker.rs"]
mod tests;
